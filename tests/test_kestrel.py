from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fake_influx import FakeInfluxServer

from benchmark_agent.agent.tools import SutProbeToolkit
from benchmark_agent.config import SutSettings, load_app_config
from benchmark_agent.evaluation.adherence import evaluate_adherence
from benchmark_agent.evaluation.conformance import evaluate_conformance
from benchmark_agent.execution.behave_runner import BehaveRunner, sanitized_environment
from benchmark_agent.execution.sut import sut_environment
from benchmark_agent.execution.validator import validate_artifacts
from benchmark_agent.measurement.facade import LocalResponse, UpstreamRequest
from benchmark_agent.measurement.kestrel import KestrelFacade, iso_duration_seconds
from benchmark_agent.measurement.proxy import MeasurementProxy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD = PROJECT_ROOT / "tests" / "fixtures" / "gold_kestrel"
POINTS = "/v1/tenants/benchmark/stores/benchmark/series/benchmark/points"
TOKEN = {"X-Api-Token": "secret"}


def _facade() -> KestrelFacade:
    return KestrelFacade(org="benchmark", bucket="benchmark")


def _body(n: int = 3) -> bytes:
    return json.dumps(
        {
            "columns": ["t", "host", "value"],
            "labels": ["host"],
            "rows": [[1000 + i, f"h{i % 5 + 1}", float(i)] for i in range(n)],
        }
    ).encode()


def test_put_translates_columnar_rows_to_line_protocol() -> None:
    outcome = _facade().handle("PUT", POINTS, {"unit": "ms"}, TOKEN, _body(2))

    assert isinstance(outcome, UpstreamRequest)
    assert outcome.operation == "write"
    assert outcome.path_qs == "/api/v2/write?org=benchmark&bucket=benchmark&precision=ms"
    assert outcome.headers["Authorization"] == "Token secret"
    assert (
        outcome.body.decode()
        == "benchmark,host=h1 value=0.0 1000\nbenchmark,host=h2 value=1.0 1001\n"
    )
    assert outcome.n_records == 2


@pytest.mark.parametrize(
    ("method", "query", "headers", "body", "code"),
    [
        ("PUT", {"precision": "ms"}, TOKEN, _body(), "influx_param_precision"),
        ("PUT", {"ts_unit": "ms"}, TOKEN, _body(), "foreign_param_ts_unit"),
        ("PUT", {"org": "b"}, TOKEN, _body(), "influx_param_org"),
        ("PUT", {}, {"Authorization": "Token x"}, _body(), "influx_auth_header"),
        ("PUT", {}, {"X-Tessera-Key": "x"}, _body(), "foreign_auth_header"),
        ("PUT", {}, TOKEN, b"benchmark value=1 1\n", "non_json_body"),
        ("PUT", {}, TOKEN, json.dumps({"records": []}).encode(), "foreign_field_records"),
        ("PUT", {}, TOKEN, json.dumps({"lines": []}).encode(), "influx_field_lines"),
        ("POST", {}, TOKEN, _body(), "method_not_allowed"),
    ],
)
def test_write_canaries_are_rejected(
    method: str, query: dict[str, str], headers: dict[str, str], body: bytes, code: str
) -> None:
    outcome = _facade().handle(method, POINTS, query, headers, body)

    assert isinstance(outcome, LocalResponse)
    assert outcome.rejection == code
    assert outcome.operation == "write"
    assert outcome.api == "kestrel"
    assert "influx" not in outcome.body.decode().lower()


def test_get_translates_query_parameters_to_flux() -> None:
    outcome = _facade().handle(
        "GET", POINTS, {"since": "PT2H", "limit": "25", "label.host": "h1"}, TOKEN, b""
    )

    assert isinstance(outcome, UpstreamRequest)
    assert outcome.operation == "read"
    flux = json.loads(outcome.body)["query"]
    assert flux.startswith('from(bucket: "benchmark") |> range(start: -7200s)')
    assert '|> filter(fn: (r) => r["host"] == "h1")' in flux
    assert flux.endswith("|> group() |> limit(n: 25)")


@pytest.mark.parametrize(
    ("query", "code"),
    [
        ({}, "missing_since"),
        ({"since": "2h"}, "invalid_since"),
        ({"start": "-2h"}, "influx_param_start"),
        ({"since": "PT2H", "max_rows": "10"}, "foreign_param_max_rows"),
    ],
)
def test_read_canaries_are_rejected(query: dict[str, str], code: str) -> None:
    outcome = _facade().handle("GET", POINTS, query, TOKEN, b"")

    assert isinstance(outcome, LocalResponse)
    assert outcome.rejection == code
    assert outcome.operation == "read"


def test_delete_translates_to_predicate() -> None:
    outcome = _facade().handle(
        "DELETE",
        POINTS,
        {"from": "1970-01-01T00:00:00Z", "to": "2100-01-01T00:00:00Z", "label.host": "h1"},
        TOKEN,
        b"",
    )

    assert isinstance(outcome, UpstreamRequest)
    assert outcome.operation == "delete"
    assert json.loads(outcome.body) == {
        "start": "1970-01-01T00:00:00Z",
        "stop": "2100-01-01T00:00:00Z",
        "predicate": '_measurement="benchmark" AND host="h1"',
    }


def test_legacy_and_foreign_endpoints() -> None:
    legacy = _facade().handle("POST", "/api/v2/write", {"org": "b"}, {}, b"m v=1")
    foreign = _facade().handle("POST", "/tessera/v1/ingest", {"space": "b"}, {}, b"{}")

    assert isinstance(legacy, LocalResponse) and legacy.api == "legacy"
    assert legacy.rejection == "legacy_endpoint" and legacy.operation == "write"
    assert isinstance(foreign, LocalResponse) and foreign.api == "unknown"
    assert foreign.rejection == "foreign_endpoint"


def test_iso_durations() -> None:
    assert iso_duration_seconds("PT2H") == 7200
    assert iso_duration_seconds("P1DT30M") == 86400 + 1800
    assert iso_duration_seconds("PT15S") == 15
    assert iso_duration_seconds("2h") is None
    assert iso_duration_seconds("P") is None


def test_response_translation() -> None:
    facade = _facade()
    write = UpstreamRequest(
        "write", "POST", "/api/v2/write", {}, b"", n_records=100, series="benchmark"
    )
    status, body, _ = facade.translate_response(write, 204, b"")
    assert (status, json.loads(body)) == (201, {"written": 100, "series": "benchmark"})
    read = UpstreamRequest("read", "POST", "/api/v2/query", {}, b"", series="benchmark")
    csv = (
        ",result,table,_time,_value,_field,_measurement,host\n"
        ",_result,0,2026-08-27T18:00:00Z,1.5,value,benchmark,h1\n"
    )
    status, body, _ = facade.translate_response(read, 200, csv.encode())
    payload = json.loads(body)
    assert status == 200 and payload["count"] == 1
    assert payload["points"][0] == {
        "t": "2026-08-27T18:00:00Z",
        "metric": "value",
        "v": 1.5,
        "host": "h1",
    }
    delete = UpstreamRequest("delete", "POST", "/api/v2/delete", {}, b"", series="benchmark")
    status, body, _ = facade.translate_response(delete, 204, b"")
    assert (status, json.loads(body)["removed"]) == (200, True)


@pytest.fixture
def kestrel_proxy(fake_influx: FakeInfluxServer) -> Any:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    proxy = MeasurementProxy(
        listen_host="127.0.0.1",
        listen_port=port,
        upstream_url=f"http://127.0.0.1:{fake_influx.port}",
        upstream_timeout_s=10.0,
        api="kestrel",
        org="benchmark",
        bucket="benchmark",
    )
    proxy.start()
    try:
        yield proxy, f"http://127.0.0.1:{port}"
    finally:
        proxy.stop()


def test_gold_kestrel_artifact_runs_through_the_facade(
    kestrel_proxy: Any, project_root: Path
) -> None:
    proxy, base_url = kestrel_proxy
    settings = SutSettings(api="kestrel", token="secret")
    env = sut_environment(settings, influx_url=base_url)
    assert set(env) == {"KESTREL_ENDPOINT", "KESTREL_TOKEN", "KESTREL_TENANT", "KESTREL_STORE"}

    validation = validate_artifacts(GOLD / "benchmark.feature", GOLD / "steps.py", env=env)
    assert validation.syntax_ok, validation.issues

    proxy.open_window()
    result = BehaveRunner(60).run(
        GOLD / "benchmark.feature",
        GOLD / "steps.py",
        env={**sanitized_environment(os.environ), **env},
    )
    stats = proxy.close_window()

    assert result.exit_code == 0, result.stdout[-1500:] + result.stderr[-500:]
    adherence = evaluate_adherence(stats, api="kestrel")
    assert adherence is not None and adherence.ok, adherence
    workload = load_app_config(project_root).workload
    conformance = evaluate_conformance(workload, stats)
    assert conformance is not None and conformance.ok, conformance.checks
    assert conformance.spec_coverage == 1.0


def test_probe_toolkit_speaks_kestrel(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, Any]] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> Any:
        captured.append({"method": method, "url": url, **kwargs})
        return SimpleNamespace(status_code=200, text="{}")

    monkeypatch.setattr("benchmark_agent.agent.tools.httpx.request", fake_request)
    toolkit = SutProbeToolkit(base_url="http://127.0.0.1:8086", auth_token="secret", api="kestrel")

    assert "HTTP 200" in toolkit.call("sut_health")
    assert captured[-1]["url"].endswith("/v1/status")
    assert captured[-1]["headers"] == {"X-Api-Token": "secret"}
    rejected = toolkit.call("http_request", {"method": "POST", "path": POINTS, "body": "{}"})
    assert rejected.startswith("Rejected: POST is not needed")
    toolkit.call("http_request", {"method": "GET", "path": f"{POINTS}?since=PT1H"})
    assert captured[-1]["method"] == "GET"
    assert len(captured) == 2


def _post(url: str, body: bytes, headers: dict[str, str]) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, method="POST")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
