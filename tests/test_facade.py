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
from benchmark_agent.config import SutSettings
from benchmark_agent.evaluation.adherence import evaluate_adherence
from benchmark_agent.evaluation.conformance import evaluate_conformance
from benchmark_agent.execution.behave_runner import BehaveRunner, sanitized_environment
from benchmark_agent.execution.sut import sut_environment
from benchmark_agent.execution.validator import validate_artifacts
from benchmark_agent.measurement.facade import (
    LocalResponse,
    TesseraFacade,
    UpstreamRequest,
    parse_csv_rows,
)
from benchmark_agent.measurement.proxy import MeasurementProxy
from benchmark_agent.models import (
    ConditionConfig,
    GuidanceContext,
    GuidanceLevel,
    OperationSpec,
    PromptingStrategy,
    WorkloadSpec,
)
from benchmark_agent.prompting.strategies import build_zero_shot_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLD = PROJECT_ROOT / "tests" / "fixtures" / "gold_tessera"
KEY = {"X-Tessera-Key": "secret"}


def _facade() -> TesseraFacade:
    return TesseraFacade(org="benchmark", bucket="benchmark")


def _records(n: int = 3) -> bytes:
    return json.dumps(
        {
            "records": [
                {"series": "benchmark", "labels": {"host": "h 1"}, "values": {"value": i}, "ts": i}
                for i in range(n)
            ]
        }
    ).encode()


def test_ingest_translates_records_to_line_protocol() -> None:
    outcome = _facade().handle(
        "POST",
        "/tessera/v1/ingest",
        {"space": "benchmark", "store": "benchmark", "ts_unit": "ms"},
        KEY,
        _records(2),
    )

    assert isinstance(outcome, UpstreamRequest)
    assert outcome.operation == "write"
    assert outcome.path_qs == "/api/v2/write?org=benchmark&bucket=benchmark&precision=ms"
    assert outcome.headers["Authorization"] == "Token secret"
    assert outcome.body.decode() == (
        "benchmark,host=h\\ 1 value=0.0 0\nbenchmark,host=h\\ 1 value=1.0 1\n"
    )
    assert outcome.n_records == 2


@pytest.mark.parametrize(
    ("query", "headers", "body", "code"),
    [
        ({"org": "b", "bucket": "b"}, KEY, _records(), "influx_param_org"),
        (
            {"space": "b", "store": "b", "precision": "ns"},
            KEY,
            _records(),
            "influx_param_precision",
        ),
        (
            {"space": "b", "store": "b"},
            {"Authorization": "Token x"},
            _records(),
            "influx_auth_header",
        ),
        ({"space": "b", "store": "b"}, {}, _records(), "missing_key"),
        ({"space": "b", "store": "b"}, KEY, b"benchmark value=1 1\n", "non_json_body"),
        (
            {"space": "b", "store": "b"},
            KEY,
            json.dumps({"lines": []}).encode(),
            "influx_field_lines",
        ),
        ({"space": "b"}, KEY, _records(), "missing_store"),
    ],
)
def test_ingest_canaries_are_rejected(
    query: dict[str, str], headers: dict[str, str], body: bytes, code: str
) -> None:
    outcome = _facade().handle("POST", "/tessera/v1/ingest", query, headers, body)

    assert isinstance(outcome, LocalResponse)
    assert outcome.rejection == code
    assert outcome.operation == "write"
    assert outcome.status in (400, 401)
    assert "error" in json.loads(outcome.body)


def test_fetch_translates_to_flux_with_global_limit() -> None:
    body = json.dumps(
        {
            "store": "benchmark",
            "series": "benchmark",
            "window": {"from": "-1h"},
            "max_rows": 10,
            "labels": {"host": "web"},
        }
    ).encode()

    outcome = _facade().handle("POST", "/tessera/v1/fetch", {"space": "benchmark"}, KEY, body)

    assert isinstance(outcome, UpstreamRequest)
    assert outcome.operation == "read"
    assert outcome.path_qs == "/api/v2/query?org=benchmark"
    payload = json.loads(outcome.body)
    assert payload["dialect"]["annotations"] == []
    flux = payload["query"]
    assert flux.startswith('from(bucket: "benchmark") |> range(start: -1h)')
    assert '|> filter(fn: (r) => r._measurement == "benchmark")' in flux
    assert '|> filter(fn: (r) => r["host"] == "web")' in flux
    assert flux.endswith("|> group() |> limit(n: 10)")


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b'from(bucket:"b") |> range(start: -1h)', "non_json_body"),
        (json.dumps({"query": "x"}).encode(), "influx_field_query"),
        (
            json.dumps(
                {"store": "b", "series": "s", "window": {"from": "-1h"}, "limit": 10}
            ).encode(),
            "influx_field_limit",
        ),
        (json.dumps({"store": "b", "series": "s"}).encode(), "missing_field_window"),
        (
            json.dumps({"store": "b", "series": "s", "window": {"from": "yesterday"}}).encode(),
            "invalid_window_from",
        ),
    ],
)
def test_fetch_canaries_are_rejected(body: bytes, code: str) -> None:
    outcome = _facade().handle("POST", "/tessera/v1/fetch", {"space": "b"}, KEY, body)

    assert isinstance(outcome, LocalResponse)
    assert outcome.rejection == code
    assert outcome.operation == "read"


def test_purge_translates_to_delete_predicate() -> None:
    body = json.dumps(
        {
            "series": "benchmark",
            "from": "1970-01-01T00:00:00Z",
            "until": "2100-01-01T00:00:00Z",
            "labels": {"host": "web"},
        }
    ).encode()

    outcome = _facade().handle(
        "POST", "/tessera/v1/purge", {"space": "benchmark", "store": "benchmark"}, KEY, body
    )

    assert isinstance(outcome, UpstreamRequest)
    assert outcome.operation == "delete"
    assert outcome.path_qs == "/api/v2/delete?org=benchmark&bucket=benchmark"
    assert json.loads(outcome.body) == {
        "start": "1970-01-01T00:00:00Z",
        "stop": "2100-01-01T00:00:00Z",
        "predicate": '_measurement="benchmark" AND host="web"',
    }


def test_purge_rejects_influx_delete_body() -> None:
    body = json.dumps(
        {"start": "1970-01-01T00:00:00Z", "stop": "2100-01-01T00:00:00Z", "predicate": "x"}
    ).encode()

    outcome = _facade().handle("POST", "/tessera/v1/purge", {"space": "b", "store": "b"}, KEY, body)

    assert isinstance(outcome, LocalResponse)
    assert outcome.rejection in {
        "influx_field_start",
        "influx_field_stop",
        "influx_field_predicate",
    }


def test_legacy_influx_endpoints_are_unknown_but_classified() -> None:
    outcome = _facade().handle(
        "POST", "/api/v2/write", {"org": "b", "bucket": "b"}, {"Authorization": "Token x"}, b"m v=1"
    )

    assert isinstance(outcome, LocalResponse)
    assert outcome.status == 404
    assert outcome.api == "legacy"
    assert outcome.rejection == "legacy_endpoint"
    assert outcome.operation == "write"
    assert "/tessera/v1" in json.loads(outcome.body)["error"]


def test_index_is_served_without_a_key() -> None:
    for path in ("/", "/tessera/v1", "/tessera/v1/"):
        outcome = _facade().handle("GET", path, {}, {}, b"")
        assert isinstance(outcome, LocalResponse)
        assert outcome.status == 200
        index = json.loads(outcome.body)
        assert index["auth"]["header"] == "X-Tessera-Key"
        assert {endpoint["path"] for endpoint in index["endpoints"]} >= {
            "/tessera/v1/ingest",
            "/tessera/v1/fetch",
            "/tessera/v1/purge",
        }


def test_response_translation() -> None:
    facade = _facade()
    write = UpstreamRequest("write", "POST", "/api/v2/write", {}, b"", n_records=100)
    assert facade.translate_response(write, 204, b"") == (
        200,
        b'{"accepted": 100}',
        "application/json",
    )
    delete = UpstreamRequest("delete", "POST", "/api/v2/delete", {}, b"", series="benchmark")
    status, body, _ = facade.translate_response(delete, 204, b"")
    assert (status, json.loads(body)) == (200, {"purged": True, "series": "benchmark"})
    read = UpstreamRequest("read", "POST", "/api/v2/query", {}, b"", series="benchmark")
    csv = ",result,table,_time,_value,_field,host\n,_result,0,2026-08-27T18:00:00Z,1.5,value,a\n\n"
    status, body, _ = facade.translate_response(read, 200, csv.encode())
    payload = json.loads(body)
    assert status == 200
    assert payload["row_count"] == 1
    assert payload["rows"][0] == {
        "ts": "2026-08-27T18:00:00Z",
        "value": 1.5,
        "metric": "value",
        "host": "a",
    }
    status, body, _ = facade.translate_response(write, 422, b"field type conflict")
    assert status == 422
    assert json.loads(body)["error"] == "field type conflict"


def test_parse_csv_rows_handles_annotations_and_multiple_tables() -> None:
    text = (
        "#datatype,string,long,dateTime:RFC3339,double\n"
        ",result,table,_time,_value\n"
        ",_result,0,2026-08-20T00:00:00Z,1\n"
        "\n"
        ",result,table,_time,_value,host\n"
        ",_result,1,2026-08-20T00:00:01Z,2,b\n"
    )
    rows = parse_csv_rows(text)
    assert [row["value"] for row in rows] == [1.0, 2.0]
    assert rows[1]["host"] == "b"


@pytest.fixture
def facade_proxy(fake_influx: FakeInfluxServer) -> Any:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    proxy = MeasurementProxy(
        listen_host="127.0.0.1",
        listen_port=port,
        upstream_url=f"http://127.0.0.1:{fake_influx.port}",
        upstream_timeout_s=10.0,
        api="tessera",
        org="benchmark",
        bucket="benchmark",
    )
    proxy.start()
    try:
        yield proxy, f"http://127.0.0.1:{port}"
    finally:
        proxy.stop()


def test_proxy_in_facade_mode_translates_and_classifies(facade_proxy: Any) -> None:
    proxy, base_url = facade_proxy
    proxy.open_window()
    ingest = _post(
        f"{base_url}/tessera/v1/ingest?space=benchmark&store=benchmark", _records(100), KEY
    )
    fetch = _post(
        f"{base_url}/tessera/v1/fetch?space=benchmark",
        json.dumps(
            {"store": "benchmark", "series": "benchmark", "window": {"from": "-1h"}, "max_rows": 10}
        ).encode(),
        KEY,
    )
    purge = _post(
        f"{base_url}/tessera/v1/purge?space=benchmark&store=benchmark",
        json.dumps(
            {"series": "benchmark", "from": "1970-01-01T00:00:00Z", "until": "2100-01-01T00:00:00Z"}
        ).encode(),
        KEY,
    )
    legacy = _post(
        f"{base_url}/api/v2/write?org=benchmark&bucket=benchmark",
        b"benchmark value=1 1\n",
        {"Authorization": "Token secret"},
    )
    canary = _post(f"{base_url}/tessera/v1/ingest?org=benchmark&bucket=benchmark", _records(1), KEY)
    stats = proxy.close_window()

    assert (ingest[0], json.loads(ingest[1])) == (200, {"accepted": 100})
    assert fetch[0] == 200 and json.loads(fetch[1])["row_count"] == 1
    assert (purge[0], json.loads(purge[1])["purged"]) == (200, True)
    assert legacy[0] == 404
    assert canary[0] == 400

    by_path: dict[str, Any] = {}
    for sample in stats.requests:
        by_path.setdefault(sample.path.split("?")[0], sample)
    assert by_path["/tessera/v1/ingest"].n_points == 100
    assert by_path["/tessera/v1/ingest"].api == "tessera"
    assert "limit(n: 10)" in (by_path["/tessera/v1/fetch"].body_excerpt or "")
    assert '"predicate": "_measurement=\\"benchmark\\""' in (
        by_path["/tessera/v1/purge"].body_excerpt or ""
    )
    assert by_path["/api/v2/write"].api == "legacy"
    assert by_path["/api/v2/write"].error is True

    workload = WorkloadSpec(
        name="w",
        description="d",
        measurement_window_seconds=60,
        operations={
            "write": OperationSpec(enabled=True, batch_size=100, batches=1),
            "read": OperationSpec(
                enabled=True,
                query_template='from(bucket: "benchmark") |> range(start: -1h) |> limit(n: 10)',
            ),
            "delete": OperationSpec(
                enabled=True,
                predicate='_measurement="benchmark"',
                start="1970-01-01T00:00:00Z",
                stop="2100-01-01T00:00:00Z",
            ),
        },
    )
    conformance = evaluate_conformance(workload, stats)
    assert conformance is not None and conformance.ok, conformance.checks
    adherence = evaluate_adherence(stats, api="tessera")
    assert adherence is not None
    assert (adherence.accepted, adherence.rejected, adherence.legacy) == (3, 1, 1)
    assert adherence.ok is False
    assert adherence.rejections == ["influx_param_org x1", "legacy_endpoint x1"]


def test_gold_tessera_artifact_runs_through_the_facade(facade_proxy: Any, tmp_path: Path) -> None:
    proxy, base_url = facade_proxy
    settings = SutSettings(api="tessera", token="secret")
    env = sut_environment(settings, influx_url=base_url)
    assert set(env) == {"TESSERA_URL", "TESSERA_KEY", "TESSERA_SPACE", "TESSERA_STORE"}

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
    adherence = evaluate_adherence(stats, api="tessera")
    assert adherence is not None and adherence.ok, adherence


def test_sut_environment_default_stays_influx() -> None:
    env = sut_environment(SutSettings(), influx_url="http://x")
    assert "INFLUX_URL" in env and "TESSERA_URL" not in env


def test_probe_toolkit_speaks_tessera(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, Any]] = []

    def fake_request(method: str, url: str, **kwargs: Any) -> Any:
        captured.append({"method": method, "url": url, **kwargs})
        return SimpleNamespace(status_code=200, text="{}")

    monkeypatch.setattr("benchmark_agent.agent.tools.httpx.request", fake_request)
    toolkit = SutProbeToolkit(base_url="http://127.0.0.1:8086", auth_token="secret", api="tessera")

    assert "HTTP 200" in toolkit.call("sut_health")
    assert captured[-1]["url"].endswith("/tessera/v1/health")
    assert captured[-1]["headers"] == {"X-Tessera-Key": "secret"}
    toolkit.call("http_request", {"method": "POST", "path": "/tessera/v1/fetch", "body": "{}"})
    assert captured[-1]["url"].endswith("/tessera/v1/fetch")
    rejected = toolkit.call(
        "http_request", {"method": "POST", "path": "/tessera/v1/ingest", "body": "{}"}
    )
    assert rejected.startswith("Rejected: POST is only allowed for /tessera/v1/fetch")
    assert len(captured) == 2


def test_prompt_names_the_facade_and_not_influx() -> None:
    condition = ConditionConfig(
        id=1,
        name="L1_gemini_zeroshot",
        guidance_level=GuidanceLevel.L1,
        model="gemini",
        prompting_strategy=PromptingStrategy.ZERO_SHOT,
    )
    context = GuidanceContext(
        level=GuidanceLevel.L1,
        workload_name="w",
        workload_prompt="task text",
        probing_enabled=True,
        probe_budget=8,
    )
    templates = PROJECT_ROOT / "src" / "benchmark_agent" / "prompting" / "templates"

    _, prompt = build_zero_shot_prompt(
        templates,
        condition,
        context,
        sut_api="tessera",
        sut_name="the Tessera time-series store (HTTP API v1)",
    )

    assert "Tessera time-series store" in prompt
    assert "X-Tessera-Key" in prompt
    assert "GET /tessera/v1/health" in prompt
    assert "InfluxDB" not in prompt
    assert "Authorization: Token" not in prompt


def _post(url: str, body: bytes, headers: dict[str, str]) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
