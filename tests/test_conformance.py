from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_evaluation import _behave_ok

from benchmark_agent.evaluation.conformance import CRITERIA_VERSION, evaluate_conformance
from benchmark_agent.evaluation.success import evaluate_semantics
from benchmark_agent.measurement.proxy import MeasurementProxy
from benchmark_agent.models import (
    ConformanceResult,
    OperationSpec,
    OperationWindowStats,
    ProxyWindowStats,
    RequestSample,
    WorkloadSpec,
)

PINNED_QUERY = (
    'from(bucket: "benchmark") |> range(start: -1h) '
    '|> filter(fn: (r) => r._measurement == "benchmark") |> limit(n: 10)'
)


def _workload() -> WorkloadSpec:
    return WorkloadSpec(
        name="influxdb_write_read_delete_smoke",
        description="pinned",
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


def _sample(
    operation: str, body: str | None, *, n_points: int = 0, error: bool = False
) -> RequestSample:
    return RequestSample(
        operation=operation,
        method="POST",
        path=f"/api/v2/{'query' if operation == 'read' else operation}",
        status_code=400 if error else 204,
        latency_ms=1.0,
        error=error,
        n_points=n_points,
        body_excerpt=body,
    )


def _stats(samples: list[RequestSample]) -> ProxyWindowStats:
    by_operation = {}
    for operation in ("write", "read", "delete"):
        own = [sample for sample in samples if sample.operation == operation]
        by_operation[operation] = OperationWindowStats(
            n_requests=len(own),
            error_count=sum(1 for sample in own if sample.error),
            n_points=sum(sample.n_points for sample in own if not sample.error),
            throughput_ops_s=float(len(own)),
        )
    return ProxyWindowStats(window_duration_s=1.0, by_operation=by_operation, requests=samples)


def _delete_body(start: str = "1970-01-01T00:00:00Z", stop: str = "2100-01-01T00:00:00Z") -> str:
    return json.dumps({"start": start, "stop": stop, "predicate": '_measurement="benchmark"'})


def test_pinned_workload_is_conformant_without_deviations() -> None:
    stats = _stats(
        [
            _sample("write", None, n_points=100),
            _sample("read", json.dumps({"query": PINNED_QUERY, "type": "flux"})),
            _sample("delete", _delete_body()),
        ]
    )

    result = evaluate_conformance(_workload(), stats)

    assert result is not None
    assert result.ok, [check for check in result.checks if not check.ok]
    assert result.deviations == []
    assert result.criteria_version == CRITERIA_VERSION
    assert {check.name for check in result.checks} == {
        "points",
        "limit",
        "range",
        "measurement",
        "predicate",
        "time_range",
    }


def test_docs_literal_stop_is_a_deviation_not_a_failure() -> None:
    stats = _stats(
        [
            _sample("write", None, n_points=100),
            _sample("read", PINNED_QUERY),
            _sample("delete", _delete_body(stop="2030-01-01T00:00:00Z")),
        ]
    )

    result = evaluate_conformance(_workload(), stats)

    assert result is not None
    assert result.ok
    assert len(result.deviations) == 1
    assert "differs from the pinned literals" in result.deviations[0]


def test_short_write_missing_limit_and_wrong_measurement_fail() -> None:
    stats = _stats(
        [
            _sample("write", None, n_points=10),
            _sample("read", 'from(bucket: "b") |> range(start: -1h)'),
            _sample(
                "delete",
                json.dumps(
                    {
                        "start": "1970-01-01T00:00:00Z",
                        "stop": "2100-01-01T00:00:00Z",
                        "predicate": '_measurement="smoke_metric"',
                    }
                ),
            ),
        ]
    )

    result = evaluate_conformance(_workload(), stats)

    assert result is not None
    assert not result.ok
    failed = {check.name for check in result.checks if not check.ok}
    assert failed == {"points", "limit", "measurement", "predicate"}
    points = next(check for check in result.checks if check.name == "points")
    assert points.detail == "accepted points 10, pinned 100"


def test_errored_requests_and_data_not_covered_do_not_count() -> None:
    stats = _stats(
        [
            _sample("write", None, n_points=100),
            _sample("read", PINNED_QUERY, error=True),
            _sample("delete", _delete_body(stop="2000-01-01T00:00:00Z")),
        ]
    )

    result = evaluate_conformance(_workload(), stats)

    assert result is not None
    failed = {check.name for check in result.checks if not check.ok}
    assert failed == {"limit", "range", "measurement", "time_range"}


def test_no_proxy_evidence_yields_none() -> None:
    assert evaluate_conformance(_workload(), None) is None


def test_semantics_requires_conformance_when_provided() -> None:
    stats = _stats(
        [
            _sample("write", None, n_points=100),
            _sample("read", PINNED_QUERY),
            _sample("delete", _delete_body()),
        ]
    )
    assert evaluate_semantics(stats, _behave_ok()) is True
    assert evaluate_semantics(stats, _behave_ok(), conformance=ConformanceResult(ok=True)) is True
    assert evaluate_semantics(stats, _behave_ok(), conformance=ConformanceResult(ok=False)) is False


def test_proxy_records_read_and_delete_bodies_but_not_write(
    proxy: tuple[MeasurementProxy, str],
) -> None:
    measurement_proxy, base_url = proxy
    measurement_proxy.open_window()
    _post(
        f"{base_url}/api/v2/write?bucket=b",
        b"benchmark,host=a value=1 1\nbenchmark,host=a value=2 2\n",
    )
    _post(f"{base_url}/api/v2/query?org=o", json.dumps({"query": PINNED_QUERY}).encode())
    _post(f"{base_url}/api/v2/delete?org=o&bucket=b", _delete_body().encode())
    stats = measurement_proxy.close_window()

    by_operation = {sample.operation: sample for sample in stats.requests}
    assert by_operation["write"].body_excerpt is None
    assert by_operation["write"].n_points == 2
    assert json.loads(by_operation["read"].body_excerpt or "{}")["query"] == PINNED_QUERY
    assert '"predicate"' in (by_operation["delete"].body_excerpt or "")

    result = evaluate_conformance(_workload(), stats)
    assert result is not None
    assert {check.name for check in result.checks if not check.ok} == {"points"}


def _post(url: str, body: bytes) -> None:
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=5) as response:
        response.read()


def test_generation_records_epoch_and_per_repetition_seed(
    project_root: Path, patch_llm_client: None, isolated_artifacts: Path
) -> None:
    from benchmark_agent.config import load_app_config
    from benchmark_agent.orchestration.runner import GenerationRunner

    defaults = load_app_config(project_root).defaults
    assert defaults.seed_per_repetition is True
    assert defaults.epoch >= 2

    GenerationRunner(project_root).run_condition("L2_gemini_zeroshot", repetition=3)

    run_directory = sorted((isolated_artifacts / "L2_gemini_zeroshot").glob("run-*"))[-1]
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["seed"] == defaults.seed + 2
    assert metadata["epoch"] == defaults.epoch
    assert metadata["workload"] == load_app_config(project_root).workload.name


def test_gemini_client_passes_seed_and_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark_agent.llm.gemini_client import GeminiLLMClient
    from benchmark_agent.models import LLMRequest, ModelConfig

    model = ModelConfig(
        provider="google",
        model_name="gemini-3.5-flash",
        model_version="gemini-3.5-flash",
        client_adapter="gemini",
        parameters={"temperature": 0.7},
    )
    client = GeminiLLMClient(model, api_key="test-key")
    captured: dict[str, object] = {}

    def fake_generate_content(*, model: str, contents: str, config: object) -> object:
        captured["config"] = config
        return SimpleNamespace(text='{"feature": "f", "steps_py": "s"}')

    monkeypatch.setattr(client._client.models, "generate_content", fake_generate_content)

    client.complete(
        LLMRequest(condition_name="c", model_name="gemini-3.5-flash", prompt="p", seed=20260717)
    )

    config = captured["config"]
    assert config.seed == 20260717
    assert config.temperature == 0.7


def test_delete_stop_now_counts_as_full_range_with_deviation() -> None:
    from datetime import UTC, datetime

    stop_now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    stats = _stats(
        [
            _sample("write", None, n_points=100),
            _sample("read", PINNED_QUERY),
            _sample("delete", _delete_body(stop=stop_now)),
        ]
    )

    result = evaluate_conformance(_workload(), stats)

    assert result is not None
    assert result.ok, [check for check in result.checks if not check.ok]
    assert len(result.deviations) == 1
