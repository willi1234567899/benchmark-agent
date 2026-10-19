from __future__ import annotations

import json
from pathlib import Path

from benchmark_agent.config import _load_yaml_model, load_app_config
from benchmark_agent.evaluation.conformance import evaluate_conformance
from benchmark_agent.models import (
    OperationWindowStats,
    ProxyWindowStats,
    RequestSample,
    WorkloadSpec,
)

FLUX_V2 = (
    'from(bucket: "benchmark") |> range(start: -2h) '
    '|> filter(fn: (r) => r._measurement == "benchmark") '
    '|> filter(fn: (r) => r["host"] == "h1") |> group() |> limit(n: 25)'
)


def _sample(
    operation: str, body: str | None, *, n_points: int = 0, upstream: str | None = None
) -> RequestSample:
    return RequestSample(
        operation=operation,
        method="POST",
        path=f"/tessera/v1/{operation}",
        status_code=200,
        latency_ms=1.0,
        error=False,
        n_points=n_points,
        body_excerpt=body,
        upstream_path=upstream,
    )


def _stats(samples: list[RequestSample]) -> ProxyWindowStats:
    by_operation = {
        op: OperationWindowStats(
            n_requests=sum(1 for sample in samples if sample.operation == op),
            n_points=sum(sample.n_points for sample in samples if sample.operation == op),
            throughput_ops_s=1.0,
        )
        for op in ("write", "read", "delete")
    }
    return ProxyWindowStats(window_duration_s=1.0, by_operation=by_operation, requests=samples)


def _purge(predicate: str) -> str:
    return json.dumps(
        {"start": "1970-01-01T00:00:00Z", "stop": "2100-01-01T00:00:00Z", "predicate": predicate}
    )


def _workload_v2(project_root: Path) -> WorkloadSpec:
    return _load_yaml_model(project_root / "config" / "workload_v2.yaml", WorkloadSpec)


def test_workload_v3_is_the_live_config(project_root: Path) -> None:
    workload = load_app_config(project_root).workload
    assert workload.name == "tsdb_write_read_delete_v3"
    write = workload.operations["write"].settings_dict()
    assert write["ts_unit"] == "ms"
    assert (write["batch_size"], write["batches"]) == (100, 10)
    assert workload.operations["read"].settings_dict()["labels"] == {"host": "h1"}
    assert workload.operations["read"].settings_dict()["repeats"] == 20
    assert workload.operations["delete"].settings_dict()["labels"] == {"host": "h1"}


def test_workload_v2_is_frozen(project_root: Path) -> None:
    v2 = _load_yaml_model(project_root / "config" / "workload_v2.yaml", WorkloadSpec)
    assert v2.name == "tsdb_write_read_delete_v2"
    assert v2.operations["write"].settings_dict()["batches"] == 1
    assert "repeats" not in v2.operations["read"].settings_dict()


def test_workload_v3_load_counts_are_soft(project_root: Path) -> None:
    workload = load_app_config(project_root).workload
    read_body = json.dumps(
        {
            "query": 'from(bucket: "benchmark") |> range(start: -2h) '
            '|> filter(fn: (r) => r._measurement == "benchmark" and r.host == "h1") '
            "|> limit(n: 25)"
        }
    )
    delete_body = _purge('_measurement="benchmark" AND host="h1"')
    conforming = _stats(
        [
            *(
                _sample("write", None, n_points=100, upstream="/api/v2/write?precision=ms")
                for _ in range(10)
            ),
            *(_sample("read", read_body) for _ in range(20)),
            _sample("delete", delete_body),
        ]
    )
    result = evaluate_conformance(workload, conforming)
    assert result is not None and result.ok, result.checks
    assert result.deviations == []
    assert result.spec_coverage == 1.0

    one_batch = _stats(
        [
            _sample("write", None, n_points=1000, upstream="/api/v2/write?precision=ms"),
            _sample("read", read_body),
            _sample("delete", delete_body),
        ]
    )
    result = evaluate_conformance(workload, one_batch)
    assert result is not None and result.ok, result.checks
    assert result.deviations == [
        "write: 1 write requests, pinned 10 batch(es)",
        "read: 1 read requests, pinned 20",
    ]

    short = _stats(
        [
            *(
                _sample("write", None, n_points=100, upstream="/api/v2/write?precision=ms")
                for _ in range(9)
            ),
            *(_sample("read", read_body) for _ in range(20)),
            _sample("delete", delete_body),
        ]
    )
    result = evaluate_conformance(workload, short)
    assert result is not None and not result.ok
    assert [check.name for check in result.checks if not check.ok] == ["points"]


def test_spec_only_affordances_pass_and_are_counted(project_root: Path) -> None:
    workload = _workload_v2(project_root)
    stats = _stats(
        [
            _sample(
                "write",
                None,
                n_points=100,
                upstream="/api/v2/write?org=benchmark&bucket=benchmark&precision=ms",
            ),
            _sample("read", json.dumps({"query": FLUX_V2})),
            _sample("delete", _purge('_measurement="benchmark" AND host="h1"')),
        ]
    )

    result = evaluate_conformance(workload, stats)

    assert result is not None
    assert result.ok, [check for check in result.checks if not check.ok]
    assert result.spec_coverage == 1.0
    assert result.spec_checks == ["write/ts_unit:ok", "read/labels:ok", "delete/labels:ok"]


def test_ignoring_the_spec_only_parts_fails_conformance(project_root: Path) -> None:
    workload = _workload_v2(project_root)
    stats = _stats(
        [
            _sample(
                "write",
                None,
                n_points=100,
                upstream="/api/v2/write?org=benchmark&bucket=benchmark&precision=ns",
            ),
            _sample(
                "read",
                json.dumps(
                    {
                        "query": 'from(bucket: "benchmark") |> range(start: -2h) '
                        '|> filter(fn: (r) => r._measurement == "benchmark") |> limit(n: 25)'
                    }
                ),
            ),
            _sample("delete", _purge('_measurement="benchmark"')),
        ]
    )

    result = evaluate_conformance(workload, stats)

    assert result is not None
    assert not result.ok
    failed = {(check.operation, check.name) for check in result.checks if not check.ok}
    assert failed == {("write", "ts_unit"), ("read", "labels"), ("delete", "labels")}
    assert result.spec_coverage == 0.0


def test_workload_v1_has_no_spec_coverage(project_root: Path) -> None:
    v1 = _load_yaml_model(project_root / "config" / "workload_v1.yaml", WorkloadSpec)
    stats = _stats([_sample("write", None, n_points=100)])
    result = evaluate_conformance(v1, stats)
    assert result is not None
    assert result.spec_coverage is None
    assert result.spec_checks == []
