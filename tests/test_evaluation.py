from __future__ import annotations

import json
import shutil
import socket
import time
from pathlib import Path
from typing import Any

import pytest
from fake_influx import start_fake_influx

from benchmark_agent.evaluation.coverage import evaluate_coverage
from benchmark_agent.evaluation.success import evaluate_semantics, evaluate_success
from benchmark_agent.execution.behave_runner import BehaveRunner
from benchmark_agent.execution.validator import validate_artifacts
from benchmark_agent.measurement.op_mapping import map_path_to_operation
from benchmark_agent.measurement.server_metrics import ServerMetricsCollector
from benchmark_agent.models import (
    BehaveRunResult,
    MetricRow,
    MetricsSnapshot,
    OperationWindowStats,
    ProxyWindowStats,
    ValidationResult,
)
from benchmark_agent.orchestration.evaluation_runner import EvaluationRunner
from benchmark_agent.results_io import (
    RAW_CSV_COLUMNS,
    append_metric_rows,
    read_metric_rows,
    replace_metric_rows,
    utc_timestamp,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLD = FIXTURES / "gold_live"
REFERENCE_FEATURES = Path(__file__).resolve().parents[1] / "reference" / "features"
OPERATIONS = ("write", "read", "delete")


def test_validate_artifacts_accepts_valid_fixture() -> None:
    result = validate_artifacts(
        FIXTURES / "valid" / "benchmark.feature",
        FIXTURES / "valid" / "steps.py",
    )
    assert result.syntax_ok is True
    assert result.issues == []


def test_validate_artifacts_rejects_broken_python() -> None:
    result = validate_artifacts(
        FIXTURES / "invalid_steps" / "benchmark.feature",
        FIXTURES / "invalid_steps" / "steps.py",
    )
    assert result.syntax_ok is False
    assert any(issue.code == "PY_COMPILE" for issue in result.issues)


def test_behave_runner_executes_fixture() -> None:
    result = BehaveRunner(timeout_seconds=30).run(
        FIXTURES / "valid" / "benchmark.feature",
        FIXTURES / "valid" / "steps.py",
    )
    assert result.timed_out is False
    assert result.exit_code == 0
    assert result.scenarios_passed == 1
    assert result.scenarios_failed == 0
    assert result.scenarios_error == 0


def test_op_mapping_classifies_influx_endpoints() -> None:
    assert map_path_to_operation("/api/v2/write?org=benchmark") == "write"
    assert map_path_to_operation("/api/v2/query") == "read"
    assert map_path_to_operation("/api/v2/delete") == "delete"
    assert map_path_to_operation("/health") is None
    assert map_path_to_operation("/write?db=x") == "write"
    assert map_path_to_operation("/query?q=x") == "read"
    assert map_path_to_operation("/api/v2/writer") is None


def test_server_metrics_delta_counts_operations() -> None:
    collector = ServerMetricsCollector()
    start = MetricsSnapshot(
        samples={
            'http_api_requests_total{path="/api/v2/write"}': 10.0,
            'http_api_requests_total{path="/api/v2/query"}': 4.0,
        }
    )
    end = MetricsSnapshot(
        samples={
            'http_api_requests_total{path="/api/v2/write"}': 15.0,
            'http_api_requests_total{path="/api/v2/query"}': 7.0,
            'http_api_requests_total{path="/api/v2/delete"}': 2.0,
        }
    )
    delta = collector.delta(start, end)
    assert delta.by_operation["write"].n_requests == 5
    assert delta.by_operation["read"].n_requests == 3
    assert delta.by_operation["delete"].n_requests == 2
    assert delta.by_operation["write"].latency_p50_ms is None


def test_results_io_roundtrip(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    row = MetricRow(
        condition_id=1,
        condition_name="L2_gemini_zeroshot",
        guidance_level="L2",
        model="gemini",
        prompting="zero_shot",
        repetition=1,
        operation="write",
        source="proxy",
        executed_ok=True,
        syntax_ok=True,
        semantics_ok=True,
        throughput_ops_s=12.5,
        latency_p50_ms=1.0,
        latency_p95_ms=2.0,
        latency_p99_ms=3.0,
        n_requests=10,
        error_count=0,
        covered=True,
        feedback_iterations=0,
        model_version="gemini-3.5-flash",
        seed=1,
        timestamp=utc_timestamp(),
    )
    append_metric_rows(csv_path, [row])
    loaded = read_metric_rows(csv_path)
    assert len(loaded) == 1
    assert loaded[0].operation == "write"
    assert loaded[0].throughput_ops_s == 12.5


def test_config_loads_sut_and_measurement(project_root: Path) -> None:
    from benchmark_agent.config import load_app_config

    config = load_app_config(project_root)
    assert config.sut.bucket == "benchmark"
    assert config.execution.behave_timeout_seconds == 120
    assert config.measurement.proxy_listen_port == 8086
    assert config.paths.results_raw_dir.name == "raw"


def test_gold_live_fixture_is_syntactically_valid() -> None:
    result = validate_artifacts(GOLD / "benchmark.feature", GOLD / "steps.py")
    assert result.syntax_ok is True, result.issues


def test_evaluate_gold_live_via_proxy_writes_metrics_csv(
    project_root: Path, tmp_path: Path
) -> None:
    server, thread = start_fake_influx()
    proxy_port = _free_port()
    run_directory = tmp_path / "run-001"
    raw_dir = tmp_path / "raw"
    _copy_gold_run(run_directory)

    try:
        runner = EvaluationRunner(project_root)
        _point_runner_at_fake(runner, proxy_port, server.port, raw_dir)
        result = runner.run_artifacts(
            run_directory,
            start_sut=False,
            stop_sut=False,
            enable_measurement=True,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert result.skipped_execution is False
    assert result.success is not None
    assert result.success.executed_ok is True, _behave_debug(result)
    assert result.coverage is not None
    assert result.coverage.coverage_ratio == 1.0
    assert set(result.coverage.covered_operations) == {"write", "read", "delete"}
    assert result.csv_path is not None
    rows = read_metric_rows(result.csv_path)
    proxy_rows = [row for row in rows if row.source == "proxy"]
    assert {row.operation for row in proxy_rows if row.n_requests > 0} == {
        "write",
        "read",
        "delete",
    }
    assert all(row.condition_name == "_gold_live" for row in rows)
    assert result.success.semantics_ok is True
    assert result.status == "ok"
    assert all(row.status == "ok" and row.semantics_ok for row in rows)
    assert all(row.run_id == f"{tmp_path.name}/run-001" for row in rows)
    assert {row.source for row in rows} == {"proxy", "server"}
    write_row = next(row for row in proxy_rows if row.operation == "write")
    assert write_row.n_points == 1000
    assert write_row.n_requests == 10
    assert write_row.throughput_ops_s > 0
    read_row = next(row for row in proxy_rows if row.operation == "read")
    assert read_row.n_requests == 20
    server_write = next(row for row in rows if row.source == "server" and row.operation == "write")
    assert server_write.n_requests == 10
    assert server_write.error_count == 0
    assert server_write.latency_p95_ms is not None
    first_path, _first_headers, first_body = server.requests[0]
    assert first_path == "/api/v2/write"
    assert b"source=seed" in first_body
    assert server.write_count == 11
    assert not list(run_directory.glob("__pycache__"))


@pytest.mark.live_sut
def test_evaluate_gold_live_against_docker_influx(project_root: Path, tmp_path: Path) -> None:
    import httpx

    from benchmark_agent.config import load_app_config

    config = load_app_config(project_root)
    try:
        response = httpx.get(f"{config.sut.influxdb_direct_url.rstrip('/')}/health", timeout=2.0)
    except httpx.HTTPError as error:
        pytest.skip(f"InfluxDB SUT not reachable: {error}")
    if response.status_code != 200:
        pytest.skip(f"InfluxDB SUT health status {response.status_code}")

    run_directory = tmp_path / "run-001"
    _copy_gold_run(run_directory)
    runner = EvaluationRunner(project_root)
    runner._app_config.paths.results_raw_dir = tmp_path / "raw"
    result = runner.run_artifacts(
        run_directory,
        start_sut=False,
        stop_sut=False,
        enable_measurement=True,
    )
    assert result.success is not None
    assert result.success.executed_ok is True, _behave_debug(result)
    assert result.coverage is not None
    assert result.coverage.coverage_ratio == 1.0


def test_behave_runner_counts_failed_and_undefined_scenarios(tmp_path: Path) -> None:
    feature = tmp_path / "benchmark.feature"
    steps = tmp_path / "steps.py"
    feature.write_text(
        "Feature: mixed\n"
        "  Scenario: passes\n    Given ok\n"
        "  Scenario: fails\n    Given fail\n"
        "  Scenario: undefined\n    Given nothing here\n",
        encoding="utf-8",
    )
    steps.write_text(
        "from behave import given\n\n"
        '@given("ok")\ndef _ok(context):\n    pass\n\n'
        '@given("fail")\ndef _fail(context):\n    assert False, "boom"\n',
        encoding="utf-8",
    )
    result = BehaveRunner(timeout_seconds=60).run(feature, steps)
    assert result.timed_out is False
    assert result.exit_code != 0
    assert result.scenarios_passed == 1
    assert result.scenarios_failed == 1
    assert result.scenarios_error == 1
    success = evaluate_success(ValidationResult(syntax_ok=True), result, semantics_ok=False)
    assert success.executed_ok is False


def test_behave_runner_timeout_kills_grandchild(tmp_path: Path) -> None:
    feature = tmp_path / "benchmark.feature"
    steps = tmp_path / "steps.py"
    feature.write_text(
        "Feature: hang\n  Scenario: hang\n    Given a hanging child\n", encoding="utf-8"
    )
    steps.write_text(
        "import subprocess, sys, time\n"
        "from behave import given\n\n"
        '@given("a hanging child")\n'
        "def _hang(context):\n"
        '    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])\n'
        "    time.sleep(30)\n",
        encoding="utf-8",
    )
    started = time.monotonic()
    result = BehaveRunner(timeout_seconds=3).run(feature, steps)
    elapsed = time.monotonic() - started
    assert result.timed_out is True
    assert result.exit_code == 124
    assert elapsed < 20, f"timeout not enforced: {elapsed:.1f}s"


def _stats(n: int, errors: int = 0) -> OperationWindowStats:
    return OperationWindowStats(n_requests=n, error_count=errors, throughput_ops_s=float(n))


def test_coverage_ignores_errored_requests_and_failed_runs() -> None:
    all_error = {op: _stats(3, errors=3) for op in OPERATIONS}
    assert evaluate_coverage(all_error).coverage_ratio == 0.0
    healthy = {op: _stats(3) for op in OPERATIONS}
    assert evaluate_coverage(healthy).coverage_ratio == 1.0
    assert evaluate_coverage(healthy, executed_ok=False).coverage_ratio == 0.0
    partial = {"write": _stats(2, errors=1), "read": _stats(1, errors=1)}
    assert evaluate_coverage(partial).covered_operations == ["write"]


def _behave_ok() -> BehaveRunResult:
    return BehaveRunResult(
        exit_code=0, stdout="", stderr="", timed_out=False, scenarios_passed=1, scenarios_failed=0
    )


def test_evaluate_semantics_requires_every_operation_without_error() -> None:
    proxy_stats = ProxyWindowStats(
        window_duration_s=1.0, by_operation={op: _stats(2) for op in OPERATIONS}
    )
    assert evaluate_semantics(proxy_stats, _behave_ok()) is True
    assert evaluate_semantics(None, _behave_ok()) is False
    missing_delete = ProxyWindowStats(
        window_duration_s=1.0,
        by_operation={"write": _stats(2), "read": _stats(2), "delete": _stats(0)},
    )
    assert evaluate_semantics(missing_delete, _behave_ok()) is False
    all_errors = ProxyWindowStats(
        window_duration_s=1.0, by_operation={op: _stats(2, errors=2) for op in OPERATIONS}
    )
    assert evaluate_semantics(all_errors, _behave_ok()) is False
    zero_scenarios = _behave_ok().model_copy(update={"scenarios_passed": 0})
    assert evaluate_semantics(proxy_stats, zero_scenarios) is False
    success = evaluate_success(ValidationResult(syntax_ok=True), zero_scenarios, semantics_ok=True)
    assert success.executed_ok is False


def test_reference_features_pass_syntax_gate() -> None:
    features = sorted(REFERENCE_FEATURES.glob("*.feature"))
    assert features, "no reference features found"
    for feature in features:
        result = validate_artifacts(feature, GOLD / "steps.py", dry_run=False)
        assert result.syntax_ok is True, (feature.name, result.issues)
        assert all(issue.code.startswith("missing-") for issue in result.warnings)


def test_validator_demotes_style_rules_to_warnings() -> None:
    result = validate_artifacts(
        FIXTURES / "invalid_feature" / "benchmark.feature", FIXTURES / "valid" / "steps.py"
    )
    assert {issue.code for issue in result.warnings} == {"missing-when-step", "missing-then-step"}
    assert [issue.code for issue in result.issues] == ["UNDEFINED_STEP"]
    assert result.syntax_ok is False


def test_validator_dry_run_flags_import_error(tmp_path: Path) -> None:
    steps = tmp_path / "steps.py"
    steps.write_text("import influxdb_client\nfrom behave import given\n", encoding="utf-8")
    result = validate_artifacts(FIXTURES / "valid" / "benchmark.feature", steps)
    assert result.syntax_ok is False
    assert [issue.code for issue in result.issues] == ["IMPORT_ERROR"]
    assert "influxdb_client" in result.issues[0].message
    assert not list(tmp_path.glob("__pycache__")), "validator must not write bytecode"


def _write_run(
    run_directory: Path,
    *,
    feature: str,
    steps: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "benchmark.feature").write_text(feature, encoding="utf-8")
    (run_directory / "steps.py").write_text(steps, encoding="utf-8")
    payload = (
        metadata
        if metadata is not None
        else json.loads((GOLD / "metadata.json").read_text(encoding="utf-8"))
    )
    (run_directory / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")


def _runner_without_sut(project_root: Path, raw_dir: Path) -> EvaluationRunner:
    runner = EvaluationRunner(project_root)
    runner._app_config.paths.results_raw_dir = raw_dir
    runner._app_config.defaults.sut.influxdb_direct_url = f"http://127.0.0.1:{_free_port()}"
    return runner


def _valid_pair() -> tuple[str, str]:
    return (
        (FIXTURES / "valid" / "benchmark.feature").read_text(encoding="utf-8"),
        (FIXTURES / "valid" / "steps.py").read_text(encoding="utf-8"),
    )


def test_validation_failed_run_writes_failure_rows(project_root: Path, tmp_path: Path) -> None:
    run_directory = tmp_path / "cond_x" / "run-003"
    _write_run(
        run_directory,
        feature=(FIXTURES / "invalid_feature" / "benchmark.feature").read_text(encoding="utf-8"),
        steps=(FIXTURES / "valid" / "steps.py").read_text(encoding="utf-8"),
    )
    runner = _runner_without_sut(project_root, tmp_path / "raw")
    result = runner.run_artifacts(run_directory, start_sut=False, enable_measurement=True)
    assert result.skipped_execution is True
    assert result.status == "validation_failed"
    assert not (run_directory / "evaluation" / "behave.json").exists()
    rows = read_metric_rows(tmp_path / "raw" / "metrics.csv")
    assert {row.operation for row in rows} == set(OPERATIONS)
    assert all(
        row.status == "validation_failed"
        and row.source == "none"
        and row.covered is False
        and row.n_requests == 0
        and row.syntax_ok is False
        and row.executed_ok is False
        and row.repetition == 3
        and row.run_id == "cond_x/run-003"
        for row in rows
    )


def test_generation_failed_metadata_writes_rows_without_executing(
    project_root: Path, tmp_path: Path
) -> None:
    run_directory = tmp_path / "L3_gemini_zeroshot" / "run-002"
    run_directory.mkdir(parents=True)
    metadata = {
        "condition": {
            "id": 7,
            "name": "L3_gemini_zeroshot",
            "guidance_level": "L3",
            "model": "gemini",
            "prompting_strategy": "zero_shot",
        },
        "status": "generation_failed",
        "error": "LLM returned no JSON",
        "feedback_iterations": 2,
        "seed": 4242,
    }
    (run_directory / "metadata.json").write_text("﻿" + json.dumps(metadata), encoding="utf-8")
    runner = _runner_without_sut(project_root, tmp_path / "raw")
    result = runner.run_artifacts(run_directory, start_sut=False)
    assert result.skipped_execution is True
    assert result.status == "generation_failed"
    assert not (run_directory / "evaluation" / "validation.json").exists()
    rows = read_metric_rows(tmp_path / "raw" / "metrics.csv")
    assert len(rows) == 3
    assert all(row.status == "generation_failed" for row in rows)
    assert rows[0].condition_id == 7
    assert rows[0].guidance_level == "L3"
    assert rows[0].feedback_iterations == 2
    assert rows[0].repetition == 2
    assert rows[0].seed == 4242
    assert all(row.source == "none" for row in rows)


def test_missing_metadata_falls_back_to_parent_dir_name(project_root: Path, tmp_path: Path) -> None:
    run_directory = tmp_path / "L1_mock_zeroshot" / "run-001"
    run_directory.mkdir(parents=True)
    shutil.copy2(
        FIXTURES / "invalid_feature" / "benchmark.feature", run_directory / "benchmark.feature"
    )
    shutil.copy2(FIXTURES / "valid" / "steps.py", run_directory / "steps.py")
    runner = _runner_without_sut(project_root, tmp_path / "raw")
    runner.run_artifacts(run_directory, start_sut=False)
    rows = read_metric_rows(tmp_path / "raw" / "metrics.csv")
    assert rows and all(row.condition_name == "L1_mock_zeroshot" for row in rows)
    assert rows[0].guidance_level is None


def test_no_measure_yields_not_measured_rows(project_root: Path, tmp_path: Path) -> None:
    run_directory = tmp_path / "run-001"
    feature, steps = _valid_pair()
    _write_run(run_directory, feature=feature, steps=steps)
    runner = _runner_without_sut(project_root, tmp_path / "raw")
    result = runner.run_artifacts(run_directory, start_sut=False, enable_measurement=False)
    assert result.skipped_execution is False
    assert result.success is not None
    assert result.success.executed_ok is True, _behave_debug(result)
    assert result.success.semantics_ok is False
    assert result.coverage is None
    assert result.status == "not_measured"
    rows = read_metric_rows(tmp_path / "raw" / "metrics.csv")
    assert {row.operation for row in rows} == set(OPERATIONS)
    assert all(row.source == "none" and row.status == "not_measured" for row in rows)
    assert all(row.executed_ok and not row.semantics_ok and not row.covered for row in rows)


def test_reevaluation_replaces_rows(project_root: Path, tmp_path: Path) -> None:
    run_directory = tmp_path / "run-001"
    feature, steps = _valid_pair()
    _write_run(run_directory, feature=feature, steps=steps)
    runner = _runner_without_sut(project_root, tmp_path / "raw")
    runner.run_artifacts(run_directory, start_sut=False, enable_measurement=False)
    (run_directory / "evaluation" / "stale.txt").write_text("old", encoding="utf-8")
    rows_first = read_metric_rows(tmp_path / "raw" / "metrics.csv")
    runner.run_artifacts(run_directory, start_sut=False, enable_measurement=False)
    rows_second = read_metric_rows(tmp_path / "raw" / "metrics.csv")
    assert len(rows_first) == len(rows_second) == 3
    assert {row.evaluation_id for row in rows_second} != {row.evaluation_id for row in rows_first}
    assert not (run_directory / "evaluation" / "stale.txt").exists()


def test_infrastructure_failure_leaves_trace(project_root: Path, tmp_path: Path) -> None:
    run_directory = tmp_path / "run-001"
    _copy_gold_run(run_directory)
    runner = _runner_without_sut(project_root, tmp_path / "raw")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        runner._app_config.defaults.measurement.proxy_listen_port = int(blocker.getsockname()[1])
        with pytest.raises(RuntimeError):
            runner.run_artifacts(run_directory, start_sut=False, enable_measurement=True)
    summary = (run_directory / "evaluation" / "summary.json").read_text(encoding="utf-8")
    assert "infrastructure_failed" in summary
    rows = read_metric_rows(tmp_path / "raw" / "metrics.csv")
    assert rows and all(row.status == "infrastructure_failed" for row in rows)


def _row(**overrides: object) -> MetricRow:
    base: dict[str, object] = {
        "condition_name": "c",
        "operation": "write",
        "source": "proxy",
        "executed_ok": True,
        "syntax_ok": True,
        "semantics_ok": True,
        "throughput_ops_s": 1.0,
        "n_requests": 1,
        "error_count": 0,
        "covered": True,
        "timestamp": utc_timestamp(),
        "run_id": "c/run-001",
    }
    base.update(overrides)
    return MetricRow.model_validate(base)


def test_append_metric_rows_rejects_header_mismatch(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(",".join(RAW_CSV_COLUMNS[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="header"):
        append_metric_rows(csv_path, [_row()])


def test_optional_string_columns_roundtrip_none(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    append_metric_rows(csv_path, [_row(guidance_level=None, model=None, model_version=None)])
    loaded = read_metric_rows(csv_path)[0]
    assert loaded.guidance_level is None
    assert loaded.model is None
    assert loaded.prompting is None
    assert loaded.model_version is None


def test_read_metric_rows_reports_line_number(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    append_metric_rows(csv_path, [_row()])
    with csv_path.open("a", encoding="utf-8") as handle:
        handle.write(",".join(["x"] * len(RAW_CSV_COLUMNS)) + "\n")
    with pytest.raises(ValueError, match=r"metrics\.csv:3"):
        read_metric_rows(csv_path)


def test_replace_metric_rows_drops_same_run_id(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    append_metric_rows(csv_path, [_row(run_id="a/run-001"), _row(run_id="b/run-001")])
    replace_metric_rows(csv_path, "a/run-001", [_row(run_id="a/run-001", n_requests=9)])
    rows = read_metric_rows(csv_path)
    assert [(row.run_id, row.n_requests) for row in rows] == [("b/run-001", 1), ("a/run-001", 9)]


def _copy_gold_run(run_directory: Path) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(GOLD / "benchmark.feature", run_directory / "benchmark.feature")
    shutil.copy2(GOLD / "steps.py", run_directory / "steps.py")
    shutil.copy2(GOLD / "metadata.json", run_directory / "metadata.json")


def _point_runner_at_fake(
    runner: EvaluationRunner,
    proxy_port: int,
    upstream_port: int,
    csv_dir: Path,
) -> None:
    runner._app_config.defaults.sut.influxdb_url = f"http://127.0.0.1:{proxy_port}"
    runner._app_config.defaults.sut.influxdb_direct_url = f"http://127.0.0.1:{upstream_port}"
    runner._app_config.defaults.measurement.proxy_listen_port = proxy_port
    runner._app_config.paths.results_raw_dir = csv_dir


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _behave_debug(result) -> str:
    if result.behave is None:
        return "behave did not run"
    return f"exit={result.behave.exit_code}\n{result.behave.stdout}\n{result.behave.stderr}"


def test_validator_dry_run_supplies_sut_env_and_hides_api_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("INFLUX_URL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret")
    steps = tmp_path / "steps.py"
    steps.write_text(
        "import os\n"
        "from behave import given\n"
        "INFLUX_URL = os.environ['INFLUX_URL']\n"
        "assert 'GEMINI_API_KEY' not in os.environ, 'api key leaked into generated code'\n"
        "@given('a running InfluxDB instance')\n"
        "def step_running(context):\n"
        "    pass\n"
        "@given('the seed data is loaded')\n"
        "def step_seed(context):\n"
        "    pass\n",
        encoding="utf-8",
    )
    feature = tmp_path / "benchmark.feature"
    feature.write_text(
        "Feature: env\n  Scenario: import\n    Given a running InfluxDB instance\n",
        encoding="utf-8",
    )

    without_env = validate_artifacts(feature, steps)
    assert [issue.code for issue in without_env.issues] == ["IMPORT_ERROR"]
    assert "INFLUX_URL" in without_env.issues[0].message

    with_env = validate_artifacts(feature, steps, env={"INFLUX_URL": "http://127.0.0.1:8086"})
    assert with_env.syntax_ok is True, with_env.issues


def test_sanitized_environment_drops_credentials() -> None:
    from benchmark_agent.execution.behave_runner import sanitized_environment

    base = {
        "PATH": "/bin",
        "BENCHMARK_AGENT_GEMINI_API_KEY": "k",
        "GEMINI_API_KEY": "k",
        "MY_SECRET": "s",
        "INFLUX_TOKEN": "t",
    }
    assert sanitized_environment(base) == {"PATH": "/bin"}


def test_seed_lines_are_reanchored_to_now() -> None:
    from benchmark_agent.execution.sut import render_seed_lines

    text = (
        "# comment\n"
        "benchmark,source=seed value=1i 1785542400000000000\n"
        "\n"
        "benchmark,source=seed value=2i 1785542401000000000\n"
        "cpu,host=h2 usage=0.31 1785542400000000000\n"
    )
    now_ns = 2_000_000_000_000_000_000
    rendered = render_seed_lines(text, now_ns=now_ns).splitlines()
    assert rendered[0] == "# comment"
    assert rendered[2] == ""
    newest = now_ns - 1_000_000_000
    assert rendered[3].endswith(f" {newest}")
    assert rendered[1].endswith(f" {newest - 1_000_000_000}")
    assert rendered[4].endswith(f" {newest - 1_000_000_000}")
    assert rendered[1].startswith("benchmark,source=seed value=1i ")
    assert render_seed_lines("m v=1\n", now_ns=now_ns) == "m v=1\n"


def test_sut_environment_and_compose_project_name(tmp_path: Path) -> None:
    from benchmark_agent.config import SutSettings
    from benchmark_agent.execution.sut import compose_project_name, sut_environment

    env = sut_environment(SutSettings(), influx_url="http://127.0.0.1:8086")
    assert env["INFLUX_URL"] == env["INFLUXDB_URL"] == "http://127.0.0.1:8086"
    assert env["INFLUX_BUCKET"] == "benchmark"
    name_a = compose_project_name(tmp_path / "a")
    assert name_a.startswith("benchmark-agent-") and len(name_a) == len("benchmark-agent-") + 8
    assert name_a != compose_project_name(tmp_path / "b")
    assert name_a == compose_project_name(tmp_path / "a")
