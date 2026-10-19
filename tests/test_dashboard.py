from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchmark_agent.cli import app
from benchmark_agent.dashboard import build_payload, render_html
from benchmark_agent.dashboard.build import build_dashboard, ceiling_summary, load_matrix_reports
from benchmark_agent.dashboard.render import DATA_ELEMENT_ID, embed_json
from benchmark_agent.dashboard.serve import create_server
from benchmark_agent.evaluation.runs import (
    STATUS_NOT_EVALUATED,
    OpStats,
    RunCollection,
    RunRecord,
    collect_runs,
)
from benchmark_agent.evaluation.seed_runs import seed_run_records
from benchmark_agent.evaluation.statistics import (
    compute_statistics,
    funnel_counts,
    iqr,
    rank_biserial,
)
from benchmark_agent.models import MetricRow
from benchmark_agent.results_io import append_metric_rows

runner = CliRunner()
OPERATIONS = ("write", "read", "delete")


def _row(
    run_id: str,
    operation: str,
    *,
    source: str = "proxy",
    level: str = "L2",
    model: str = "gemini",
    prompting: str = "zero_shot",
    executed_ok: bool = True,
    semantics_ok: bool = True,
    throughput: float = 50.0,
    n_requests: int = 10,
    error_count: int = 0,
    status: str = "ok",
    evaluation_id: str = "2026-08-27T10:00:00+00:00",
) -> MetricRow:
    return MetricRow(
        condition_id=7,
        condition_name=f"{level}_{model}_{prompting.replace('_', '')}",
        guidance_level=level,
        model=model,
        prompting=prompting,
        repetition=int(run_id.rsplit("-", 1)[-1]),
        operation=operation,
        source=source,
        executed_ok=executed_ok,
        syntax_ok=True,
        semantics_ok=semantics_ok,
        throughput_ops_s=throughput,
        latency_p50_ms=2.0 if source == "proxy" else None,
        latency_p95_ms=5.0 if source == "proxy" else None,
        latency_p99_ms=7.0 if source == "proxy" else None,
        n_requests=n_requests,
        error_count=error_count,
        covered=executed_ok and n_requests - error_count > 0,
        feedback_iterations=0,
        model_version="gemini-3.5-flash",
        seed=1,
        timestamp=evaluation_id,
        run_id=run_id,
        evaluation_id=evaluation_id,
        status=status,
    )


def _write_results_tree(root: Path) -> tuple[Path, Path]:
    csv_path = root / "raw" / "metrics.csv"
    rows: list[MetricRow] = []
    for operation in OPERATIONS:
        rows.append(_row("L2_gemini_zeroshot/run-001", operation, throughput=40.0))
        rows.append(_row("L2_gemini_zeroshot/run-001", operation, source="server", throughput=41.0))
    for operation in OPERATIONS:
        rows.append(
            _row(
                "L3_gemini_zeroshot/run-001",
                operation,
                level="L3",
                executed_ok=False,
                semantics_ok=False,
                throughput=0.0,
                n_requests=0,
                status="execution_failed",
            )
        )
    append_metric_rows(csv_path, rows)

    artifacts = root / "artifacts"
    evaluated = artifacts / "L2_gemini_zeroshot" / "run-001"
    evaluated.mkdir(parents=True)
    (evaluated / "metadata.json").write_text(
        json.dumps(
            {
                "condition": {
                    "id": 7,
                    "name": "L2_gemini_zeroshot",
                    "guidance_level": "L2",
                    "model": "gemini",
                    "prompting_strategy": "zero_shot",
                },
                "model": {"model_version": "gemini-3.5-flash"},
                "status": "ok",
                "feedback_iterations": 0,
                "attempt": 0,
                "probe_calls": 0,
                "repetition": 1,
                "seed": 1,
                "timestamp": "2026-08-27T09:55:00+00:00",
                "usage": {"total_token_count": 1234},
            }
        ),
        encoding="utf-8",
    )
    (evaluated / "evaluation").mkdir()
    (evaluated / "evaluation" / "behave.json").write_text(
        json.dumps(
            {
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "timed_out": False,
                "scenarios_passed": 3,
                "scenarios_failed": 0,
                "scenarios_error": 0,
            }
        ),
        encoding="utf-8",
    )
    pending = artifacts / "L4_gemini_zeroshot" / "run-002"
    pending.mkdir(parents=True)
    (pending / "metadata.json").write_text(
        json.dumps(
            {
                "condition": {
                    "id": 19,
                    "name": "L4_gemini_zeroshot",
                    "guidance_level": "L4",
                    "model": "gemini",
                    "prompting_strategy": "zero_shot",
                },
                "status": "ok",
                "feedback_iterations": 2,
                "attempt": 2,
                "repetition": 2,
                "timestamp": "2026-08-27T11:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (pending / "benchmark.feature").write_text("Feature: x\n", encoding="utf-8")
    (pending / "steps.py").write_text("", encoding="utf-8")
    (artifacts / "_fake_not_gemini" / "L2_gemini_zeroshot_run-001").mkdir(parents=True)
    return csv_path, artifacts


def test_collect_runs_merges_csv_and_artifacts(tmp_path: Path) -> None:
    csv_path, artifacts = _write_results_tree(tmp_path)

    collection = collect_runs(csv_path=csv_path, artifacts_dir=artifacts, project_root=tmp_path)

    assert collection.issues == []
    assert [run.run_id for run in collection.runs] == [
        "L2_gemini_zeroshot/run-001",
        "L3_gemini_zeroshot/run-001",
        "L4_gemini_zeroshot/run-002",
    ]
    measured = collection.runs[0]
    assert measured.success is True
    assert measured.coverage_ratio == 1.0
    assert measured.covered_operations == ["write", "read", "delete"]
    assert measured.op("proxy", "write") is not None
    assert measured.op("server", "write") is not None
    assert measured.stage_b_value("proxy", "write", "throughput_ops_s") == 40.0
    assert measured.scenarios_passed == 3
    assert measured.usage == {"total_token_count": 1234}
    assert measured.attempts == 1
    assert measured.run_directory == "artifacts/L2_gemini_zeroshot/run-001"

    failed = collection.runs[1]
    assert failed.status == "execution_failed"
    assert failed.stage_b_value("proxy", "write", "throughput_ops_s") is None

    pending = collection.runs[2]
    assert pending.status == STATUS_NOT_EVALUATED
    assert pending.feedback_iterations == 2
    assert pending.repetition == 2
    assert len(collection.measured) == 2


def test_collect_runs_reports_broken_csv_instead_of_raising(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("condition_name,operation\nx,write\n", encoding="utf-8")

    collection = collect_runs(csv_path=csv_path, artifacts_dir=None)

    assert collection.runs == []
    assert collection.issues and "metrics.csv" in collection.issues[0]


def test_collect_runs_uses_summary_when_csv_rows_are_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "L2_gemini_zeroshot" / "run-003"
    (run_dir / "evaluation").mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"condition": {"name": "L2_gemini_zeroshot", "guidance_level": "L2"}}),
        encoding="utf-8",
    )
    (run_dir / "evaluation" / "summary.json").write_text(
        json.dumps(
            {
                "status": "validation_failed",
                "success": {"executed_ok": False, "syntax_ok": False, "semantics_ok": False},
                "validation": {"syntax_ok": False, "issues": [{"message": "bad gherkin"}]},
                "skip_reason": "validation failed",
            }
        ),
        encoding="utf-8",
    )

    collection = collect_runs(csv_path=None, artifacts_dir=tmp_path)

    (run,) = collection.runs
    assert run.status == "validation_failed"
    assert run.validation_issues == ["bad gherkin"]
    assert run.skip_reason == "validation failed"
    assert run.repetition == 3


def test_statistics_definitions_on_small_sample(tmp_path: Path) -> None:
    csv_path, artifacts = _write_results_tree(tmp_path)
    runs = collect_runs(csv_path=csv_path, artifacts_dir=artifacts).runs

    report = compute_statistics(runs, operation="write", source="proxy")

    assert report.n_measured == 2
    assert report.funnel == {"evaluated": 2, "syntax_ok": 2, "executed_ok": 1, "semantics_ok": 1}
    assert report.overall.success_rate == 0.5
    assert report.overall.n_stage_b == 1
    assert report.overall.median_throughput == 40.0
    assert report.overall.naive_mean_throughput == 20.0
    by_level = {group.key: group for group in report.by_level}
    assert by_level["L2"].n_success == 1 and by_level["L3"].n_success == 0
    assert all(
        test.applicable is False for test in report.tests if test.name != "Proxy vs. /metrics"
    )
    proxy_check = next(test for test in report.tests if test.name == "Proxy vs. /metrics")
    assert proxy_check.applicable is True and proxy_check.n == 1
    assert report.notes and "Nur 2" in report.notes[0]


def test_statistics_tests_run_on_seed_matrix() -> None:
    runs = seed_run_records()
    assert len(runs) == 5 * 3 * 2 * 8
    assert all(run.synthetic for run in runs)

    report = compute_statistics(runs, operation="write", source="proxy")

    tests = {test.name: test for test in report.tests}
    for name in ("Kruskal-Wallis", "Spearman", "Mann-Whitney U", "Chi-Quadrat"):
        assert tests[name].applicable, name
        p_value = tests[name].p_value
        assert p_value is not None
        assert 0.0 <= p_value <= 1.0
    assert tests["Kruskal-Wallis"].groups == ["L1", "L2", "L3", "L4", "L5"]
    assert tests["Spearman"].statistic is not None and tests["Spearman"].statistic > 0.5
    assert len(report.cells) == 30
    assert report.cells[0].guidance_level == "L1"


def test_statistics_excludes_fixtures_by_default() -> None:
    fixture = RunRecord(
        run_id="fixtures/gold_live",
        condition_name="_gold_live",
        is_fixture=True,
        status="ok",
        syntax_ok=True,
        executed_ok=True,
        semantics_ok=True,
        success=True,
    )
    model_run = RunRecord(
        run_id="L2_gemini_zeroshot/run-001",
        condition_name="L2_gemini_zeroshot",
        guidance_level="L2",
        status="execution_failed",
        syntax_ok=True,
    )

    report = compute_statistics([fixture, model_run])
    assert report.n_runs == 1 and report.overall.n_success == 0
    report_all = compute_statistics([fixture, model_run], include_fixtures=True)
    assert report_all.n_runs == 2 and report_all.overall.n_success == 1


def test_ceiling_summary_uses_only_gold_fixture_runs() -> None:
    def gold_run(run_id: str, write_throughput: float) -> RunRecord:
        return RunRecord(
            run_id=run_id,
            condition_name="_gold_tessera",
            is_fixture=True,
            guidance_level=None,
            status="ok",
            measurements={
                "proxy": {
                    "write": OpStats(
                        n_requests=10,
                        throughput_ops_s=write_throughput,
                        latency_p50_ms=20.0,
                        latency_p95_ms=30.0,
                        latency_p99_ms=40.0,
                    )
                }
            },
        )

    gold_runs = [gold_run("_gold_tessera/run-001", 40.0), gold_run("_gold_tessera/run-002", 60.0)]
    unmeasured_gold = RunRecord(
        run_id="_gold_tessera/run-003",
        condition_name="_gold_tessera",
        is_fixture=True,
        status="not_evaluated",
    )
    model_run = RunRecord(
        run_id="L2_gemini_zeroshot/run-001",
        condition_name="L2_gemini_zeroshot",
        guidance_level="L2",
        model="gemini",
        status="ok",
        measurements={"proxy": {"write": OpStats(n_requests=10, throughput_ops_s=999.0)}},
    )

    summary = ceiling_summary([*gold_runs, unmeasured_gold, model_run])

    assert summary["proxy"]["write"]["n"] == 2
    assert summary["proxy"]["write"]["median_throughput_ops_s"] == pytest.approx(50.0)
    assert summary["proxy"]["write"]["median_latency_p95_ms"] == pytest.approx(30.0)
    assert "read" not in summary["proxy"]


def test_statistics_helpers() -> None:
    assert iqr([1.0]) is None
    assert iqr([1.0, 2.0, 3.0, 4.0]) == pytest.approx(1.5)
    assert rank_biserial(6.0, 3, 2) == 1.0
    assert rank_biserial(0.0, 3, 2) == -1.0
    assert funnel_counts([]) == {
        "evaluated": 0,
        "syntax_ok": 0,
        "executed_ok": 0,
        "semantics_ok": 0,
    }


def test_embed_json_is_script_safe() -> None:
    text = embed_json({"x": "</script><!--  "})
    assert "</script" not in text
    assert "<!--" not in text
    assert json.loads(text) == {"x": "</script><!--  "}


def test_render_html_embeds_payload_and_synthetic_banner(tmp_path: Path) -> None:
    csv_path, artifacts = _write_results_tree(tmp_path)
    collection = collect_runs(csv_path=csv_path, artifacts_dir=artifacts, project_root=tmp_path)

    payload = build_payload(collection, generated_at="2026-08-27T12:00:00+00:00")
    html = render_html(payload)

    assert f'id="{DATA_ELEMENT_ID}"' in html
    assert "L2_gemini_zeroshot/run-001" in html
    assert "SYNTHETISCHE SEED-DATEN" not in html
    assert 'http-equiv="refresh"' not in html
    assert set(payload["stats"]) == {
        f"{op}|{source}" for op in OPERATIONS for source in ("proxy", "server")
    }
    assert payload["levels"] == ["L2", "L3", "L4"]
    assert payload["statuses"] == ["ok", "execution_failed", "not_evaluated"]

    synthetic = render_html(
        build_payload(RunCollection(runs=seed_run_records()), synthetic=True), refresh_seconds=15
    )
    assert "SYNTHETISCHE SEED-DATEN" in synthetic
    assert 'http-equiv="refresh" content="15"' in synthetic


def test_build_dashboard_writes_html_and_tables(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path, artifacts = _write_results_tree(tmp_path)
    monkeypatch.setenv("BENCHMARK_AGENT_PATHS__RESULTS_RAW_DIR", str(csv_path.parent))
    monkeypatch.setenv("BENCHMARK_AGENT_PATHS__RESULTS_ARTIFACTS_DIR", str(artifacts))
    (csv_path.parent / "matrix_20260827T100000Z.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-08-27T10:00:00+00:00",
                "outcomes": [
                    {"condition_name": "L2_gemini_zeroshot", "success": True},
                    {"condition_name": "L3_gemini_zeroshot", "success": False, "error": "boom"},
                ],
            }
        ),
        encoding="utf-8",
    )

    built = build_dashboard(output_dir=tmp_path / "out")

    assert built.html_path == (tmp_path / "out" / "dashboard.html").resolve()
    assert built.html_path.is_file()
    assert built.n_runs == 3 and built.n_measured == 2
    names = {path.name for path in built.table_paths}
    assert names == {"runs.csv", "cells.csv", "statistics.json"}
    runs_csv = (tmp_path / "out" / "runs.csv").read_text(encoding="utf-8")
    assert "proxy_write_throughput_ops_s" in runs_csv.splitlines()[0]
    assert "L4_gemini_zeroshot/run-002" in runs_csv
    cells_csv = (tmp_path / "out" / "cells.csv").read_text(encoding="utf-8").splitlines()
    assert cells_csv[0].endswith("operation,source,synthetic")
    assert len(cells_csv) == 3
    matrices = load_matrix_reports(csv_path.parent)
    assert matrices[0]["n_runs"] == 2 and matrices[0]["n_failed"] == 1
    assert matrices[0]["errors"] == ["boom"]


def test_dashboard_cli_seed_demo_marks_synthetic(project_root: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["dashboard", "--seed-demo"])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert "SYNTHETIC seed data" in result.stdout
    assert not (tmp_path / "reports" / "dashboard").exists()
    seed_dir = tmp_path / "reports" / "seed" / "dashboard"
    html = (seed_dir / "dashboard.html").read_text(encoding="utf-8")
    assert "SYNTHETISCHE SEED-DATEN" in html
    cells = (seed_dir / "cells.csv").read_text(encoding="utf-8").splitlines()
    assert all(line.endswith(",True") for line in cells[1:])


def test_dashboard_cli_default_output_is_isolated_reports_dir(
    project_root: Path, tmp_path: Path
) -> None:
    result = runner.invoke(app, ["dashboard", "--no-tables"])

    assert result.exit_code == 0, result.stdout + result.stderr
    expected = tmp_path / "reports" / "dashboard" / "dashboard.html"
    assert expected.is_file()
    assert str(expected.resolve()) in result.stdout
    assert not (tmp_path / "reports" / "dashboard" / "runs.csv").exists()


def test_run_cli_refreshes_dashboard(
    project_root: Path, patch_llm_client: None, isolated_artifacts: Path, tmp_path: Path
) -> None:
    result = runner.invoke(
        app, ["run", "--condition", "L2_gemini_zeroshot", "--no-start-sut", "--no-measure"]
    )

    output = result.stdout + result.stderr
    assert "Dashboard: " in output, output
    html_path = tmp_path / "reports" / "dashboard" / "dashboard.html"
    assert html_path.is_file()
    assert "L2_gemini_zeroshot/run-001" in html_path.read_text(encoding="utf-8")


def test_serve_rebuilds_on_each_request() -> None:
    counter = {"n": 0}

    def build_html() -> str:
        counter["n"] += 1
        return f"<html><body>build {counter['n']}</body></html>"

    server = create_server(build_html, port=0, build_payload=lambda: {"n": counter["n"]})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        first = urllib.request.urlopen(f"{base}/").read().decode("utf-8")
        second = urllib.request.urlopen(f"{base}/dashboard.html").read().decode("utf-8")
        data = json.loads(urllib.request.urlopen(f"{base}/data.json").read().decode("utf-8"))
        health = urllib.request.urlopen(f"{base}/healthz").read()
    finally:
        server.shutdown()
        server.server_close()

    assert first == "<html><body>build 1</body></html>"
    assert second == "<html><body>build 2</body></html>"
    assert data == {"n": 2}
    assert health == b"ok"
