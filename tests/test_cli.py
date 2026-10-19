from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchmark_agent.cli import app

runner = CliRunner()


def test_generate_cli_runs_condition(
    project_root: Path, patch_llm_client: None, isolated_artifacts: Path
) -> None:
    result = runner.invoke(app, ["generate", "--condition", "L2_gemini_zeroshot"])

    assert result.exit_code == 0
    assert "Condition: L2_gemini_zeroshot" in result.stdout

    run_directory = _extract_run_directory(result.stdout)
    assert run_directory.is_relative_to(isolated_artifacts)


def test_generate_cli_reports_unknown_condition_without_quotes(project_root: Path) -> None:
    result = runner.invoke(app, ["generate", "--condition", "NOPE"])

    assert result.exit_code == 2
    output = result.stdout + result.stderr
    assert "Unknown condition: NOPE" in output
    assert "'Unknown condition: NOPE'" not in output


def test_generate_cli_l1_requires_reachable_sut(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "benchmark_agent.execution.sut.SutManager.is_reachable",
        lambda self, *, timeout_seconds=2.0: False,
    )
    result = runner.invoke(app, ["generate", "--condition", "L1_gemini_zeroshot"])

    assert result.exit_code == 1
    assert "L1 probing needs a reachable SUT" in (result.stdout + result.stderr)


def test_generate_cli_reports_configuration_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["generate", "--condition", "L2_gemini_zeroshot"])

    assert result.exit_code == 2
    assert "Configuration error (run from the project root?)" in (result.stdout + result.stderr)


def test_evaluate_cli_rejects_directory_without_artifacts(
    project_root: Path, tmp_path: Path
) -> None:
    result = runner.invoke(app, ["evaluate", "--run", str(tmp_path), "--no-start-sut"])

    assert result.exit_code == 2
    assert "does not contain benchmark.feature" in (result.stdout + result.stderr)


def test_evaluate_cli_records_generation_failed_run(
    project_root: Path, tmp_path: Path, isolated_results_raw: Path
) -> None:
    import json

    from benchmark_agent.results_io import read_metric_rows

    run_directory = tmp_path / "L2_gemini_zeroshot" / "run-000"
    run_directory.mkdir(parents=True)
    (run_directory / "metadata.json").write_text(
        json.dumps({"status": "generation_failed", "error": "no JSON", "seed": 7}),
        encoding="utf-8",
    )

    result = runner.invoke(
        app, ["evaluate", "--run", str(run_directory), "--no-start-sut", "--no-measure"]
    )

    assert result.exit_code == 1, result.stdout + result.stderr
    assert "Status: generation_failed" in result.stdout
    assert "Execution skipped" in result.stdout
    rows = read_metric_rows(isolated_results_raw / "metrics.csv")
    assert len(rows) == 3
    assert all(row.status == "generation_failed" and row.seed == 7 for row in rows)


def test_generate_cli_prints_run_directory_on_failure(
    project_root: Path, isolated_artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakes import ProseThenValidFakeLLMClient

    def _factory(model_config, *, gemini_api_key=None, **_ignored):
        return ProseThenValidFakeLLMClient(model_config)

    monkeypatch.setattr("benchmark_agent.agent.agent.create_llm_client", _factory)
    result = runner.invoke(app, ["generate", "--condition", "L2_gemini_zeroshot"])

    assert result.exit_code == 1
    output = result.stdout + result.stderr
    assert "Run directory: " in output
    assert _extract_run_directory(output).is_relative_to(isolated_artifacts)


def test_llm_check_cli_rejects_unknown_model(project_root: Path) -> None:
    result = runner.invoke(app, ["llm-check", "--model", "nope"])

    assert result.exit_code == 2
    assert "Unknown model: nope" in (result.stdout + result.stderr)


def test_evaluate_cli_skips_invalid_artifacts(project_root: Path, tmp_path: Path) -> None:
    run_directory = tmp_path / "run-001"
    run_directory.mkdir()
    (run_directory / "benchmark.feature").write_text(
        "Feature: broken\n  Scenario: x\n    Given nowhere\n",
        encoding="utf-8",
    )
    (run_directory / "steps.py").write_text("def broken(\n", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--run",
            str(run_directory),
            "--no-start-sut",
            "--no-measure",
        ],
    )

    assert result.exit_code == 1
    assert "Status: " in result.stdout
    assert "Syntax OK: False" in result.stdout
    assert "Execution skipped" in result.stdout


def _extract_run_directory(output: str) -> Path:
    for line in output.splitlines():
        if line.startswith("Run directory: "):
            return Path(line.removeprefix("Run directory: ").strip())
    raise AssertionError("Run directory was not printed by the CLI.")


def test_llm_check_cli_without_key(project_root: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "benchmark_agent.llm.base.resolve_gemini_api_key",
        lambda configured=None: None,
    )
    result = runner.invoke(app, ["llm-check"])

    assert result.exit_code == 1
    assert "API key missing" in (result.stdout + result.stderr)


def test_plot_seed_cli_writes_figures(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plot-seed", "--output", str(tmp_path)])

    assert result.exit_code == 0, result.stdout + result.stderr
    assert (tmp_path / "01_triptych_ratio.png").is_file()
    assert (tmp_path / "12_board.png").is_file()
    assert (tmp_path / "tables" / "cell_summary.csv").is_file()
    assert (tmp_path / "tables" / "inferential.txt").is_file()
    assert "synthetic seed data, not measured" in result.stdout


def test_plot_seed_default_output_is_package_anchored() -> None:
    from benchmark_agent.cli import PACKAGE_ROOT, SEED_REPORT_DIR

    assert SEED_REPORT_DIR.is_absolute()
    assert (PACKAGE_ROOT / "pyproject.toml").is_file()
    assert SEED_REPORT_DIR == PACKAGE_ROOT / "results" / "reports" / "seed"


def test_sut_status_cli_reports_unreachable(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCHMARK_AGENT_SUT__INFLUXDB_DIRECT_URL", "http://127.0.0.1:9")
    result = runner.invoke(app, ["sut", "status"])

    assert result.exit_code == 1
    assert "Compose project: benchmark-agent-" in result.stdout
    assert "Reachable: False" in result.stdout
