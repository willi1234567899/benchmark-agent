from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchmark_agent.cli import app
from benchmark_agent.evaluation.runs import collect_runs
from benchmark_agent.models import MetricRow
from benchmark_agent.orchestration.quarantine import ARCHIVED_ROWS_NAME, quarantine_run
from benchmark_agent.orchestration.workspace import (
    QUARANTINE_DIR_NAME,
    QUARANTINE_FILE_NAME,
    WorkspaceManager,
)
from benchmark_agent.results_io import append_metric_rows, read_metric_rows

CONDITION = "L2_gemini_zeroshot"


def _row(run_id: str, operation: str = "write") -> MetricRow:
    return MetricRow(
        condition_id=7,
        condition_name=run_id.split("/")[0],
        guidance_level="L2",
        model="gemini",
        prompting="zero_shot",
        repetition=1,
        operation=operation,
        source="proxy",
        executed_ok=False,
        syntax_ok=False,
        semantics_ok=False,
        throughput_ops_s=0.0,
        n_requests=0,
        error_count=0,
        covered=False,
        timestamp="2026-08-27T00:00:00+00:00",
        run_id=run_id,
        evaluation_id="eval-1",
        status="generation_failed",
    )


def _make_run(artifacts: Path, condition: str, name: str) -> Path:
    run_directory = artifacts / condition / name
    run_directory.mkdir(parents=True)
    metadata = {
        "condition": {
            "id": 7,
            "name": condition,
            "guidance_level": "L2",
            "model": "gemini",
            "prompting_strategy": "zero_shot",
        },
        "status": "generation_failed",
        "error": "Model output was not valid JSON",
        "repetition": 1,
    }
    (run_directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (run_directory / "raw_response.json").write_text("{}", encoding="utf-8")
    return run_directory


def test_quarantine_moves_run_archives_rows_and_hides_it_from_the_dataset(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    csv_path = tmp_path / "raw" / "metrics.csv"
    run_directory = _make_run(artifacts, CONDITION, "run-001")
    _make_run(artifacts, CONDITION, "run-002")
    append_metric_rows(
        csv_path,
        [
            _row(f"{CONDITION}/run-001"),
            _row(f"{CONDITION}/run-001", "read"),
            _row(f"{CONDITION}/run-002"),
        ],
    )

    result = quarantine_run(
        run_directory,
        artifacts_dir=artifacts,
        csv_path=csv_path,
        reason="pre-fix JSON parser rejected a valid reply",
    )

    target = artifacts / QUARANTINE_DIR_NAME / CONDITION / "run-001"
    assert result.run_id == f"{CONDITION}/run-001"
    assert result.rows_removed == 2
    assert not run_directory.exists()
    assert (target / "raw_response.json").is_file(), "run directory moves intact"
    note = json.loads((target / QUARANTINE_FILE_NAME).read_text(encoding="utf-8"))
    assert note["run_id"] == f"{CONDITION}/run-001"
    assert note["reason"] == "pre-fix JSON parser rejected a valid reply"
    assert note["metrics_rows_removed"] == 2
    assert [row.run_id for row in read_metric_rows(csv_path)] == [f"{CONDITION}/run-002"]
    archived = read_metric_rows(target / ARCHIVED_ROWS_NAME)
    assert [row.operation for row in archived] == ["write", "read"]

    collection = collect_runs(csv_path=csv_path, artifacts_dir=artifacts)
    assert [run.run_id for run in collection.runs] == [f"{CONDITION}/run-002"]
    assert collection.quarantined == [
        {
            "run_id": f"{CONDITION}/run-001",
            "reason": "pre-fix JSON parser rejected a valid reply",
            "quarantined_at": note["quarantined_at"],
        }
    ]


def test_quarantine_without_csv_rows_still_moves_the_run(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    csv_path = tmp_path / "raw" / "metrics.csv"
    run_directory = _make_run(artifacts, CONDITION, "run-001")

    result = quarantine_run(
        run_directory, artifacts_dir=artifacts, csv_path=csv_path, reason="API quota"
    )

    assert result.rows_removed == 0
    assert not csv_path.exists()
    assert not (result.target / ARCHIVED_ROWS_NAME).exists()
    assert (result.target / QUARANTINE_FILE_NAME).is_file()


def test_quarantine_rejects_bad_targets_without_touching_anything(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    csv_path = tmp_path / "raw" / "metrics.csv"
    run_directory = _make_run(artifacts, CONDITION, "run-001")
    append_metric_rows(csv_path, [_row(f"{CONDITION}/run-001")])
    elsewhere = tmp_path / "elsewhere" / CONDITION / "run-001"
    elsewhere.mkdir(parents=True)
    fixture = _make_run(artifacts, "_gold_live", "run-001")

    with pytest.raises(ValueError, match="reason is required"):
        quarantine_run(run_directory, artifacts_dir=artifacts, csv_path=csv_path, reason="  ")
    with pytest.raises(ValueError, match="inside"):
        quarantine_run(elsewhere, artifacts_dir=artifacts, csv_path=csv_path, reason="x")
    with pytest.raises(ValueError, match="run-NNN"):
        quarantine_run(
            artifacts / CONDITION, artifacts_dir=artifacts, csv_path=csv_path, reason="x"
        )
    with pytest.raises(ValueError, match="fixture"):
        quarantine_run(fixture, artifacts_dir=artifacts, csv_path=csv_path, reason="x")
    with pytest.raises(FileNotFoundError):
        quarantine_run(
            artifacts / CONDITION / "run-999",
            artifacts_dir=artifacts,
            csv_path=csv_path,
            reason="x",
        )

    assert run_directory.is_dir()
    assert len(read_metric_rows(csv_path)) == 1
    assert not (artifacts / QUARANTINE_DIR_NAME).exists()


def test_quarantine_refuses_to_overwrite_an_existing_entry(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    csv_path = tmp_path / "raw" / "metrics.csv"
    run_directory = _make_run(artifacts, CONDITION, "run-001")
    (artifacts / QUARANTINE_DIR_NAME / CONDITION / "run-001").mkdir(parents=True)

    with pytest.raises(FileExistsError):
        quarantine_run(run_directory, artifacts_dir=artifacts, csv_path=csv_path, reason="x")
    assert run_directory.is_dir()


def test_workspace_never_reuses_a_quarantined_run_number(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    (artifacts / CONDITION / "run-001").mkdir(parents=True)
    (artifacts / QUARANTINE_DIR_NAME / CONDITION / "run-003").mkdir(parents=True)

    created = WorkspaceManager(artifacts).create_run_directory(CONDITION)

    assert created.name == "run-004"


def test_quarantine_cli_moves_runs_and_refreshes_the_dashboard(
    project_root: Path, isolated_artifacts: Path, isolated_results_raw: Path
) -> None:
    first = _make_run(isolated_artifacts, CONDITION, "run-001")
    second = _make_run(isolated_artifacts, CONDITION, "run-002")
    csv_path = isolated_results_raw / "metrics.csv"
    append_metric_rows(csv_path, [_row(f"{CONDITION}/run-001"), _row(f"{CONDITION}/run-002")])

    result = CliRunner().invoke(
        app,
        [
            "quarantine",
            "--run",
            str(first),
            "--run",
            str(second),
            "--reason",
            "Vertex 429 after retries",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert f"Quarantined {CONDITION}/run-001" in result.stdout
    assert f"Quarantined {CONDITION}/run-002" in result.stdout
    assert "Dashboard:" in result.stdout
    quarantine_root = isolated_artifacts / QUARANTINE_DIR_NAME / CONDITION
    assert (quarantine_root / "run-001" / QUARANTINE_FILE_NAME).is_file()
    assert (quarantine_root / "run-002" / QUARANTINE_FILE_NAME).is_file()
    assert read_metric_rows(csv_path) == []


def test_quarantine_cli_reports_failures(project_root: Path, isolated_artifacts: Path) -> None:
    result = CliRunner().invoke(
        app, ["quarantine", "--run", str(isolated_artifacts / "nope"), "--reason", "x"]
    )

    assert result.exit_code == 1
    assert "Quarantine failed" in (result.stdout + result.stderr)
