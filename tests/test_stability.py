from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from benchmark_agent.cli import app
from benchmark_agent.orchestration.stability import (
    BUDGET_STOP_STATUS,
    classify_outcome,
    consecutive_clean,
    format_stability,
    load_stability,
)


def _outcome(**fields: object) -> dict[str, object]:
    base: dict[str, object] = {
        "condition_name": "L2_gemini_zeroshot",
        "repetition": 1,
        "generation_status": "ok",
        "evaluation_status": "ok",
        "success": True,
        "error": None,
    }
    base.update(fields)
    return base


def test_classify_outcome_separates_model_from_harness() -> None:
    assert classify_outcome(_outcome()) == "success"
    assert classify_outcome(_outcome(success=False, evaluation_status="ok")) == "model_failure"
    assert (
        classify_outcome(
            _outcome(
                success=False,
                generation_status="generation_failed",
                error="Model output violated the JSON contract: Expecting value",
            )
        )
        == "model_failure"
    )
    assert (
        classify_outcome(
            _outcome(
                success=False,
                generation_status="generation_failed",
                error="Gemini API call failed (gemini-3.5-flash) after 8 attempt(s): 503",
            )
        )
        == "harness_failure"
    )
    assert (
        classify_outcome(_outcome(success=None, evaluation_status="infrastructure_failed"))
        == "harness_failure"
    )
    assert (
        classify_outcome(
            _outcome(success=None, evaluation_status=BUDGET_STOP_STATUS, error="budget guard: x")
        )
        == "skipped"
    )


def _write_matrix(raw_dir: Path, stamp: str, outcomes: list[dict[str, object]]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"matrix_{stamp}.json").write_text(
        json.dumps({"timestamp": stamp, "outcomes": outcomes}), encoding="utf-8"
    )


def test_load_stability_counts_and_consecutive_clean(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _write_matrix(raw, "20260828T100000Z", [_outcome(), _outcome(success=False)])
    _write_matrix(
        raw,
        "20260828T110000Z",
        [_outcome(), _outcome(success=None, evaluation_status="infrastructure_failed")],
    )
    _write_matrix(raw, "20260828T120000Z", [_outcome(), _outcome(repetition=2)])
    _write_matrix(raw, "20260828T130000Z", [_outcome(success=False, error="behave failed")])
    (raw / "matrix_broken.json").write_text("{", encoding="utf-8")

    reports = load_stability(raw)

    assert [report.file for report in reports] == [
        "matrix_20260828T130000Z.json",
        "matrix_20260828T120000Z.json",
        "matrix_20260828T110000Z.json",
        "matrix_20260828T100000Z.json",
    ]
    assert [report.clean for report in reports] == [True, True, False, True]
    assert reports[2].n_harness_failures == 1 and reports[2].harness_errors
    assert reports[3].n_model_failures == 1
    assert consecutive_clean(reports) == 2
    text = format_stability(reports, required=2)
    assert "2 of 2 required -> STABLE" in text
    assert "infrastructure_failed" in text
    assert consecutive_clean(reports[2:]) == 0
    assert "not yet stable" in format_stability(reports[2:], required=2)
    assert format_stability([]) == "No matrix reports found."


def test_stability_cli(project_root: Path, tmp_path: Path) -> None:
    _write_matrix(tmp_path / "raw", "20260828T140000Z", [_outcome()])
    result = CliRunner().invoke(app, ["stability", "--required", "1"])
    assert result.exit_code == 0, result.output
    assert "STABLE" in result.output


def test_cost_cli_runs_on_the_live_config(project_root: Path) -> None:
    result = CliRunner().invoke(app, ["cost"])
    assert result.exit_code == 0, result.output
    assert "Estimated total" in result.output
