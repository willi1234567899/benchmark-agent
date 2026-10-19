from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import PROJECT_ROOT
from fakes import FakeLLMClient
from typer.testing import CliRunner

from benchmark_agent.cli import app
from benchmark_agent.config import load_app_config
from benchmark_agent.llm.base import LLMClient
from benchmark_agent.models import (
    STATUS_GENERATION_FAILED,
    STATUS_VALIDATION_FAILED,
    ConditionConfig,
    GuidanceLevel,
    LLMRequest,
    LLMResponse,
    ModelConfig,
    PromptingStrategy,
)
from benchmark_agent.orchestration.matrix import MatrixRunner, RunOutcome, expand_conditions


class InvalidJsonFakeLLMClient(LLMClient):
    def __init__(self, model_config: ModelConfig) -> None:
        self._model_config = model_config

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            provider=self._model_config.provider,
            model_name=self._model_config.model_name,
            model_version=self._model_config.model_version,
            content="this is not json {",
        )


@pytest.fixture
def results_raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw_dir = tmp_path / "raw"
    monkeypatch.setenv("BENCHMARK_AGENT_PATHS__RESULTS_RAW_DIR", str(raw_dir))
    return raw_dir


def _run_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*/run-*") if path.is_dir())


def test_matrix_runs_two_conditions_twice(
    project_root: Path,
    patch_llm_client: None,
    isolated_artifacts: Path,
    results_raw_dir: Path,
) -> None:
    matrix = MatrixRunner(project_root)
    conditions = [
        matrix.app_config.resolve_condition("L2_gemini_zeroshot"),
        matrix.app_config.resolve_condition("L3_gemini_multishot"),
    ]

    outcomes = matrix.run(conditions, 2, start_sut=False, stop_sut=False, measure=False)

    assert len(outcomes) == 4
    assert [(outcome.condition_name, outcome.repetition) for outcome in outcomes] == [
        ("L2_gemini_zeroshot", 1),
        ("L2_gemini_zeroshot", 2),
        ("L3_gemini_multishot", 1),
        ("L3_gemini_multishot", 2),
    ]
    assert len(_run_dirs(isolated_artifacts)) == 4
    for outcome in outcomes:
        assert outcome.run_directory is not None
        assert (outcome.run_directory / "evaluation" / "summary.json").is_file()
        assert outcome.generation_status == "ok"
        assert outcome.coverage_ratio is None

    report = matrix.write_outcomes(outcomes)
    assert report.parent == results_raw_dir
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert len(payload["outcomes"]) == 4


def test_matrix_records_generation_failure_and_continues(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_artifacts: Path,
    results_raw_dir: Path,
) -> None:
    calls = {"count": 0}

    def _factory(model_config, *, gemini_api_key=None, **_ignored):
        calls["count"] += 1
        if calls["count"] == 1:
            return InvalidJsonFakeLLMClient(model_config)
        return FakeLLMClient(model_config)

    monkeypatch.setattr("benchmark_agent.agent.agent.create_llm_client", _factory)
    matrix = MatrixRunner(project_root)
    condition = matrix.app_config.resolve_condition("L2_gemini_zeroshot")

    outcomes = matrix.run([condition], 2, start_sut=False, stop_sut=False, measure=False)

    assert len(outcomes) == 2
    failed, healthy = outcomes
    assert failed.generation_status in {STATUS_GENERATION_FAILED, STATUS_VALIDATION_FAILED}
    assert failed.success is not True
    assert failed.error
    assert failed.run_directory is not None
    metadata = json.loads((failed.run_directory / "metadata.json").read_text(encoding="utf-8-sig"))
    assert metadata["status"] == failed.generation_status
    assert healthy.generation_status == "ok"
    assert healthy.repetition == 2


def test_run_outcome_ok_requires_success() -> None:
    def outcome(success: bool | None) -> RunOutcome:
        return RunOutcome(
            condition_name="c",
            repetition=1,
            generation_status="ok",
            evaluation_status="ok",
            success=success,
        )

    assert outcome(True).ok is True
    assert outcome(False).ok is False
    assert outcome(None).ok is False


def test_expand_conditions_matches_documented_naming_and_ids() -> None:
    expanded = expand_conditions(["gemini", "model_b", "model_c"])

    assert len(expanded) == 30
    assert [cond.id for cond in expanded] == list(range(1, 31))
    assert expanded[0] == ConditionConfig(
        id=1,
        name="L1_gemini_zeroshot",
        guidance_level=GuidanceLevel.L1,
        model="gemini",
        prompting_strategy=PromptingStrategy.ZERO_SHOT,
    )
    assert expanded[1].name == "L1_gemini_multishot"
    assert expanded[6].name == "L2_gemini_zeroshot"
    assert expanded[29].name == "L5_model_c_multishot"
    assert len({cond.name for cond in expanded}) == 30


def test_expand_conditions_uses_configured_model_keys() -> None:
    app_config = load_app_config(PROJECT_ROOT)
    expanded = expand_conditions(app_config.models.models)
    names = {cond.name for cond in expanded}
    for configured in app_config.conditions.conditions:
        assert configured.name in names


_FROZEN_LEGACY_MATRIX: list[tuple[int, str, str, str, str]] = [
    (1, "L1_gemini_zeroshot", "L1", "gemini", "zero_shot"),
    (2, "L1_gemini_multishot", "L1", "gemini", "multi_shot"),
    (3, "L1_gemini_lite_zeroshot", "L1", "gemini_lite", "zero_shot"),
    (4, "L1_gemini_lite_multishot", "L1", "gemini_lite", "multi_shot"),
    (5, "L1_model_c_zeroshot", "L1", "model_c", "zero_shot"),
    (6, "L1_model_c_multishot", "L1", "model_c", "multi_shot"),
    (7, "L2_gemini_zeroshot", "L2", "gemini", "zero_shot"),
    (8, "L2_gemini_multishot", "L2", "gemini", "multi_shot"),
    (9, "L2_gemini_lite_zeroshot", "L2", "gemini_lite", "zero_shot"),
    (10, "L2_gemini_lite_multishot", "L2", "gemini_lite", "multi_shot"),
    (11, "L2_model_c_zeroshot", "L2", "model_c", "zero_shot"),
    (12, "L2_model_c_multishot", "L2", "model_c", "multi_shot"),
    (13, "L3_gemini_zeroshot", "L3", "gemini", "zero_shot"),
    (14, "L3_gemini_multishot", "L3", "gemini", "multi_shot"),
    (15, "L3_gemini_lite_zeroshot", "L3", "gemini_lite", "zero_shot"),
    (16, "L3_gemini_lite_multishot", "L3", "gemini_lite", "multi_shot"),
    (17, "L3_model_c_zeroshot", "L3", "model_c", "zero_shot"),
    (18, "L3_model_c_multishot", "L3", "model_c", "multi_shot"),
    (19, "L4_gemini_zeroshot", "L4", "gemini", "zero_shot"),
    (20, "L4_gemini_multishot", "L4", "gemini", "multi_shot"),
    (21, "L4_gemini_lite_zeroshot", "L4", "gemini_lite", "zero_shot"),
    (22, "L4_gemini_lite_multishot", "L4", "gemini_lite", "multi_shot"),
    (23, "L4_model_c_zeroshot", "L4", "model_c", "zero_shot"),
    (24, "L4_model_c_multishot", "L4", "model_c", "multi_shot"),
    (25, "L5_gemini_zeroshot", "L5", "gemini", "zero_shot"),
    (26, "L5_gemini_multishot", "L5", "gemini", "multi_shot"),
    (27, "L5_gemini_lite_zeroshot", "L5", "gemini_lite", "zero_shot"),
    (28, "L5_gemini_lite_multishot", "L5", "gemini_lite", "multi_shot"),
    (29, "L5_model_c_zeroshot", "L5", "model_c", "zero_shot"),
    (30, "L5_model_c_multishot", "L5", "model_c", "multi_shot"),
]


def _brief(conditions: list[ConditionConfig]) -> list[tuple[int, str, str, str, str]]:
    return [
        (
            condition.id,
            condition.name,
            condition.guidance_level.value,
            condition.model,
            condition.prompting_strategy.value,
        )
        for condition in conditions
    ]


def test_expand_conditions_first_three_models_are_byte_identical_to_frozen_baseline() -> None:
    expanded = expand_conditions(["gemini", "gemini_lite", "model_c"])

    assert len(expanded) == 30
    assert _brief(expanded) == _FROZEN_LEGACY_MATRIX


def test_expand_conditions_fourth_model_gets_trailing_block() -> None:
    expanded = expand_conditions(["gemini", "gemini_lite", "model_c", "model_d"])

    assert len(expanded) == 40
    assert _brief(expanded[:30]) == _FROZEN_LEGACY_MATRIX
    assert _brief(expanded[30:]) == [
        (31, "L1_model_d_zeroshot", "L1", "model_d", "zero_shot"),
        (32, "L1_model_d_multishot", "L1", "model_d", "multi_shot"),
        (33, "L2_model_d_zeroshot", "L2", "model_d", "zero_shot"),
        (34, "L2_model_d_multishot", "L2", "model_d", "multi_shot"),
        (35, "L3_model_d_zeroshot", "L3", "model_d", "zero_shot"),
        (36, "L3_model_d_multishot", "L3", "model_d", "multi_shot"),
        (37, "L4_model_d_zeroshot", "L4", "model_d", "zero_shot"),
        (38, "L4_model_d_multishot", "L4", "model_d", "multi_shot"),
        (39, "L5_model_d_zeroshot", "L5", "model_d", "zero_shot"),
        (40, "L5_model_d_multishot", "L5", "model_d", "multi_shot"),
    ]


def test_expand_conditions_fifth_model_stacks_another_trailing_block() -> None:
    expanded = expand_conditions(["gemini", "gemini_lite", "model_c", "model_d", "model_e"])

    assert len(expanded) == 50
    assert _brief(expanded[:30]) == _FROZEN_LEGACY_MATRIX
    assert [condition.id for condition in expanded[40:]] == list(range(41, 51))
    assert expanded[40].name == "L1_model_e_zeroshot"
    assert expanded[49].name == "L5_model_e_multishot"


def test_expand_conditions_matches_real_configured_models(project_root: Path) -> None:
    app_config = load_app_config(project_root)
    assert list(app_config.models.models) == [
        "gemini",
        "gemini_lite",
        "model_c",
        "model_d",
        "claude_haiku",
        "glm",
    ]
    expanded = expand_conditions(app_config.models.models)
    assert _brief(expanded[:30]) == _FROZEN_LEGACY_MATRIX
    assert [condition.id for condition in expanded[30:]] == list(range(31, 61))
    assert expanded[40].name == "L1_claude_haiku_zeroshot"
    assert expanded[50].name == "L1_glm_zeroshot"
    assert expanded[59].name == "L5_glm_multishot"


def test_run_cli_single_condition(
    project_root: Path,
    patch_llm_client: None,
    isolated_artifacts: Path,
    results_raw_dir: Path,
) -> None:
    result = CliRunner().invoke(
        app,
        ["run", "--condition", "L2_gemini_zeroshot", "--no-start-sut", "--no-measure"],
    )

    assert "L2_gemini_zeroshot" in result.stdout
    assert "Runs: 1" in result.stdout
    assert "Outcomes: " in result.stdout
    assert len(_run_dirs(isolated_artifacts)) == 1
    assert list(results_raw_dir.glob("matrix_*.json"))
    assert result.exit_code == 1
    assert "failed: 1" in result.stdout
    table_row = next(line for line in result.stdout.splitlines() if "not_measured" in line)
    assert "False" in table_row


def test_run_cli_l1_requires_reachable_sut(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_artifacts: Path,
    results_raw_dir: Path,
) -> None:
    monkeypatch.setattr(
        "benchmark_agent.execution.sut.SutManager.is_reachable",
        lambda self, *, timeout_seconds=2.0: False,
    )
    result = CliRunner().invoke(
        app, ["run", "--condition", "L1_gemini_zeroshot", "--no-start-sut", "--no-measure"]
    )

    assert result.exit_code == 1
    assert "L1 probing" in (result.stdout + result.stderr)
    assert "needs a reachable SUT" in (result.stdout + result.stderr)
    assert not _run_dirs(isolated_artifacts)


def test_run_cli_all_expand_selects_full_matrix(project_root: Path) -> None:
    result = CliRunner().invoke(app, ["run", "--expand", "--no-start-sut", "--no-measure"])
    assert result.exit_code == 2
    assert "--expand only makes sense together with --all" in (result.stdout + result.stderr)


def test_matrix_repetition_matches_csv_even_with_older_runs(
    project_root: Path,
    patch_llm_client: None,
    isolated_artifacts: Path,
    results_raw_dir: Path,
) -> None:
    from benchmark_agent.results_io import read_metric_rows

    matrix = MatrixRunner(project_root)
    condition = matrix.app_config.resolve_condition("L2_gemini_zeroshot")

    first = matrix.run([condition], 1, start_sut=False, stop_sut=False, measure=False)
    second = matrix.run([condition], 1, start_sut=False, stop_sut=False, measure=False)

    assert [outcome.repetition for outcome in first + second] == [1, 1]
    assert [outcome.run_directory.name for outcome in first + second if outcome.run_directory] == [
        "run-001",
        "run-002",
    ]
    rows = read_metric_rows(results_raw_dir / "metrics.csv")
    assert {row.repetition for row in rows} == {1}
    assert {row.run_id for row in rows} == {
        "L2_gemini_zeroshot/run-001",
        "L2_gemini_zeroshot/run-002",
    }
    assert second[0].run_directory is not None
    metadata = json.loads((second[0].run_directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["repetition"] == 1


def test_matrix_pre_run_failure_lands_in_isolated_artifacts(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_artifacts: Path,
    results_raw_dir: Path,
) -> None:
    def _factory(model_config, *, gemini_api_key=None, **_ignored):
        raise ValueError("API key missing (simulated)")

    monkeypatch.setattr("benchmark_agent.agent.agent.create_llm_client", _factory)
    matrix = MatrixRunner(project_root)
    condition = matrix.app_config.resolve_condition("L2_gemini_zeroshot")
    real_condition_dir = PROJECT_ROOT / "results" / "artifacts" / "L2_gemini_zeroshot"
    before = _snapshot(real_condition_dir)

    outcomes = matrix.run([condition], 1, start_sut=False, stop_sut=False, measure=False)

    assert outcomes[0].generation_status == STATUS_GENERATION_FAILED
    assert outcomes[0].run_directory is not None
    assert outcomes[0].run_directory.is_relative_to(isolated_artifacts)
    assert _snapshot(real_condition_dir) == before


def _snapshot(directory: Path) -> set[str]:
    return {entry.name for entry in directory.iterdir()} if directory.is_dir() else set()


def test_run_cli_requires_a_selection(project_root: Path) -> None:
    result = CliRunner().invoke(app, ["run", "--no-start-sut", "--no-measure"])
    assert result.exit_code == 2
    assert "Select conditions" in (result.stdout + result.stderr)


def test_conditions_cli_expand(project_root: Path) -> None:
    result = CliRunner().invoke(app, ["conditions", "--expand"])
    assert result.exit_code == 0, result.stdout + result.stderr
    assert "Full matrix (60 conditions)" in result.stdout
    assert "L5_model_d_multishot" in result.stdout

    plain = CliRunner().invoke(app, ["conditions"])
    assert plain.exit_code == 0
    assert "Configured conditions (60" in plain.stdout


def test_matrix_first_repetition_continues_the_numbering_and_seeds(
    project_root: Path,
    patch_llm_client: None,
    isolated_artifacts: Path,
    results_raw_dir: Path,
) -> None:
    matrix = MatrixRunner(project_root)
    condition = matrix.app_config.resolve_condition("L2_gemini_zeroshot")

    outcomes = matrix.run(
        [condition], 2, start_sut=False, stop_sut=False, measure=False, first_repetition=6
    )

    assert [outcome.repetition for outcome in outcomes] == [6, 7]
    base = matrix.app_config.defaults.seed
    for outcome in outcomes:
        assert outcome.run_directory is not None
        metadata = json.loads(
            (outcome.run_directory / "metadata.json").read_text(encoding="utf-8-sig")
        )
        assert metadata["repetition"] == outcome.repetition
        if matrix.app_config.defaults.seed_per_repetition:
            assert metadata["seed"] == base + outcome.repetition - 1
    with pytest.raises(ValueError):
        matrix.run([condition], 1, start_sut=False, measure=False, first_repetition=0)


def test_matrix_interleave_orders_runs_repetition_major(
    project_root: Path,
    patch_llm_client: None,
    isolated_artifacts: Path,
    results_raw_dir: Path,
) -> None:
    matrix = MatrixRunner(project_root)
    conditions = [
        matrix.app_config.resolve_condition("L2_gemini_zeroshot"),
        matrix.app_config.resolve_condition("L3_gemini_multishot"),
    ]

    outcomes = matrix.run(
        conditions, 2, start_sut=False, stop_sut=False, measure=False, interleave=True
    )

    assert [(outcome.condition_name, outcome.repetition) for outcome in outcomes] == [
        ("L2_gemini_zeroshot", 1),
        ("L3_gemini_multishot", 1),
        ("L2_gemini_zeroshot", 2),
        ("L3_gemini_multishot", 2),
    ]
    assert all(outcome.generation_status == "ok" for outcome in outcomes)
