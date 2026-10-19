from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes import FakeLLMClient

from benchmark_agent.config import load_app_config
from benchmark_agent.evaluation.cost import (
    CostReport,
    collect_costs,
    cost_of_usage,
    format_cost_report,
    run_cost,
)
from benchmark_agent.models import LLMRequest, LLMResponse, ModelConfig, ModelPricing
from benchmark_agent.orchestration.matrix import BUDGET_STOP_STATUS, SKIPPED_STATUS, MatrixRunner

PRICING = ModelPricing(input_usd_per_m=1.5, output_usd_per_m=9.0, cached_usd_per_m=0.15)
MODELS = {
    "gemini": ModelConfig(
        provider="google",
        model_name="gemini-3.5-flash",
        model_version="gemini-3.5-flash",
        client_adapter="gemini",
        pricing=PRICING,
    ),
    "model_c": ModelConfig(
        provider="placeholder-provider",
        model_name="placeholder-model-c",
        model_version="v",
        client_adapter="placeholder",
    ),
}


class UsageFakeLLMClient(FakeLLMClient):
    usage = {"prompt_token_count": 1000, "candidates_token_count": 100, "thoughts_token_count": 50}

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = super().complete(request)
        return response.model_copy(update={"usage": dict(self.usage)})


def test_cost_of_usage_bills_cached_tokens_separately() -> None:
    usage = {
        "prompt_token_count": 1_000_000,
        "cached_content_token_count": 200_000,
        "candidates_token_count": 100_000,
        "thoughts_token_count": 50_000,
    }
    cost, billed_input, output, cached = cost_of_usage(usage, PRICING)
    assert (billed_input, output, cached) == (800_000, 150_000, 200_000)
    assert cost == pytest.approx(800_000 * 1.5e-6 + 150_000 * 9e-6 + 200_000 * 0.15e-6)


def _metadata(model: str = "gemini", **extra: object) -> dict[str, object]:
    return {
        "condition": {"id": 7, "name": f"L2_{model}_zeroshot", "model": model},
        "model": {"model_name": MODELS[model].model_name},
        "epoch": 6,
        "timestamp": "2026-08-28T12:00:00+00:00",
        **extra,
    }


def test_run_cost_prefers_usage_total_and_flags_partial_runs() -> None:
    final = {"prompt_token_count": 1000, "candidates_token_count": 100}
    total = {"prompt_token_count": 5000, "candidates_token_count": 400, "thoughts_token_count": 100}
    complete = run_cost(
        _metadata(usage=final, usage_total=total, llm_calls=3), run_id="a/run-001", models=MODELS
    )
    assert complete is not None and complete.complete and complete.llm_calls == 3
    assert complete.cost_usd == pytest.approx((5000 * 1.5 + 500 * 9.0) / 1e6)

    partial = run_cost(_metadata(usage=final), run_id="a/run-002", models=MODELS)
    assert partial is not None and not partial.complete
    assert partial.cost_usd == pytest.approx((1000 * 1.5 + 100 * 9.0) / 1e6)

    assert run_cost(_metadata(), run_id="a/run-003", models=MODELS) is None
    unpriced = run_cost(_metadata("model_c", usage=final), run_id="c/run-001", models=MODELS)
    assert unpriced is not None and unpriced.cost_usd is None and unpriced.prompt_tokens == 1000


def _write_run(root: Path, condition: str, number: int, metadata: dict[str, object]) -> None:
    run_dir = root / condition / f"run-{number:03d}"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_collect_costs_tracks_since_and_skips_fixtures(
    project_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("BENCHMARK_AGENT_PATHS__RESULTS_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setenv("BENCHMARK_AGENT_BUDGET__ENABLED", "true")
    monkeypatch.setenv("BENCHMARK_AGENT_BUDGET__LIMIT_USD", "50")
    monkeypatch.setenv("BENCHMARK_AGENT_BUDGET__STOP_AT_USD", "45")
    monkeypatch.setenv("BENCHMARK_AGENT_BUDGET__SPENT_BEFORE_USD", "19.9")
    monkeypatch.setenv("BENCHMARK_AGENT_BUDGET__TRACKED_SINCE", "2026-08-28T09:30:00Z")
    usage = {"prompt_token_count": 1_000_000, "candidates_token_count": 0}
    _write_run(artifacts, "L2_gemini_zeroshot", 1, _metadata(usage_total=usage, llm_calls=1))
    old = _metadata(usage=usage)
    old["timestamp"] = "2026-08-27T12:00:00+00:00"
    old["epoch"] = 2
    _write_run(artifacts, "L2_gemini_zeroshot", 2, old)
    _write_run(artifacts, "_gold_live", 1, _metadata(usage_total=usage))
    quarantined = _metadata(usage_total=usage)
    _write_run(artifacts / "_quarantine", "L3_gemini_zeroshot", 4, quarantined)

    report = collect_costs(load_app_config(project_root))

    assert [run.run_id for run in report.runs] == [
        "L2_gemini_zeroshot/run-002",
        "L2_gemini_zeroshot/run-001",
        "L3_gemini_zeroshot/run-004",
    ]
    assert report.tracked_runs == 2 and report.tracked_usd == pytest.approx(3.0)
    assert report.untracked_usd == pytest.approx(1.5) and report.partial_runs == 1
    assert report.total_usd == pytest.approx(22.9) and report.remaining_usd == pytest.approx(27.1)
    assert report.by_epoch == {"6": pytest.approx(3.0), "2": pytest.approx(1.5)}
    assert next(run for run in report.runs if run.run_id.endswith("run-004")).quarantined
    assert report.stop_reason is None
    assert "22.90 USD of 50.00 USD" in format_cost_report(report)

    tripped = CostReport(limit_usd=50, stop_at_usd=20, spent_before_usd=19.9, tracked_usd=0.2)
    assert tripped.stop_reason is not None and "20.00 USD" in tripped.stop_reason


def test_generation_sums_usage_over_all_turns(
    project_root: Path, monkeypatch: pytest.MonkeyPatch, isolated_artifacts: Path
) -> None:
    monkeypatch.setattr(
        "benchmark_agent.agent.agent.create_llm_client",
        lambda model_config, **_ignored: UsageFakeLLMClient(model_config),
    )
    matrix = MatrixRunner(project_root)
    condition = matrix.app_config.resolve_condition("L2_gemini_zeroshot")
    outcomes = matrix.run([condition], 1, start_sut=False, stop_sut=False, measure=False)
    assert outcomes[0].run_directory is not None
    metadata = json.loads(
        (outcomes[0].run_directory / "metadata.json").read_text(encoding="utf-8-sig")
    )
    assert metadata["llm_calls"] == 1
    assert metadata["usage_total"] == UsageFakeLLMClient.usage
    assert metadata["usage"] == UsageFakeLLMClient.usage


def test_budget_guard_skips_the_rest_of_the_matrix(
    project_root: Path, monkeypatch: pytest.MonkeyPatch, isolated_artifacts: Path
) -> None:
    monkeypatch.setattr(
        "benchmark_agent.agent.agent.create_llm_client",
        lambda model_config, **_ignored: UsageFakeLLMClient(model_config),
    )
    monkeypatch.setenv("BENCHMARK_AGENT_BUDGET__ENABLED", "true")
    monkeypatch.setenv("BENCHMARK_AGENT_BUDGET__LIMIT_USD", "1")
    monkeypatch.setenv("BENCHMARK_AGENT_BUDGET__STOP_AT_USD", "0.002")
    monkeypatch.setenv("BENCHMARK_AGENT_BUDGET__SPENT_BEFORE_USD", "0")
    monkeypatch.delenv("BENCHMARK_AGENT_BUDGET__TRACKED_SINCE", raising=False)
    matrix = MatrixRunner(project_root)
    conditions = [
        matrix.app_config.resolve_condition("L2_gemini_zeroshot"),
        matrix.app_config.resolve_condition("L3_gemini_zeroshot"),
    ]

    outcomes = matrix.run(conditions, 2, start_sut=False, stop_sut=False, measure=False)

    assert [outcome.generation_status for outcome in outcomes] == [
        "ok",
        SKIPPED_STATUS,
        SKIPPED_STATUS,
        SKIPPED_STATUS,
    ]
    assert all(outcome.evaluation_status == BUDGET_STOP_STATUS for outcome in outcomes[1:])
    assert all(
        outcome.success is None and outcome.run_directory is None for outcome in outcomes[1:]
    )
    assert outcomes[1].error is not None and "budget guard" in outcomes[1].error
    assert len(sorted(isolated_artifacts.glob("*/run-*"))) == 1
    report = matrix.write_outcomes(outcomes)
    payload = json.loads(report.read_text(encoding="utf-8"))
    statuses = [outcome["evaluation_status"] for outcome in payload["outcomes"]]
    assert statuses.count(BUDGET_STOP_STATUS) == 3


def test_budget_guard_is_off_by_default_in_tests(
    project_root: Path, monkeypatch: pytest.MonkeyPatch, isolated_artifacts: Path
) -> None:
    monkeypatch.setenv("BENCHMARK_AGENT_BUDGET__ENABLED", "false")
    matrix = MatrixRunner(project_root)
    assert matrix._budget_stop_reason() is None
