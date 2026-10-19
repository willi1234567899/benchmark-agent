from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field

from benchmark_agent.config import AppConfig, BudgetSettings
from benchmark_agent.models import ModelConfig, ModelPricing

QUARANTINE_DIR_NAME = "_quarantine"
_USAGE_TOTAL = "usage_total"
_USAGE_FINAL = "usage"


class RunCost(BaseModel):
    run_id: str
    condition_name: str
    model: str | None = None
    model_name: str | None = None
    epoch: int | None = None
    timestamp: str | None = None
    llm_calls: int | None = None
    complete: bool = False
    prompt_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None
    quarantined: bool = False

    @property
    def priced(self) -> bool:
        return self.cost_usd is not None


class CostReport(BaseModel):
    runs: list[RunCost] = Field(default_factory=list)
    limit_usd: float = 0.0
    stop_at_usd: float | None = None
    spent_before_usd: float = 0.0
    tracked_since: str | None = None
    tracked_usd: float = 0.0
    tracked_runs: int = 0
    untracked_usd: float = 0.0
    unpriced_runs: int = 0
    partial_runs: int = 0
    by_epoch: dict[str, float] = Field(default_factory=dict)
    by_model: dict[str, float] = Field(default_factory=dict)

    @property
    def total_usd(self) -> float:
        return self.spent_before_usd + self.tracked_usd

    @property
    def remaining_usd(self) -> float:
        return self.limit_usd - self.total_usd

    @property
    def stop_reason(self) -> str | None:
        if self.stop_at_usd is None:
            return None
        if self.total_usd >= self.stop_at_usd:
            return (
                f"budget guard: estimated spend {self.total_usd:.2f} USD reached the stop "
                f"threshold {self.stop_at_usd:.2f} USD (limit {self.limit_usd:.2f} USD); "
                "raise budget.stop_at_usd in config/default.yaml to continue"
            )
        return None


def cost_of_usage(usage: dict[str, object], pricing: ModelPricing) -> tuple[float, int, int, int]:
    prompt = _int(usage.get("prompt_token_count"))
    cached = min(_int(usage.get("cached_content_token_count")), prompt)
    output = _int(usage.get("candidates_token_count")) + _int(usage.get("thoughts_token_count"))
    billed_input = prompt - cached
    cost = (
        billed_input * pricing.input_usd_per_m
        + output * pricing.output_usd_per_m
        + cached * pricing.cached_usd_per_m
    ) / 1_000_000
    return cost, billed_input, output, cached


def run_cost(
    metadata: dict[str, object],
    *,
    run_id: str,
    models: dict[str, ModelConfig],
    quarantined: bool = False,
) -> RunCost | None:
    condition = metadata.get("condition")
    condition_name = (
        str(condition.get("name", "")) if isinstance(condition, dict) else str(run_id.split("/")[0])
    )
    model_key = str(condition.get("model")) if isinstance(condition, dict) else None
    model_block = metadata.get("model")
    model_name = str(model_block.get("model_name")) if isinstance(model_block, dict) else None
    usage = metadata.get(_USAGE_TOTAL)
    complete = isinstance(usage, dict)
    if not complete:
        usage = metadata.get(_USAGE_FINAL)
    if not isinstance(usage, dict):
        return None
    pricing = _pricing_for(model_key, model_name, models)
    record = RunCost(
        run_id=run_id,
        condition_name=condition_name,
        model=model_key,
        model_name=model_name,
        epoch=_optional_int(metadata.get("epoch")),
        timestamp=_optional_str(metadata.get("timestamp")),
        llm_calls=_optional_int(metadata.get("llm_calls")),
        complete=complete,
        quarantined=quarantined,
    )
    if pricing is None:
        record.prompt_tokens = _int(usage.get("prompt_token_count"))
        record.output_tokens = _int(usage.get("candidates_token_count")) + _int(
            usage.get("thoughts_token_count")
        )
        return record
    cost, billed_input, output, cached = cost_of_usage(usage, pricing)
    record.cost_usd = cost
    record.prompt_tokens = billed_input
    record.output_tokens = output
    record.cached_tokens = cached
    return record


def collect_costs(app_config: AppConfig) -> CostReport:
    settings: BudgetSettings = app_config.defaults.budget
    report = CostReport(
        limit_usd=settings.limit_usd,
        stop_at_usd=settings.stop_at_usd,
        spent_before_usd=settings.spent_before_usd,
        tracked_since=settings.tracked_since,
    )
    models = app_config.models.models
    for metadata_path, quarantined in _metadata_files(app_config.paths.results_artifacts_dir):
        run_dir = metadata_path.parent
        if run_dir.parent.name.startswith("_"):
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        except OSError, ValueError:
            continue
        if not isinstance(metadata, dict):
            continue
        record = run_cost(
            metadata,
            run_id=f"{run_dir.parent.name}/{run_dir.name}",
            models=models,
            quarantined=quarantined,
        )
        if record is None:
            continue
        report.runs.append(record)
        if record.cost_usd is None:
            report.unpriced_runs += 1
            continue
        if not record.complete:
            report.partial_runs += 1
        tracked = settings.tracked_since is None or (
            record.timestamp is not None and record.timestamp >= settings.tracked_since
        )
        if tracked:
            report.tracked_usd += record.cost_usd
            report.tracked_runs += 1
        else:
            report.untracked_usd += record.cost_usd
        epoch_key = str(record.epoch) if record.epoch is not None else "?"
        report.by_epoch[epoch_key] = report.by_epoch.get(epoch_key, 0.0) + record.cost_usd
        model_key = record.model or record.model_name or "?"
        report.by_model[model_key] = report.by_model.get(model_key, 0.0) + record.cost_usd
    report.runs.sort(key=lambda run: (run.timestamp or "", run.run_id))
    return report


def format_cost_report(report: CostReport) -> str:
    lines = [
        f"Priced runs: {len(report.runs) - report.unpriced_runs} "
        f"(unpriced {report.unpriced_runs}, final-turn-only {report.partial_runs})",
        f"Tracked since {report.tracked_since or 'always'}: {report.tracked_runs} runs, "
        f"{report.tracked_usd:.3f} USD",
        f"Before tracking (metadata, informational): {report.untracked_usd:.3f} USD; "
        f"booked as spent_before_usd = {report.spent_before_usd:.2f} USD",
        f"Estimated total: {report.total_usd:.2f} USD of {report.limit_usd:.2f} USD "
        f"(remaining {report.remaining_usd:.2f} USD"
        + (f", hard stop at {report.stop_at_usd:.2f} USD)" if report.stop_at_usd else ")"),
    ]
    if report.by_epoch:
        lines.append(
            "By epoch (USD): "
            + ", ".join(
                f"{epoch}: {cost:.3f}"
                for epoch, cost in sorted(report.by_epoch.items(), key=_epoch_sort)
            )
        )
    if report.by_model:
        by_model = ", ".join(
            f"{model}: {cost:.3f}" for model, cost in sorted(report.by_model.items())
        )
        lines.append(f"By model (USD): {by_model}")
    reason = report.stop_reason
    if reason:
        lines.append(f"STOP: {reason}")
    return "\n".join(lines)


def _metadata_files(artifacts_dir: Path) -> Iterator[tuple[Path, bool]]:
    if not artifacts_dir.is_dir():
        return
    for path in sorted(artifacts_dir.glob("*/run-*/metadata.json")):
        yield path, False
    quarantine = artifacts_dir / QUARANTINE_DIR_NAME
    if quarantine.is_dir():
        for path in sorted(quarantine.glob("*/run-*/metadata.json")):
            yield path, True


def _pricing_for(
    model_key: str | None, model_name: str | None, models: dict[str, ModelConfig]
) -> ModelPricing | None:
    if model_key and model_key in models and models[model_key].pricing is not None:
        return models[model_key].pricing
    if model_name:
        for config in models.values():
            if config.model_name == model_name and config.pricing is not None:
                return config.pricing
    return None


def _epoch_sort(item: tuple[str, float]) -> tuple[int, str]:
    key = item[0]
    return (int(key), "") if key.isdigit() else (10**6, key)


def _int(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) else None


def _optional_str(value: object) -> str | None:
    return str(value) if value else None
