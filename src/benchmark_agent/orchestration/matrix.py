from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from benchmark_agent.conditions import expand_conditions
from benchmark_agent.config import AppConfig, load_app_config
from benchmark_agent.errors import GenerationError
from benchmark_agent.execution import SutManager
from benchmark_agent.models import (
    STATUS_GENERATION_FAILED,
    STATUS_INFRASTRUCTURE_FAILED,
    ConditionConfig,
)
from benchmark_agent.orchestration.evaluation_runner import EvaluationRunner
from benchmark_agent.orchestration.runner import GenerationRunner
from benchmark_agent.orchestration.stability import BUDGET_STOP_STATUS
from benchmark_agent.results_io import utc_timestamp

__all__ = [
    "BUDGET_STOP_STATUS",
    "MatrixRunner",
    "RunOutcome",
    "SKIPPED_STATUS",
    "expand_conditions",
    "format_outcome_table",
    "read_metadata_status",
]

SKIPPED_STATUS = "skipped"


class RunOutcome(BaseModel):
    condition_name: str
    repetition: int
    run_directory: Path | None = None
    generation_status: str
    evaluation_status: str
    success: bool | None = None
    coverage_ratio: float | None = None
    csv_path: Path | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.success is True


class MatrixRunner:
    def __init__(self, project_root: Path | None = None) -> None:
        self._app_config: AppConfig = load_app_config(project_root)
        self._project_root = self._app_config.paths.project_root
        self._generation = GenerationRunner(self._project_root)
        self._evaluation = EvaluationRunner(self._project_root)
        self._sut = SutManager(self._project_root, self._app_config.sut)

    @property
    def app_config(self) -> AppConfig:
        return self._app_config

    def run(
        self,
        conditions: Sequence[ConditionConfig],
        repetitions: int | None = None,
        *,
        start_sut: bool = True,
        stop_sut: bool = False,
        measure: bool = True,
        first_repetition: int = 1,
        interleave: bool = False,
    ) -> list[RunOutcome]:
        repetition_count = repetitions or self._app_config.defaults.repetitions
        if repetition_count < 1:
            raise ValueError("repetitions must be >= 1")
        if first_repetition < 1:
            raise ValueError("first_repetition must be >= 1")
        repetition_numbers = range(first_repetition, first_repetition + repetition_count)
        plan: list[tuple[ConditionConfig, int]] = []
        if interleave:
            for repetition in repetition_numbers:
                for condition in conditions:
                    plan.append((condition, repetition))
        else:
            for condition in conditions:
                for repetition in repetition_numbers:
                    plan.append((condition, repetition))

        if start_sut:
            self._sut.up()
        else:
            self._ensure_sut_for_probing(conditions)

        outcomes: list[RunOutcome] = []
        stop_reason: str | None = None
        try:
            for condition, repetition in plan:
                if stop_reason is None:
                    stop_reason = self._budget_stop_reason()
                if stop_reason is not None:
                    outcomes.append(
                        RunOutcome(
                            condition_name=condition.name,
                            repetition=repetition,
                            generation_status=SKIPPED_STATUS,
                            evaluation_status=BUDGET_STOP_STATUS,
                            success=None,
                            error=stop_reason,
                        )
                    )
                    continue
                outcomes.append(
                    self._run_one(
                        condition,
                        repetition,
                        start_sut=start_sut,
                        measure=measure,
                    )
                )
        finally:
            if stop_sut:
                self._sut.down()
        return outcomes

    def _budget_stop_reason(self) -> str | None:
        budget = self._app_config.defaults.budget
        if not budget.enabled or budget.stop_at_usd is None:
            return None
        from benchmark_agent.evaluation.cost import collect_costs

        return collect_costs(self._app_config).stop_reason

    def _ensure_sut_for_probing(self, conditions: Sequence[ConditionConfig]) -> None:
        needing_sut: list[str] = []
        for condition in conditions:
            level = self._app_config.resolve_guidance_level(condition.guidance_level.value)
            if level.probing_enabled or (level.feedback_enabled and level.execution_feedback):
                needing_sut.append(condition.name)
        if needing_sut and not self._sut.is_reachable():
            raise RuntimeError(
                f"L1 probing / execution feedback ({', '.join(needing_sut)}) needs a reachable "
                f"SUT at {self._app_config.sut.influxdb_direct_url}; use --start-sut or "
                "`docker compose -f sut/docker-compose.yaml up -d`."
            )

    def write_outcomes(self, outcomes: Sequence[RunOutcome]) -> Path:
        raw_dir = self._app_config.paths.results_raw_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = raw_dir / f"matrix_{stamp}.json"
        payload = {
            "timestamp": utc_timestamp(),
            "outcomes": [outcome.model_dump(mode="json") for outcome in outcomes],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _run_one(
        self,
        condition: ConditionConfig,
        repetition: int,
        *,
        start_sut: bool,
        measure: bool,
    ) -> RunOutcome:
        run_directory: Path | None = None
        error: str | None = None
        try:
            generation = self._generation.run_condition(condition.name, repetition=repetition)
            run_directory = generation.artifacts.run_directory
            generation_status = generation.status
        except GenerationError as failure:
            generation_status = STATUS_GENERATION_FAILED
            error = str(failure)
            run_directory = failure.run_directory
        except (ValueError, RuntimeError, FileNotFoundError, KeyError) as failure:
            generation_status = STATUS_GENERATION_FAILED
            error = str(failure)

        if run_directory is None:
            run_directory = self._write_failed_run(
                condition, repetition, error or "unknown generation error"
            )
        else:
            generation_status = read_metadata_status(run_directory) or generation_status

        try:
            evaluation = self._evaluation.run_artifacts(
                run_directory,
                start_sut=start_sut,
                stop_sut=False,
                enable_measurement=measure,
            )
        except (TimeoutError, RuntimeError, OSError) as failure:
            return RunOutcome(
                condition_name=condition.name,
                repetition=repetition,
                run_directory=run_directory,
                generation_status=generation_status,
                evaluation_status=STATUS_INFRASTRUCTURE_FAILED,
                success=None,
                error=error or str(failure),
            )

        success = evaluation.success.success if evaluation.success is not None else False
        return RunOutcome(
            condition_name=condition.name,
            repetition=repetition,
            run_directory=run_directory,
            generation_status=generation_status,
            evaluation_status=evaluation.status,
            success=success,
            coverage_ratio=(
                evaluation.coverage.coverage_ratio if evaluation.coverage is not None else None
            ),
            csv_path=evaluation.csv_path,
            error=error,
        )

    def _write_failed_run(self, condition: ConditionConfig, repetition: int, error: str) -> Path:
        run_directory = self._generation.workspace_manager.create_run_directory(condition.name)
        model = self._app_config.models.models.get(condition.model)
        metadata = {
            "condition": condition.model_dump(mode="json"),
            "model": model.model_dump(mode="json") if model is not None else {},
            "status": STATUS_GENERATION_FAILED,
            "error": error,
            "feedback_iterations": 0,
            "repetition": repetition,
            "seed": self._app_config.defaults.seed,
            "timestamp": utc_timestamp(),
        }
        (run_directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return run_directory


def read_metadata_status(run_directory: Path) -> str | None:
    path = run_directory / "metadata.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError, ValueError:
        return None
    status = payload.get("status") if isinstance(payload, dict) else None
    return status if isinstance(status, str) else None


def format_outcome_table(outcomes: Sequence[RunOutcome]) -> str:
    header = ("condition", "rep", "generation", "evaluation", "success", "coverage", "run_dir")
    rows: list[tuple[str, ...]] = [header]
    for outcome in outcomes:
        rows.append(
            (
                outcome.condition_name,
                str(outcome.repetition),
                outcome.generation_status,
                outcome.evaluation_status,
                "-" if outcome.success is None else str(outcome.success),
                "-" if outcome.coverage_ratio is None else f"{outcome.coverage_ratio:.2f}",
                (
                    "-"
                    if outcome.run_directory is None
                    else f"{outcome.run_directory.parent.name}/{outcome.run_directory.name}"
                ),
            )
        )
    widths = [max(len(row[col]) for row in rows) for col in range(len(header))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)
