from __future__ import annotations

from datetime import UTC, datetime, timedelta

from benchmark_agent.conditions import STRATEGY_SUFFIX
from benchmark_agent.evaluation.runs import SOURCE_PROXY, SOURCE_SERVER, OpStats, RunRecord
from benchmark_agent.evaluation.seed_matrix import (
    LEVELS,
    SeedRep,
    generate_reps,
    latency_p95_ratio,
    server_ops,
    unit,
)
from benchmark_agent.models import (
    STATUS_EXECUTION_FAILED,
    STATUS_OK,
    STATUS_VALIDATION_FAILED,
    GuidanceLevel,
    PromptingStrategy,
)

SEED_MODELS = ("gemini", "model_b", "model_c")
SEED_BASE_TIME = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
_BASE_LATENCY_MS = 6.0


def seed_run_records() -> list[RunRecord]:
    records: list[RunRecord] = []
    tick = 0
    strategies = tuple(PromptingStrategy)
    for level_index, level in enumerate(LEVELS):
        for model_index, model in enumerate(SEED_MODELS):
            for prompting_index, strategy in enumerate(strategies):
                condition_id = len(records) // 8 + 1
                name = f"{level}_{model}_{STRATEGY_SUFFIX[strategy]}"
                reps = generate_reps(level_index, model_index, prompting_index)
                for repetition, rep in enumerate(reps, start=1):
                    tick += 1
                    stamp = (SEED_BASE_TIME + timedelta(minutes=7 * tick)).isoformat()
                    seed = level_index * 100 + model_index * 20 + prompting_index * 8 + repetition
                    records.append(
                        _record(
                            run_id=f"{name}/run-{repetition:03d}",
                            condition_id=condition_id,
                            name=name,
                            level=GuidanceLevel(level).value,
                            model=model,
                            prompting=strategy.value,
                            repetition=repetition,
                            stamp=stamp,
                            seed=seed,
                            rep=rep,
                        )
                    )
    return records


def _record(
    *,
    run_id: str,
    condition_id: int,
    name: str,
    level: str,
    model: str,
    prompting: str,
    repetition: int,
    stamp: str,
    seed: int,
    rep: SeedRep,
) -> RunRecord:
    syntax_ok = bool(rep.syntax_ok)
    executed_ok = bool(rep.executed_ok)
    n_requests = int(rep.n_requests)
    ops_per_sec = rep.ops_per_sec
    ratio = rep.ratio
    feedback = 1 if level == "L4" and unit(seed + 5) < 0.3 else 0
    measurements: dict[str, dict[str, OpStats]] = {}
    covered: list[str] = []
    if executed_ok and ops_per_sec is not None and ratio is not None:
        p95 = latency_p95_ratio(ratio) * _BASE_LATENCY_MS * 2.0
        write = OpStats(
            n_requests=n_requests,
            error_count=0,
            n_points=n_requests * 100,
            throughput_ops_s=float(ops_per_sec),
            latency_p50_ms=round(_BASE_LATENCY_MS * (0.8 + 0.4 * unit(seed + 7)), 2),
            latency_p95_ms=round(p95, 2),
            latency_p99_ms=round(p95 * 1.4, 2),
            covered=True,
        )
        read = OpStats(
            n_requests=10,
            error_count=0,
            throughput_ops_s=round(40.0 * (0.6 + 0.8 * ratio), 1),
            latency_p50_ms=round(12.0 * (0.9 + 0.3 * unit(seed + 17)), 2),
            latency_p95_ms=round(28.0 * latency_p95_ratio(ratio), 2),
            latency_p99_ms=round(40.0 * latency_p95_ratio(ratio), 2),
            covered=True,
        )
        delete_covered = unit(seed + 23) < 0.85
        if delete_covered:
            delete = OpStats(
                n_requests=1,
                error_count=0,
                throughput_ops_s=round(6.0 * (0.7 + 0.6 * ratio), 1),
                latency_p50_ms=round(30.0 * (0.9 + 0.4 * unit(seed + 29)), 2),
                latency_p95_ms=round(45.0 * latency_p95_ratio(ratio), 2),
                latency_p99_ms=round(60.0 * latency_p95_ratio(ratio), 2),
                covered=True,
            )
        else:
            delete = OpStats(
                n_requests=0,
                error_count=0,
                throughput_ops_s=0.0,
                latency_p50_ms=None,
                latency_p95_ms=None,
                latency_p99_ms=None,
                covered=False,
            )
        measurements[SOURCE_PROXY] = {"write": write, "read": read, "delete": delete}
        measurements[SOURCE_SERVER] = {
            "write": write.model_copy(
                update={
                    "throughput_ops_s": server_ops(float(ops_per_sec), seed),
                    "latency_p50_ms": None,
                    "latency_p95_ms": None,
                    "latency_p99_ms": None,
                }
            ),
            "read": read.model_copy(
                update={"latency_p50_ms": None, "latency_p95_ms": None, "latency_p99_ms": None}
            ),
            "delete": delete.model_copy(
                update={"latency_p50_ms": None, "latency_p95_ms": None, "latency_p99_ms": None}
            ),
        }
        covered = [
            operation
            for operation in ("write", "read", "delete")
            if measurements[SOURCE_PROXY][operation].covered
        ]
    semantics_ok = executed_ok and len(covered) == 3
    if not syntax_ok:
        status = STATUS_VALIDATION_FAILED
    elif not executed_ok:
        status = STATUS_EXECUTION_FAILED
    else:
        status = STATUS_OK
    return RunRecord(
        run_id=run_id,
        condition_id=condition_id,
        condition_name=name,
        guidance_level=level,
        model=model,
        prompting=prompting,
        repetition=repetition,
        seed=seed,
        model_version=f"{model}-seed",
        status=status,
        generation_status=STATUS_VALIDATION_FAILED if not syntax_ok else STATUS_OK,
        syntax_ok=syntax_ok,
        executed_ok=executed_ok,
        semantics_ok=semantics_ok,
        success=syntax_ok and executed_ok and semantics_ok,
        coverage_ratio=(len(covered) / 3) if executed_ok else 0.0,
        covered_operations=covered,
        feedback_iterations=feedback,
        attempts=feedback + 1,
        generated_at=stamp,
        evaluated_at=stamp,
        scenarios_passed=3 if executed_ok else 0,
        scenarios_failed=0 if executed_ok else (1 if syntax_ok else None),
        scenarios_error=0 if syntax_ok else None,
        behave_exit_code=0 if executed_ok else (1 if syntax_ok else None),
        timed_out=False if syntax_ok else None,
        error=None if syntax_ok else "synthetic: validation failed (seed)",
        measurements=measurements,
        run_directory=None,
        is_fixture=False,
        synthetic=True,
    )
