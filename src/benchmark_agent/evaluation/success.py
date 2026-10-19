from __future__ import annotations

from benchmark_agent.evaluation.coverage import REQUIRED_OPERATIONS, operation_exercised
from benchmark_agent.models import (
    BehaveRunResult,
    ConformanceResult,
    ProxyWindowStats,
    SuccessResult,
    ValidationResult,
)


def behave_executed_ok(behave: BehaveRunResult | None) -> bool:
    return (
        behave is not None
        and not behave.timed_out
        and behave.exit_code == 0
        and (behave.scenarios_failed or 0) == 0
        and (behave.scenarios_error or 0) == 0
        and (behave.scenarios_passed or 0) >= 1
    )


def evaluate_semantics(
    proxy_stats: ProxyWindowStats | None,
    behave: BehaveRunResult | None,
    *,
    required_operations: tuple[str, ...] = REQUIRED_OPERATIONS,
    conformance: ConformanceResult | None = None,
) -> bool:
    if proxy_stats is None or not behave_executed_ok(behave):
        return False
    exercised = all(
        operation in proxy_stats.by_operation
        and operation_exercised(proxy_stats.by_operation[operation])
        for operation in required_operations
    )
    if not exercised:
        return False
    return conformance is None or conformance.ok


def evaluate_success(
    validation: ValidationResult,
    behave: BehaveRunResult | None,
    *,
    semantics_ok: bool,
) -> SuccessResult:
    return SuccessResult(
        executed_ok=behave_executed_ok(behave),
        syntax_ok=validation.syntax_ok,
        semantics_ok=semantics_ok,
    )
