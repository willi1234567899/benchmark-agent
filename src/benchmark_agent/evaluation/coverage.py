from __future__ import annotations

from benchmark_agent.models import CoverageResult, OperationWindowStats

REQUIRED_OPERATIONS: tuple[str, ...] = ("write", "read", "delete")


def operation_exercised(stats: OperationWindowStats) -> bool:
    return (stats.n_requests - stats.error_count) > 0


def evaluate_coverage(
    by_operation: dict[str, OperationWindowStats],
    *,
    executed_ok: bool = True,
    required_operations: tuple[str, ...] = REQUIRED_OPERATIONS,
) -> CoverageResult:
    covered = [
        operation
        for operation in required_operations
        if executed_ok and operation_exercised(by_operation.get(operation, OperationWindowStats()))
    ]
    ratio = len(covered) / len(required_operations) if required_operations else 0.0
    return CoverageResult(covered_operations=covered, coverage_ratio=ratio)
