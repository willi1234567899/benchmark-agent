from typing import Any

__all__ = ["EvaluationRunner", "GenerationRunner", "MatrixRunner"]


def __getattr__(name: str) -> Any:
    if name == "EvaluationRunner":
        from benchmark_agent.orchestration.evaluation_runner import EvaluationRunner

        return EvaluationRunner
    if name == "GenerationRunner":
        from benchmark_agent.orchestration.runner import GenerationRunner

        return GenerationRunner
    if name == "MatrixRunner":
        from benchmark_agent.orchestration.matrix import MatrixRunner

        return MatrixRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
