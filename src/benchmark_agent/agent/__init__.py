from typing import Any

__all__ = ["GenerationAgent", "validate_generated_artifacts"]


def __getattr__(name: str) -> Any:
    if name in {"GenerationAgent", "validate_generated_artifacts"}:
        from benchmark_agent.agent.agent import GenerationAgent, validate_generated_artifacts

        values = {
            "GenerationAgent": GenerationAgent,
            "validate_generated_artifacts": validate_generated_artifacts,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
