from __future__ import annotations

from collections.abc import Iterable

from benchmark_agent.models import ConditionConfig, GuidanceLevel, PromptingStrategy

STRATEGY_SUFFIX = {
    PromptingStrategy.ZERO_SHOT: "zeroshot",
    PromptingStrategy.MULTI_SHOT: "multishot",
}


def expand_conditions(
    models: Iterable[str],
    levels: Iterable[GuidanceLevel] = tuple(GuidanceLevel),
    strategies: Iterable[PromptingStrategy] = tuple(PromptingStrategy),
) -> list[ConditionConfig]:
    model_keys = list(models)
    level_list = list(levels)
    strategy_list = list(strategies)
    legacy_models, extra_models = model_keys[:3], model_keys[3:]

    conditions: list[ConditionConfig] = []
    for level in level_list:
        for model in legacy_models:
            for strategy in strategy_list:
                conditions.append(_condition(len(conditions) + 1, level, model, strategy))

    for model in extra_models:
        for level in level_list:
            for strategy in strategy_list:
                conditions.append(_condition(len(conditions) + 1, level, model, strategy))

    return conditions


def _condition(
    condition_id: int, level: GuidanceLevel, model: str, strategy: PromptingStrategy
) -> ConditionConfig:
    return ConditionConfig(
        id=condition_id,
        name=f"{level.value}_{model}_{STRATEGY_SUFFIX[strategy]}",
        guidance_level=level,
        model=model,
        prompting_strategy=strategy,
    )


def matrix_by_name(models: Iterable[str]) -> dict[str, ConditionConfig]:
    return {condition.name: condition for condition in expand_conditions(models)}
