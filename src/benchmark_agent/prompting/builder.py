from __future__ import annotations

from pathlib import Path

from benchmark_agent.config import AppConfig
from benchmark_agent.guidance.common import load_example_documents
from benchmark_agent.models import (
    ConditionConfig,
    GuidanceContext,
    PromptBuildResult,
    PromptingStrategy,
)
from benchmark_agent.prompting.strategies import build_multi_shot_prompt, build_zero_shot_prompt


class PromptBuilder:
    def __init__(self, templates_dir: Path) -> None:
        self._templates_dir = templates_dir

    def build(
        self,
        condition: ConditionConfig,
        context: GuidanceContext,
        *,
        app_config: AppConfig | None = None,
    ) -> PromptBuildResult:
        enriched = context
        if app_config is not None:
            sut_org = app_config.sut.org
            sut_bucket = app_config.sut.bucket
            sut_api = app_config.sut.api
            sut_name = app_config.sut.display_name
        else:
            sut_org = "benchmark"
            sut_bucket = "benchmark"
            sut_api = "influx"
            sut_name = "InfluxDB v2"
        if condition.prompting_strategy is PromptingStrategy.MULTI_SHOT:
            if app_config is None:
                raise ValueError("Multi-shot prompting requires app_config to load examples.")
            enriched = context.model_copy(
                update={"example_documents": load_example_documents(app_config)}
            )
            if not enriched.example_documents:
                raise ValueError("Multi-shot prompting requires examples under inputs/examples/.")
            template_name, user_prompt = build_multi_shot_prompt(
                self._templates_dir,
                condition,
                enriched,
                sut_org=sut_org,
                sut_bucket=sut_bucket,
                sut_api=sut_api,
                sut_name=sut_name,
            )
        elif condition.prompting_strategy is PromptingStrategy.ZERO_SHOT:
            template_name, user_prompt = build_zero_shot_prompt(
                self._templates_dir,
                condition,
                enriched,
                sut_org=sut_org,
                sut_bucket=sut_bucket,
                sut_api=sut_api,
                sut_name=sut_name,
            )
        else:
            raise ValueError(
                f"Unsupported prompting strategy: {condition.prompting_strategy.value}"
            )

        if context.probing_enabled:
            system_prompt = (
                "You are a careful benchmark-generation assistant. "
                "You may first return probe tool-call JSON against the configured SUT, "
                "then the final artifact JSON."
            )
        else:
            system_prompt = (
                "You are a careful benchmark-generation assistant. "
                "Return only valid JSON that matches the requested schema."
            )
        return PromptBuildResult(
            template_name=template_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            full_prompt=f"{system_prompt}\n\n{user_prompt}",
        )
