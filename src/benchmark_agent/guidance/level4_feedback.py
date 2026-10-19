from __future__ import annotations

from benchmark_agent.config import AppConfig
from benchmark_agent.guidance.base import GuidanceProvider
from benchmark_agent.guidance.common import (
    load_feedback_documents,
    load_openapi_documents,
    load_use_case_documents,
    load_workload_prompt,
)
from benchmark_agent.models import ConditionConfig, GuidanceContext


class Level4FeedbackGuidanceProvider(GuidanceProvider):
    def build_context(self, app_config: AppConfig, condition: ConditionConfig) -> GuidanceContext:
        guidance_level = app_config.resolve_guidance_level(condition.guidance_level.value)
        if not guidance_level.feedback_enabled:
            raise ValueError("L4 requires feedback_enabled=true in guidance config.")

        return GuidanceContext(
            level=condition.guidance_level,
            workload_name=app_config.workload.name,
            workload_prompt=load_workload_prompt(app_config, guidance_level),
            openapi_documents=load_openapi_documents(app_config, guidance_level),
            use_case_documents=load_use_case_documents(app_config, guidance_level),
            feedback_documents=load_feedback_documents(app_config, guidance_level),
            notes=[
                "L4 sieht zuerst nur L3 (Workload, OpenAPI, Use Cases).",
                "Iteratives Feedback kommt erst nach einem fehlgeschlagenen Validierungsversuch.",
                f"Feedback-Modus: {guidance_level.feedback_mode or 'unspecified'}.",
                "Referenz bleibt fuer L4 ausgeschlossen.",
            ],
        )
