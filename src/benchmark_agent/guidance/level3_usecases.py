from __future__ import annotations

from benchmark_agent.config import AppConfig
from benchmark_agent.guidance.base import GuidanceProvider
from benchmark_agent.guidance.common import (
    load_openapi_documents,
    load_use_case_documents,
    load_workload_prompt,
)
from benchmark_agent.models import ConditionConfig, GuidanceContext


class Level3UseCasesGuidanceProvider(GuidanceProvider):
    def build_context(self, app_config: AppConfig, condition: ConditionConfig) -> GuidanceContext:
        guidance_level = app_config.resolve_guidance_level(condition.guidance_level.value)
        return GuidanceContext(
            level=condition.guidance_level,
            workload_name=app_config.workload.name,
            workload_prompt=load_workload_prompt(app_config, guidance_level),
            openapi_documents=load_openapi_documents(app_config, guidance_level),
            use_case_documents=load_use_case_documents(app_config, guidance_level),
            notes=[
                "L3 sieht Workload, OpenAPI und Use-Case-Beschreibungen.",
                "Feedback und Referenz bleiben fuer L3 ausgeschlossen.",
            ],
        )
