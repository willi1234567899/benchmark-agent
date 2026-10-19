from __future__ import annotations

from benchmark_agent.config import AppConfig
from benchmark_agent.guidance.base import GuidanceProvider
from benchmark_agent.guidance.common import load_openapi_documents, load_workload_prompt
from benchmark_agent.models import ConditionConfig, GuidanceContext


class Level2OpenApiGuidanceProvider(GuidanceProvider):
    def build_context(self, app_config: AppConfig, condition: ConditionConfig) -> GuidanceContext:
        guidance_level = app_config.resolve_guidance_level(condition.guidance_level.value)
        return GuidanceContext(
            level=condition.guidance_level,
            workload_name=app_config.workload.name,
            workload_prompt=load_workload_prompt(app_config, guidance_level),
            openapi_documents=load_openapi_documents(app_config, guidance_level),
            notes=[
                "L2 darf nur Workload und OpenAPI sehen.",
                "Use Cases, Feedback und Referenz bleiben fuer L2 ausgeschlossen.",
            ],
        )
