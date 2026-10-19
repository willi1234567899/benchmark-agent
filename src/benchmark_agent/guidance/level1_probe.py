from __future__ import annotations

from benchmark_agent.config import AppConfig
from benchmark_agent.guidance.base import GuidanceProvider
from benchmark_agent.guidance.common import load_workload_prompt
from benchmark_agent.models import ConditionConfig, GuidanceContext


class Level1ProbeGuidanceProvider(GuidanceProvider):
    def build_context(self, app_config: AppConfig, condition: ConditionConfig) -> GuidanceContext:
        guidance_level = app_config.resolve_guidance_level(condition.guidance_level.value)
        if not guidance_level.probing_enabled:
            raise ValueError("L1 requires probing_enabled=true in guidance config.")

        workload_prompt = ""
        if guidance_level.include_workload:
            workload_prompt = load_workload_prompt(app_config, guidance_level)

        return GuidanceContext(
            level=condition.guidance_level,
            workload_name=app_config.workload.name,
            workload_prompt=workload_prompt,
            probing_enabled=True,
            probe_budget=app_config.probing.max_calls,
            notes=[
                "L1 darf OpenAPI, Use Cases, Feedback und Referenz nicht sehen.",
                "Probing laeuft zur Laufzeit gegen das SUT, nicht gegen die lokale Spec.",
            ],
        )
