from __future__ import annotations

from benchmark_agent.guidance.base import GuidanceProvider
from benchmark_agent.guidance.level1_probe import Level1ProbeGuidanceProvider
from benchmark_agent.guidance.level2_openapi import Level2OpenApiGuidanceProvider
from benchmark_agent.guidance.level3_usecases import Level3UseCasesGuidanceProvider
from benchmark_agent.guidance.level4_feedback import Level4FeedbackGuidanceProvider
from benchmark_agent.guidance.level5_reference import Level5ReferenceGuidanceProvider
from benchmark_agent.models import GuidanceLevel


def create_guidance_provider(level: GuidanceLevel) -> GuidanceProvider:
    if level is GuidanceLevel.L1:
        return Level1ProbeGuidanceProvider()
    if level is GuidanceLevel.L2:
        return Level2OpenApiGuidanceProvider()
    if level is GuidanceLevel.L3:
        return Level3UseCasesGuidanceProvider()
    if level is GuidanceLevel.L4:
        return Level4FeedbackGuidanceProvider()
    if level is GuidanceLevel.L5:
        return Level5ReferenceGuidanceProvider()
    raise ValueError(f"Guidance level not implemented yet: {level.value}")
