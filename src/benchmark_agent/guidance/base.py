from __future__ import annotations

from abc import ABC, abstractmethod

from benchmark_agent.config import AppConfig
from benchmark_agent.models import ConditionConfig, GuidanceContext


class GuidanceProvider(ABC):
    @abstractmethod
    def build_context(self, app_config: AppConfig, condition: ConditionConfig) -> GuidanceContext:
        raise NotImplementedError
