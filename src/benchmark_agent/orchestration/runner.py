from __future__ import annotations

from pathlib import Path

from benchmark_agent.agent.agent import GenerationAgent
from benchmark_agent.config import AppConfig, load_app_config
from benchmark_agent.models import GenerationResult
from benchmark_agent.orchestration.workspace import WorkspaceManager


class GenerationRunner:
    def __init__(self, project_root: Path | None = None) -> None:
        self._app_config: AppConfig = load_app_config(project_root)
        self._agent = GenerationAgent(self._app_config)

    @property
    def workspace_manager(self) -> WorkspaceManager:
        return self._agent.workspace_manager

    def run_condition(self, condition_name: str, *, repetition: int = 1) -> GenerationResult:
        condition = self._app_config.resolve_condition(condition_name)
        return self._agent.generate(condition, repetition=repetition)
