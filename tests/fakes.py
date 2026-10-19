from __future__ import annotations

import json

from benchmark_agent.llm.base import LLMClient
from benchmark_agent.models import LLMRequest, LLMResponse, ModelConfig

_FAKE_PAYLOAD = {
    "feature": (
        "Feature: InfluxDB benchmark smoke workload\n\n"
        "  Scenario Outline: Exercise one benchmark operation\n"
        "    Given a benchmark context exists\n"
        '    When the "<operation>" request is prepared\n'
        '    Then the "<operation>" action is recorded\n\n'
        "    Examples:\n"
        "      | operation |\n"
        "      | write     |\n"
        "      | read      |\n"
        "      | delete    |\n"
    ),
    "steps_py": (
        "from behave import given, then, when\n\n\n"
        '@given("a benchmark context exists")\n'
        "def step_benchmark_context_exists(context):\n"
        "    context.recorded_operations = []\n\n\n"
        "@when('the \"{operation}\" request is prepared')\n"
        "def step_request_is_prepared(context, operation):\n"
        "    context.current_operation = operation\n\n\n"
        "@then('the \"{operation}\" action is recorded')\n"
        "def step_action_is_recorded(context, operation):\n"
        "    assert context.current_operation == operation\n"
        "    context.recorded_operations.append(operation)\n"
    ),
    "summary": "Deterministic fake LLM response for harness tests.",
}

_INVALID_PAYLOAD = {
    "feature": (
        "Feature: Broken draft\n\n  Scenario: broken\n    Given a benchmark context exists\n"
    ),
    "steps_py": "def broken(\n",
    "summary": "Invalid first draft for feedback-loop tests.",
}


class FakeLLMClient(LLMClient):
    def __init__(self, model_config: ModelConfig) -> None:
        self._model_config = model_config

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            provider=self._model_config.provider,
            model_name=self._model_config.model_name,
            model_version=self._model_config.model_version,
            content=json.dumps(_FAKE_PAYLOAD),
        )


class _ScriptedFakeLLMClient(LLMClient):
    def __init__(self, model_config: ModelConfig) -> None:
        self._model_config = model_config
        self.calls = 0
        self.prompts: list[str] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        self.prompts.append(request.prompt)
        return LLMResponse(
            provider=self._model_config.provider,
            model_name=self._model_config.model_name,
            model_version=self._model_config.model_version,
            content=self._content(request.prompt),
        )

    def _content(self, prompt: str) -> str:
        raise NotImplementedError


class ProbingFakeLLMClient(_ScriptedFakeLLMClient):
    def _content(self, prompt: str) -> str:
        if self.calls == 1:
            return json.dumps({"tool": "sut_health", "args": {}})
        return json.dumps(_FAKE_PAYLOAD)


class BudgetHungryFakeLLMClient(_ScriptedFakeLLMClient):
    def _content(self, prompt: str) -> str:
        if "probe budget is exhausted" in prompt:
            return json.dumps(_FAKE_PAYLOAD)
        return json.dumps({"tool": "http_request", "args": {"method": "GET", "path": "/ping"}})


class RecoveringFakeLLMClient(_ScriptedFakeLLMClient):
    def _content(self, prompt: str) -> str:
        payload = _INVALID_PAYLOAD if self.calls == 1 else _FAKE_PAYLOAD
        return json.dumps(payload)


class ProseThenValidFakeLLMClient(_ScriptedFakeLLMClient):
    def _content(self, prompt: str) -> str:
        if self.calls == 1:
            return (
                "Sure! Here are the artifacts you asked for:\n\n"
                "```json\n"
                '{"feature": "Feature: truncated\\n", "steps_py": "from behave import given\n'
                "```\n"
                "Let me know if you need anything else."
            )
        return json.dumps(_FAKE_PAYLOAD)


class AlwaysInvalidFakeLLMClient(_ScriptedFakeLLMClient):
    def _content(self, prompt: str) -> str:
        return json.dumps(_INVALID_PAYLOAD)
