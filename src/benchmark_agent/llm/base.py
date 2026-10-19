from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

from benchmark_agent.models import LLMRequest, LLMResponse, ModelConfig, VertexSettings

_API_KEY_ENV_VARS = (
    "BENCHMARK_AGENT_GEMINI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


def resolve_api_key(configured: str | None, env_vars: Iterable[str]) -> str | None:
    if configured:
        return configured
    for name in env_vars:
        value = _read_env(name)
        if value:
            return value
    return None


def resolve_gemini_api_key(configured: str | None = None) -> str | None:
    return resolve_api_key(configured, _API_KEY_ENV_VARS)


def _read_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


class LLMClient(ABC):
    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError


def create_llm_client(
    model_config: ModelConfig,
    *,
    gemini_api_key: str | None = None,
    vertex: VertexSettings | None = None,
    provider_api_keys: Mapping[str, str | None] | None = None,
) -> LLMClient:
    if model_config.client_adapter == "gemini":
        from benchmark_agent.llm.gemini_client import GeminiLLMClient

        if vertex is not None:
            return GeminiLLMClient(model_config, vertex=vertex)
        api_key = resolve_gemini_api_key(gemini_api_key)
        if not api_key:
            raise ValueError(
                "Gemini API key missing. Set BENCHMARK_AGENT_GEMINI_API_KEY in `.env` "
                "or the environment (GEMINI_API_KEY / GOOGLE_API_KEY also work)."
            )
        return GeminiLLMClient(model_config, api_key=api_key)

    if model_config.client_adapter == "openai_compatible":
        from benchmark_agent.llm.openai_client import OpenAICompatibleLLMClient

        params = model_config.parameters
        base_url = params.get("base_url")
        if not base_url:
            raise ValueError(
                f"Model {model_config.model_name!r} uses client_adapter "
                "'openai_compatible' but has no parameters.base_url configured in "
                "config/models.yaml."
            )
        env_vars: tuple[str, ...] = tuple(params.get("api_key_env") or ())
        configured = (provider_api_keys or {}).get(model_config.provider)
        api_key = resolve_api_key(configured, env_vars)
        if not api_key:
            names = ", ".join(env_vars) if env_vars else "parameters.api_key_env"
            raise ValueError(
                f"API key missing for model {model_config.model_name!r} (provider "
                f"{model_config.provider!r}). Set one of {names} in `.env` or the "
                "environment, or add an api_key_env entry in config/models.yaml."
            )
        return OpenAICompatibleLLMClient(model_config, api_key=api_key, base_url=base_url)

    if model_config.client_adapter == "claude_cli":
        from benchmark_agent.llm.claude_cli_client import ClaudeCliLLMClient

        return ClaudeCliLLMClient(model_config)

    raise ValueError(
        f"Unsupported LLM client adapter: {model_config.client_adapter}. "
        "Configure a supported adapter in config/models.yaml."
    )
