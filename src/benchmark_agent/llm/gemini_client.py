from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from google import genai
from google.genai import errors, types

from benchmark_agent.llm.base import LLMClient
from benchmark_agent.models import LLMRequest, LLMResponse, ModelConfig, VertexSettings

_USAGE_FIELDS: tuple[str, ...] = (
    "prompt_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "cached_content_token_count",
    "total_token_count",
)

_RETRYABLE_CODES: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504})
_DEFAULT_MAX_ATTEMPTS = 8
_DEFAULT_BACKOFF_SECONDS = 2.0


class GeminiLLMClient(LLMClient):
    def __init__(
        self,
        model_config: ModelConfig,
        api_key: str | None = None,
        *,
        vertex: VertexSettings | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._model_config = model_config
        if vertex is not None:
            self._client = genai.Client(
                vertexai=True, project=vertex.project, location=vertex.location
            )
            self.backend = f"vertex:{vertex.project}@{vertex.location}"
        else:
            if not api_key:
                raise ValueError("Gemini API key missing.")
            self._client = genai.Client(api_key=api_key)
            self.backend = "api_key"
        params = model_config.parameters
        self._max_attempts = max(1, int(params.get("max_attempts", _DEFAULT_MAX_ATTEMPTS)))
        self._backoff_seconds = float(params.get("retry_backoff_seconds", _DEFAULT_BACKOFF_SECONDS))
        self._sleep = sleep

    def complete(self, request: LLMRequest) -> LLMResponse:
        temperature = float(self._model_config.parameters.get("temperature", 0))
        response = self._generate_with_retry(request.prompt, temperature, request.seed)

        content = response.text
        if not content:
            raise ValueError("Gemini returned an empty response.")

        served_version = getattr(response, "model_version", None)
        return LLMResponse(
            provider=self._model_config.provider,
            model_name=self._model_config.model_name,
            model_version=(
                str(served_version) if served_version else self._model_config.model_version
            ),
            content=content,
            finish_reason=_finish_reason(response),
            response_id=_optional_str(getattr(response, "response_id", None)),
            usage=_usage(response),
        )

    def _generate_with_retry(self, prompt: str, temperature: float, seed: int | None = None) -> Any:
        model_name = self._model_config.model_name
        config = types.GenerateContentConfig(temperature=temperature, seed=seed)
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
            except Exception as error:
                if not _is_retryable(error) or attempt >= self._max_attempts:
                    raise RuntimeError(
                        f"Gemini API call failed ({model_name}) after {attempt} attempt(s): {error}"
                    ) from error
                self._sleep(_retry_delay(error, attempt, self._backoff_seconds))
        raise AssertionError("unreachable: retry loop always returns or raises")


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, errors.ServerError):
        return True
    code = getattr(error, "code", None)
    return isinstance(code, int) and code in _RETRYABLE_CODES


_RETRY_HINT_RE = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_RATE_LIMIT_MIN_DELAY_SECONDS = 30.0
_MAX_RETRY_DELAY_SECONDS = 120.0


def _retry_delay(error: Exception, attempt: int, backoff_seconds: float) -> float:
    delay: float = backoff_seconds * 2 ** (attempt - 1)
    hint = _RETRY_HINT_RE.search(str(error))
    if hint:
        hinted: float = float(hint.group(1)) + 1.0
        delay = max(delay, hinted)
    elif getattr(error, "code", None) == 429:
        delay = max(delay, _RATE_LIMIT_MIN_DELAY_SECONDS)
    return min(delay, _MAX_RETRY_DELAY_SECONDS)


def _usage(response: object) -> dict[str, Any] | None:
    metadata = getattr(response, "usage_metadata", None)
    if metadata is None:
        return None
    usage: dict[str, Any] = {}
    for field_name in _USAGE_FIELDS:
        value = getattr(metadata, field_name, None)
        if value is not None:
            usage[field_name] = value
    return usage or None


def _finish_reason(response: object) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return None
    return str(getattr(reason, "name", reason))


def _optional_str(value: object) -> str | None:
    return str(value) if value else None
