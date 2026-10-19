from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol, cast

import httpx

from benchmark_agent.llm.base import LLMClient
from benchmark_agent.models import LLMRequest, LLMResponse, ModelConfig


class _HttpResponse(Protocol):
    status_code: int
    text: str

    def json(self) -> Any: ...


class _HttpClient(Protocol):
    def post(self, path: str, /, *, json: dict[str, Any]) -> _HttpResponse: ...


_USAGE_KEY_MAP: tuple[tuple[str, str], ...] = (
    ("prompt_tokens", "prompt_token_count"),
    ("completion_tokens", "candidates_token_count"),
    ("total_tokens", "total_token_count"),
)

_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504})
_DEFAULT_MAX_ATTEMPTS = 8
_DEFAULT_BACKOFF_SECONDS = 2.0
_DEFAULT_TIMEOUT_SECONDS = 120.0
_RATE_LIMIT_MIN_DELAY_SECONDS = 30.0
_MAX_RETRY_DELAY_SECONDS = 120.0


class OpenAICompatibleLLMClient(LLMClient):
    def __init__(
        self,
        model_config: ModelConfig,
        *,
        api_key: str,
        base_url: str,
        sleep: Callable[[float], None] = time.sleep,
        http_client: _HttpClient | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("API key missing.")
        self._model_config = model_config
        params = model_config.parameters
        self._max_attempts = max(1, int(params.get("max_attempts", _DEFAULT_MAX_ATTEMPTS)))
        self._backoff_seconds = float(params.get("retry_backoff_seconds", _DEFAULT_BACKOFF_SECONDS))
        self._sleep = sleep
        self._client: _HttpClient = http_client or cast(
            _HttpClient,
            httpx.Client(
                base_url=base_url.rstrip("/"),
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=float(params.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)),
            ),
        )
        self.backend = "api_key"

    def complete(self, request: LLMRequest) -> LLMResponse:
        temperature = float(self._model_config.parameters.get("temperature", 0))
        body = self._post_with_retry(request.prompt, temperature, request.seed)

        choices = body.get("choices") or []
        if not choices:
            raise ValueError("OpenAI-compatible API returned no choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content:
            raise ValueError("OpenAI-compatible API returned an empty response.")

        served_version = body.get("model")
        return LLMResponse(
            provider=self._model_config.provider,
            model_name=self._model_config.model_name,
            model_version=(
                str(served_version) if served_version else self._model_config.model_version
            ),
            content=content,
            finish_reason=_optional_str(choices[0].get("finish_reason")),
            response_id=_optional_str(body.get("id")),
            usage=_usage(body.get("usage")),
        )

    def _post_with_retry(self, prompt: str, temperature: float, seed: int | None) -> dict[str, Any]:
        model_name = self._model_config.model_name
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if seed is not None:
            payload["seed"] = seed
        extra_body = self._model_config.parameters.get("extra_body")
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                payload[key] = _deep_copy(value)

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.TransportError as error:
                if attempt >= self._max_attempts:
                    raise RuntimeError(
                        f"OpenAI-compatible API call failed ({model_name}) after {attempt} "
                        f"attempt(s): {error}"
                    ) from error
                self._sleep(_retry_delay(None, attempt, self._backoff_seconds))
                continue

            if response.status_code < 400:
                result: dict[str, Any] = response.json()
                return result

            if response.status_code not in _RETRYABLE_STATUS_CODES or attempt >= self._max_attempts:
                raise RuntimeError(
                    f"OpenAI-compatible API call failed ({model_name}) after {attempt} "
                    f"attempt(s): HTTP {response.status_code} {response.text}"
                )
            self._sleep(_retry_delay(response, attempt, self._backoff_seconds))
        raise AssertionError("unreachable: retry loop always returns or raises")


def _retry_delay(response: Any, attempt: int, backoff_seconds: float) -> float:
    delay: float = backoff_seconds * 2 ** (attempt - 1)
    hint = _retry_after_seconds(response) if response is not None else None
    if hint is not None:
        delay = max(delay, hint + 1.0)
    elif response is not None and getattr(response, "status_code", None) == 429:
        delay = max(delay, _RATE_LIMIT_MIN_DELAY_SECONDS)
    return min(delay, _MAX_RETRY_DELAY_SECONDS)


def _retry_after_seconds(response: Any) -> float | None:
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    getter = getattr(headers, "get", None)
    header = getter("retry-after") if callable(getter) else None
    if not header:
        return None
    try:
        return float(header)
    except TypeError, ValueError:
        return None


def _usage(usage: object) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    result: dict[str, Any] = {}
    for source_key, target_key in _USAGE_KEY_MAP:
        value = usage.get(source_key)
        if value is not None:
            result[target_key] = value
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached_tokens = details.get("cached_tokens")
        if cached_tokens is not None:
            result["cached_content_token_count"] = cached_tokens
    completion_details = usage.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        reasoning_tokens = completion_details.get("reasoning_tokens")
        if reasoning_tokens is not None:
            result["reasoning_token_count"] = reasoning_tokens
    provider_cost = usage.get("cost")
    if isinstance(provider_cost, int | float):
        result["provider_cost_usd"] = float(provider_cost)
    return result or None


def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


def _optional_str(value: object) -> str | None:
    return str(value) if value else None
