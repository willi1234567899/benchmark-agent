from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmark_agent.agent.agent import (
    _extract_json_content,
    _parse_model_output,
    _parse_model_output_with_repair,
    _unescape_whitespace_between_tokens,
)
from benchmark_agent.config import load_app_config
from benchmark_agent.llm.base import create_llm_client, resolve_api_key, resolve_gemini_api_key
from benchmark_agent.llm.gemini_client import GeminiLLMClient
from benchmark_agent.llm.openai_client import OpenAICompatibleLLMClient
from benchmark_agent.models import (
    GeneratedArtifactPayload,
    LLMRequest,
    ModelConfig,
    VertexSettings,
)


def test_create_llm_client_requires_gemini_api_key(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BENCHMARK_AGENT_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    app_config = load_app_config(project_root)
    model_config = app_config.resolve_model("gemini")

    with pytest.raises(ValueError, match="Gemini API key missing"):
        create_llm_client(model_config, gemini_api_key=None)


def test_resolve_gemini_api_key_uses_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BENCHMARK_AGENT_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "  google-key  ")
    assert resolve_gemini_api_key() == "google-key"


def test_load_app_config_reads_prefixed_env_key(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BENCHMARK_AGENT_GEMINI_API_KEY", "from-env")
    app_config = load_app_config(project_root)
    assert app_config.gemini_api_key == "from-env"


def test_llm_client_kwargs_default_to_api_key_backend(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCHMARK_AGENT_GEMINI_BACKEND", "api_key")
    monkeypatch.setenv("BENCHMARK_AGENT_GEMINI_API_KEY", "from-env")
    monkeypatch.delenv("BENCHMARK_AGENT_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("BENCHMARK_AGENT_DEEPSEEK_API_KEY", raising=False)
    kwargs = load_app_config(project_root).llm_client_kwargs
    assert set(kwargs) == {"gemini_api_key", "provider_api_keys"}
    assert kwargs["gemini_api_key"] == "from-env"
    keys = kwargs["provider_api_keys"]
    assert set(keys) == {"openai", "deepseek", "openrouter"}
    assert keys["openai"] is None and keys["deepseek"] is None


def test_llm_client_kwargs_carries_openai_compatible_provider_keys(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCHMARK_AGENT_OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("BENCHMARK_AGENT_DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("BENCHMARK_AGENT_OPENROUTER_API_KEY", "sk-or-test")
    kwargs = load_app_config(project_root).llm_client_kwargs
    assert kwargs["provider_api_keys"] == {
        "openai": "sk-openai",
        "deepseek": "sk-deepseek",
        "openrouter": "sk-or-test",
    }


def test_llm_client_kwargs_vertex_backend(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCHMARK_AGENT_GEMINI_BACKEND", "vertex")
    monkeypatch.setenv("BENCHMARK_AGENT_VERTEX_PROJECT", "my-project")
    monkeypatch.setenv("BENCHMARK_AGENT_VERTEX_LOCATION", "europe-west1")
    vertex = load_app_config(project_root).llm_client_kwargs["vertex"]
    assert (vertex.project, vertex.location) == ("my-project", "europe-west1")


def test_vertex_backend_requires_project(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCHMARK_AGENT_GEMINI_BACKEND", "vertex")
    monkeypatch.setenv("BENCHMARK_AGENT_VERTEX_PROJECT", "")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(ValueError, match="VERTEX_PROJECT"):
        _ = load_app_config(project_root).llm_client_kwargs


def test_gemini_client_vertex_backend_uses_adc(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        text = '{"feature": "x", "steps_py": "y"}'

    class FakeModels:
        def generate_content(self, **kwargs: object) -> FakeResponse:
            return FakeResponse()

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.models = FakeModels()

    monkeypatch.setattr("benchmark_agent.llm.gemini_client.genai.Client", FakeClient)
    client = create_llm_client(
        _gemini_model(), vertex=VertexSettings(project="my-project", location="global")
    )
    response = client.complete(
        LLMRequest(condition_name="c", model_name="gemini-3.5-flash", prompt="p")
    )
    assert captured == {"vertexai": True, "project": "my-project", "location": "global"}
    assert client.backend == "vertex:my-project@global"
    assert response.content == '{"feature": "x", "steps_py": "y"}'


def test_gemini_client_calls_generate_content(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        text = '{"feature": "x", "steps_py": "y"}'

    class FakeModels:
        def generate_content(self, **kwargs: object) -> FakeResponse:
            captured.update(kwargs)
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            captured["api_key"] = api_key
            self.models = FakeModels()

    monkeypatch.setattr("benchmark_agent.llm.gemini_client.genai.Client", FakeClient)
    model = ModelConfig(
        provider="google",
        model_name="gemini-3.5-flash",
        model_version="gemini-3.5-flash",
        client_adapter="gemini",
        parameters={"temperature": 0},
    )
    client = GeminiLLMClient(model, api_key="secret")
    response = client.complete(
        LLMRequest(condition_name="L2_gemini_zeroshot", model_name=model.model_name, prompt="ping")
    )

    assert captured["api_key"] == "secret"
    assert captured["model"] == "gemini-3.5-flash"
    assert captured["contents"] == "ping"
    assert response.content == '{"feature": "x", "steps_py": "y"}'
    assert response.model_version == "gemini-3.5-flash"
    assert response.usage is None
    assert response.finish_reason is None
    assert response.response_id is None


def test_gemini_client_records_usage_and_served_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FinishReason:
        name = "STOP"

    class Candidate:
        finish_reason = FinishReason()

    class UsageMetadata:
        prompt_token_count = 171_000
        candidates_token_count = 1_200
        thoughts_token_count = 300
        cached_content_token_count = None
        total_token_count = 172_500

    class FakeResponse:
        text = '{"feature": "x", "steps_py": "y"}'
        model_version = "gemini-3.5-flash-001"
        response_id = "resp-123"
        candidates = [Candidate()]
        usage_metadata = UsageMetadata()

    class FakeModels:
        def generate_content(self, **kwargs: object) -> FakeResponse:
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    monkeypatch.setattr("benchmark_agent.llm.gemini_client.genai.Client", FakeClient)
    model = ModelConfig(
        provider="google",
        model_name="gemini-3.5-flash",
        model_version="gemini-3.5-flash",
        client_adapter="gemini",
        parameters={"temperature": 0},
    )
    response = GeminiLLMClient(model, api_key="secret").complete(
        LLMRequest(condition_name="L2_gemini_zeroshot", model_name=model.model_name, prompt="ping")
    )

    assert response.model_version == "gemini-3.5-flash-001"
    assert response.response_id == "resp-123"
    assert response.finish_reason == "STOP"
    assert response.usage == {
        "prompt_token_count": 171_000,
        "candidates_token_count": 1_200,
        "thoughts_token_count": 300,
        "total_token_count": 172_500,
    }


@pytest.mark.gemini_live
def test_gemini_live_ping(project_root: Path) -> None:
    app_config = load_app_config(project_root)
    if not app_config.gemini_api_key:
        pytest.skip("No Gemini API key configured")
    client = create_llm_client(
        app_config.resolve_model("gemini"),
        gemini_api_key=app_config.gemini_api_key,
    )
    response = client.complete(
        LLMRequest(
            condition_name="_llm_check",
            model_name="gemini-3.5-flash",
            prompt="Reply with the single word pong.",
        )
    )
    assert "pong" in response.content.lower()


def test_extract_json_content_strips_markdown_fence() -> None:
    fenced = """```json
{"feature": "x", "steps_py": "y"}
```"""
    assert _extract_json_content(fenced) == '{"feature": "x", "steps_py": "y"}'


def test_extract_json_content_passes_through_plain_json() -> None:
    plain = '{"feature": "x", "steps_py": "y"}'
    assert _extract_json_content(plain) == plain


def test_parse_model_output_repairs_escaped_whitespace_between_tokens() -> None:
    text = (
        '{"feature": "Feature: f\\n  Scenario: s",\\n'
        '  "steps_py": "x = \\"a\\"\\nprint(x)",\\n'
        '  "summary": "s"\\n}'
    )
    payload = _parse_model_output(text)
    assert isinstance(payload, GeneratedArtifactPayload)
    assert payload.feature == "Feature: f\n  Scenario: s"
    assert payload.steps_py == 'x = "a"\nprint(x)'
    assert payload.summary == "s"


def test_parse_model_output_repairs_forgotten_escape_inside_code_string() -> None:
    text = '{"feature": "f", "steps_py": "url = f"{x}/api"\\nprint(url)", "summary": "s"}'
    parsed, repair = _parse_model_output_with_repair(text)
    assert repair == "unescaped_quotes"
    assert isinstance(parsed, GeneratedArtifactPayload)
    assert parsed.steps_py == 'url = f"{x}/api"\nprint(url)'
    assert parsed.summary == "s"


def test_parse_model_output_repairs_python_style_escapes() -> None:
    text = (
        '{"feature": "f", "steps_py": "pred = f\\\'_m=\\"{m}\\"\\\'\\n'
        'import re\\nre.match(\\"\\d+\\", s)", "summary": "s"}'
    )
    parsed, repair = _parse_model_output_with_repair(text)
    assert repair == "invalid_escapes"
    assert isinstance(parsed, GeneratedArtifactPayload)
    assert parsed.steps_py == 'pred = f\'_m="{m}"\'\nimport re\nre.match("\\d+", s)'


def test_parse_model_output_rejects_truncated_output() -> None:
    text = '{"feature": "f", "steps_py": "import os\\nurl = f"{x}'
    with pytest.raises(ValueError, match="not valid JSON"):
        _parse_model_output(text)


def test_parse_model_output_salvages_escaped_trailing_members() -> None:
    text = (
        '{"feature": "Feature: f",\\n'
        '  "steps_py": "assert x == 1, f\\"got {x}\\"",\\n'
        '  \\"summary\\": \\"InfluxDB smoke\\"\\n}'
    )
    parsed, repair = _parse_model_output_with_repair(text)
    assert repair == "leading_members"
    assert isinstance(parsed, GeneratedArtifactPayload)
    assert parsed.feature == "Feature: f"
    assert parsed.steps_py == 'assert x == 1, f"got {x}"'
    assert parsed.summary is None


def test_parse_model_output_reports_no_repair_for_valid_json() -> None:
    parsed, repair = _parse_model_output_with_repair('{"feature": "f", "steps_py": "s"}')
    assert repair is None
    assert isinstance(parsed, GeneratedArtifactPayload)


def test_salvage_requires_intact_required_members() -> None:
    broken = '{"feature": "f", \\"steps_py\\": \\"print(1)\\", \\"summary\\": \\"s\\"}'
    with pytest.raises(ValueError, match="not valid JSON"):
        _parse_model_output(broken)


def test_unescape_whitespace_keeps_string_content() -> None:
    assert _unescape_whitespace_between_tokens('{"a":\\n"x\\ny",\\t"b": "\\\\n"}') == (
        '{"a":\n"x\\ny",\t"b": "\\\\n"}'
    )


def _gemini_model() -> ModelConfig:
    return ModelConfig(
        provider="google",
        model_name="gemini-3.5-flash",
        model_version="gemini-3.5-flash",
        client_adapter="gemini",
        parameters={"temperature": 0, "retry_backoff_seconds": 0.5},
    )


def test_gemini_client_retries_transient_server_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.genai import errors

    calls = {"n": 0}
    sleeps: list[float] = []

    class FakeResponse:
        text = '{"feature": "x", "steps_py": "y"}'

    class FakeModels:
        def generate_content(self, **kwargs: object) -> FakeResponse:
            calls["n"] += 1
            if calls["n"] < 3:
                raise errors.ServerError(503, {"error": {"message": "high demand"}})
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    monkeypatch.setattr("benchmark_agent.llm.gemini_client.genai.Client", FakeClient)
    client = GeminiLLMClient(_gemini_model(), api_key="secret", sleep=sleeps.append)
    response = client.complete(
        LLMRequest(condition_name="L2_gemini_zeroshot", model_name="gemini-3.5-flash", prompt="p")
    )

    assert response.content == '{"feature": "x", "steps_py": "y"}'
    assert calls["n"] == 3
    assert sleeps == [0.5, 1.0]


def test_gemini_client_waits_for_rate_limit_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.genai import errors

    calls = {"n": 0}
    sleeps: list[float] = []

    class FakeResponse:
        text = '{"feature": "x", "steps_py": "y"}'

    class FakeModels:
        def generate_content(self, **kwargs: object) -> FakeResponse:
            calls["n"] += 1
            if calls["n"] == 1:
                raise errors.ClientError(
                    429,
                    {
                        "error": {
                            "message": "You exceeded your current quota. Please retry in 25.38s.",
                            "status": "RESOURCE_EXHAUSTED",
                        }
                    },
                )
            if calls["n"] == 2:
                raise errors.ClientError(429, {"error": {"message": "quota exceeded"}})
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    monkeypatch.setattr("benchmark_agent.llm.gemini_client.genai.Client", FakeClient)
    client = GeminiLLMClient(_gemini_model(), api_key="secret", sleep=sleeps.append)
    client.complete(LLMRequest(condition_name="c", model_name="gemini-3.5-flash", prompt="p"))

    assert calls["n"] == 3
    assert sleeps == [pytest.approx(26.38), 30.0]


def test_gemini_client_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.genai import errors

    calls = {"n": 0}

    class FakeModels:
        def generate_content(self, **kwargs: object) -> object:
            calls["n"] += 1
            raise errors.ServerError(503, {"error": {"message": "high demand"}})

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    monkeypatch.setattr("benchmark_agent.llm.gemini_client.genai.Client", FakeClient)
    model = _gemini_model()
    model.parameters["max_attempts"] = 3
    client = GeminiLLMClient(model, api_key="secret", sleep=lambda _seconds: None)
    with pytest.raises(RuntimeError, match="after 3 attempt"):
        client.complete(LLMRequest(condition_name="c", model_name="gemini-3.5-flash", prompt="p"))
    assert calls["n"] == 3


def test_gemini_client_does_not_retry_client_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from google.genai import errors

    calls = {"n": 0}

    class FakeModels:
        def generate_content(self, **kwargs: object) -> object:
            calls["n"] += 1
            raise errors.ClientError(400, {"error": {"message": "bad request"}})

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.models = FakeModels()

    monkeypatch.setattr("benchmark_agent.llm.gemini_client.genai.Client", FakeClient)
    client = GeminiLLMClient(_gemini_model(), api_key="secret", sleep=lambda _seconds: None)
    with pytest.raises(RuntimeError, match="after 1 attempt"):
        client.complete(LLMRequest(condition_name="c", model_name="gemini-3.5-flash", prompt="p"))
    assert calls["n"] == 1


class _FakeHttpResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = str(self._payload)
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeHttpClient:
    def __init__(self, responses: list[_FakeHttpResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post(self, path: str, *, json: dict[str, Any]) -> _FakeHttpResponse:
        self.calls.append({"path": path, "json": json})
        return self._responses.pop(0)


def _ok_response(
    content: str = '{"feature": "x", "steps_py": "y"}', **extra: object
) -> _FakeHttpResponse:
    payload: dict[str, Any] = {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}]
    }
    payload.update(extra)
    return _FakeHttpResponse(200, payload)


def _openai_model() -> ModelConfig:
    return ModelConfig(
        provider="openai",
        model_name="gpt-5-mini",
        model_version="gpt-5-mini",
        client_adapter="openai_compatible",
        parameters={
            "base_url": "https://api.openai.com/v1",
            "api_key_env": ["BENCHMARK_AGENT_OPENAI_API_KEY", "OPENAI_API_KEY"],
            "temperature": 0.7,
            "retry_backoff_seconds": 0.5,
        },
    )


def test_openai_compatible_client_requires_api_key() -> None:
    with pytest.raises(ValueError, match="API key missing"):
        OpenAICompatibleLLMClient(_openai_model(), api_key="", base_url="https://api.openai.com/v1")


def test_openai_compatible_client_calls_chat_completions() -> None:
    fake = _FakeHttpClient([_ok_response()])
    client = OpenAICompatibleLLMClient(
        _openai_model(), api_key="secret", base_url="https://api.openai.com/v1", http_client=fake
    )
    response = client.complete(
        LLMRequest(
            condition_name="L2_model_c_zeroshot",
            model_name="gpt-5-mini",
            prompt="ping",
            seed=7,
        )
    )

    assert len(fake.calls) == 1
    assert fake.calls[0]["path"] == "/chat/completions"
    body = fake.calls[0]["json"]
    assert body == {
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.7,
        "seed": 7,
    }
    assert response.content == '{"feature": "x", "steps_py": "y"}'
    assert response.finish_reason == "stop"
    assert response.model_version == "gpt-5-mini"
    assert response.usage is None
    assert response.response_id is None


def test_openai_compatible_client_omits_seed_when_not_given() -> None:
    fake = _FakeHttpClient([_ok_response()])
    client = OpenAICompatibleLLMClient(
        _openai_model(), api_key="secret", base_url="https://api.openai.com/v1", http_client=fake
    )
    client.complete(LLMRequest(condition_name="c", model_name="gpt-5-mini", prompt="ping"))

    assert "seed" not in fake.calls[0]["json"]


def test_openai_compatible_client_records_usage_and_served_version() -> None:
    fake = _FakeHttpClient(
        [
            _ok_response(
                id="chatcmpl-123",
                model="gpt-5-mini-2026-08-01",
                usage={
                    "prompt_tokens": 171_000,
                    "completion_tokens": 1_200,
                    "total_tokens": 172_200,
                    "prompt_tokens_details": {"cached_tokens": 4_096},
                },
            )
        ]
    )
    response = OpenAICompatibleLLMClient(
        _openai_model(), api_key="secret", base_url="https://api.openai.com/v1", http_client=fake
    ).complete(LLMRequest(condition_name="c", model_name="gpt-5-mini", prompt="ping"))

    assert response.model_version == "gpt-5-mini-2026-08-01"
    assert response.response_id == "chatcmpl-123"
    assert response.usage == {
        "prompt_token_count": 171_000,
        "candidates_token_count": 1_200,
        "total_token_count": 172_200,
        "cached_content_token_count": 4_096,
    }


def _glm_model() -> ModelConfig:
    return ModelConfig(
        provider="openrouter",
        model_name="z-ai/glm-5.3",
        model_version="z-ai/glm-5.3",
        client_adapter="openai_compatible",
        parameters={
            "base_url": "https://openrouter.ai/api/v1",
            "temperature": 0.7,
            "extra_body": {
                "max_tokens": 32768,
                "reasoning": {"effort": "low"},
                "provider": {"order": ["z-ai"], "allow_fallbacks": False},
            },
        },
    )


def test_openai_compatible_client_sends_extra_body_fields() -> None:
    model = _glm_model()
    fake = _FakeHttpClient([_ok_response(), _ok_response()])
    client = OpenAICompatibleLLMClient(
        model, api_key="secret", base_url="https://openrouter.ai/api/v1", http_client=fake
    )
    request = LLMRequest(condition_name="L2_glm_zeroshot", model_name="z-ai/glm-5.3", prompt="p")
    client.complete(request)

    body = fake.calls[0]["json"]
    assert body["model"] == "z-ai/glm-5.3"
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 32768
    assert body["reasoning"] == {"effort": "low"}
    assert body["provider"] == {"order": ["z-ai"], "allow_fallbacks": False}
    body["provider"]["order"].append("other")
    client.complete(request)
    assert fake.calls[1]["json"]["provider"]["order"] == ["z-ai"]
    assert model.parameters["extra_body"]["provider"]["order"] == ["z-ai"]


def test_openai_compatible_client_records_reasoning_tokens_and_provider_cost() -> None:
    fake = _FakeHttpClient(
        [
            _ok_response(
                usage={
                    "prompt_tokens": 4_000,
                    "completion_tokens": 6_000,
                    "total_tokens": 10_000,
                    "completion_tokens_details": {"reasoning_tokens": 4_500},
                    "cost": 0.0320,
                }
            )
        ]
    )
    response = OpenAICompatibleLLMClient(
        _glm_model(), api_key="secret", base_url="https://openrouter.ai/api/v1", http_client=fake
    ).complete(LLMRequest(condition_name="c", model_name="z-ai/glm-5.3", prompt="p"))

    assert response.usage == {
        "prompt_token_count": 4_000,
        "candidates_token_count": 6_000,
        "total_token_count": 10_000,
        "reasoning_token_count": 4_500,
        "provider_cost_usd": 0.0320,
    }
    assert "thoughts_token_count" not in response.usage


def test_openai_compatible_client_retries_transient_server_errors() -> None:
    sleeps: list[float] = []
    fake = _FakeHttpClient([_FakeHttpResponse(503), _FakeHttpResponse(503), _ok_response()])
    client = OpenAICompatibleLLMClient(
        _openai_model(),
        api_key="secret",
        base_url="https://api.openai.com/v1",
        http_client=fake,
        sleep=sleeps.append,
    )
    response = client.complete(LLMRequest(condition_name="c", model_name="gpt-5-mini", prompt="p"))

    assert response.content == '{"feature": "x", "steps_py": "y"}'
    assert len(fake.calls) == 3
    assert sleeps == [0.5, 1.0]


def test_openai_compatible_client_waits_for_retry_after_header() -> None:
    sleeps: list[float] = []
    fake = _FakeHttpClient(
        [
            _FakeHttpResponse(429, headers={"retry-after": "25"}),
            _FakeHttpResponse(429),
            _ok_response(),
        ]
    )
    client = OpenAICompatibleLLMClient(
        _openai_model(),
        api_key="secret",
        base_url="https://api.openai.com/v1",
        http_client=fake,
        sleep=sleeps.append,
    )
    client.complete(LLMRequest(condition_name="c", model_name="gpt-5-mini", prompt="p"))

    assert len(fake.calls) == 3
    assert sleeps == [26.0, 30.0]


def test_openai_compatible_client_gives_up_after_max_attempts() -> None:
    fake = _FakeHttpClient([_FakeHttpResponse(503) for _ in range(3)])
    model = _openai_model()
    model.parameters["max_attempts"] = 3
    client = OpenAICompatibleLLMClient(
        model,
        api_key="secret",
        base_url="https://api.openai.com/v1",
        http_client=fake,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(RuntimeError, match="after 3 attempt"):
        client.complete(LLMRequest(condition_name="c", model_name="gpt-5-mini", prompt="p"))
    assert len(fake.calls) == 3


def test_openai_compatible_client_does_not_retry_client_errors() -> None:
    fake = _FakeHttpClient([_FakeHttpResponse(400, {"error": {"message": "bad request"}})])
    client = OpenAICompatibleLLMClient(
        _openai_model(),
        api_key="secret",
        base_url="https://api.openai.com/v1",
        http_client=fake,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(RuntimeError, match="after 1 attempt"):
        client.complete(LLMRequest(condition_name="c", model_name="gpt-5-mini", prompt="p"))
    assert len(fake.calls) == 1


def test_resolve_api_key_checks_configured_then_env_vars_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("_TEST_FIRST_VAR", raising=False)
    monkeypatch.setenv("_TEST_SECOND_VAR", "second-value")
    assert resolve_api_key(None, ["_TEST_FIRST_VAR", "_TEST_SECOND_VAR"]) == "second-value"
    assert (
        resolve_api_key("configured-value", ["_TEST_FIRST_VAR", "_TEST_SECOND_VAR"])
        == "configured-value"
    )
    assert resolve_api_key(None, []) is None


def test_create_llm_client_openai_compatible_requires_base_url() -> None:
    model = _openai_model()
    del model.parameters["base_url"]
    with pytest.raises(ValueError, match="base_url"):
        create_llm_client(model)


def test_create_llm_client_openai_compatible_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BENCHMARK_AGENT_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key missing"):
        create_llm_client(_openai_model())


def test_create_llm_client_openai_compatible_prefers_configured_provider_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BENCHMARK_AGENT_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured: dict[str, object] = {}

    class FakeHttpxClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("benchmark_agent.llm.openai_client.httpx.Client", FakeHttpxClient)
    client = create_llm_client(_openai_model(), provider_api_keys={"openai": "configured-key"})

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert captured["headers"] == {"Authorization": "Bearer configured-key"}


def test_create_llm_client_openai_compatible_falls_back_to_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BENCHMARK_AGENT_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    captured: dict[str, object] = {}

    class FakeHttpxClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("benchmark_agent.llm.openai_client.httpx.Client", FakeHttpxClient)
    create_llm_client(_openai_model())

    assert captured["headers"] == {"Authorization": "Bearer env-key"}


def test_create_llm_client_supports_deepseek_model_via_same_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BENCHMARK_AGENT_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    captured: dict[str, object] = {}

    class FakeHttpxClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("benchmark_agent.llm.openai_client.httpx.Client", FakeHttpxClient)
    model = ModelConfig(
        provider="deepseek",
        model_name="deepseek-v3.2",
        model_version="deepseek-v3.2",
        client_adapter="openai_compatible",
        parameters={
            "base_url": "https://api.deepseek.com",
            "api_key_env": ["BENCHMARK_AGENT_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"],
            "temperature": 0.7,
        },
    )
    client = create_llm_client(model)

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert captured["base_url"] == "https://api.deepseek.com"
    assert captured["headers"] == {"Authorization": "Bearer sk-deepseek"}
