from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from benchmark_agent.llm.base import create_llm_client
from benchmark_agent.llm.claude_cli_client import ClaudeCliLLMClient
from benchmark_agent.models import LLMRequest, ModelConfig


def _config(**params: object) -> ModelConfig:
    return ModelConfig(
        provider="anthropic",
        model_name="claude-haiku-5-5",
        model_version="claude-haiku-5-5",
        client_adapter="claude_cli",
        parameters=dict(params),
    )


def _body(**overrides: object) -> str:
    body: dict[str, object] = {
        "is_error": False,
        "result": '{"feature": "f", "steps_py": "s"}',
        "stop_reason": "end_turn",
        "session_id": "abc",
        "total_cost_usd": 0.01,
        "usage": {
            "input_tokens": 100,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 5,
            "output_tokens": 300,
        },
        "modelUsage": {"claude-haiku-5-5": {}, "claude-haiku-4-5-20251001": {}},
    }
    body.update(overrides)
    return json.dumps(body)


class _Runner:
    def __init__(self, outputs: list[tuple[int, str, str]]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[list[str], str, dict[str, str]]] = []

    def __call__(
        self, args: Sequence[str], stdin: str, cwd: Path, env: dict[str, str], timeout: float
    ) -> tuple[int, str, str]:
        self.calls.append((list(args), stdin, env))
        return self.outputs.pop(0)


def _request() -> LLMRequest:
    return LLMRequest(condition_name="c", model_name="claude_haiku", prompt="PROMPT", seed=7)


def test_complete_maps_result_and_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    runner = _Runner([(0, _body(), "")])
    client = ClaudeCliLLMClient(_config(), runner=runner, workdir=tmp_path)

    response = client.complete(_request())

    assert response.content == '{"feature": "f", "steps_py": "s"}'
    assert response.model_version == "claude-haiku-5-5"
    assert response.finish_reason == "end_turn"
    assert response.usage == {
        "prompt_token_count": 125,
        "candidates_token_count": 300,
        "total_token_count": 425,
        "cached_content_token_count": 20,
        "cli_list_price_usd": 0.01,
    }
    args, stdin, env = runner.calls[0]
    assert stdin == "PROMPT"
    assert "ANTHROPIC_API_KEY" not in env
    assert args[args.index("--model") + 1] == "claude-haiku-5-5"
    assert args[args.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in args


def test_complete_retries_then_raises(tmp_path: Path) -> None:
    runner = _Runner([(1, "", "boom"), (0, _body(is_error=True, result="rate limit"), "")])
    sleeps: list[float] = []
    client = ClaudeCliLLMClient(
        _config(max_attempts=2), runner=runner, sleep=sleeps.append, workdir=tmp_path
    )

    with pytest.raises(RuntimeError, match="rate limit"):
        client.complete(_request())
    assert len(runner.calls) == 2
    assert len(sleeps) == 1


def test_complete_recovers_after_transient_error(tmp_path: Path) -> None:
    runner = _Runner([(1, "", "boom"), (0, _body(), "")])
    client = ClaudeCliLLMClient(_config(), runner=runner, sleep=lambda _: None, workdir=tmp_path)

    assert client.complete(_request()).content


def test_factory_builds_claude_cli_client() -> None:
    assert isinstance(create_llm_client(_config()), ClaudeCliLLMClient)
