from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from benchmark_agent.llm.base import LLMClient
from benchmark_agent.models import LLMRequest, LLMResponse, ModelConfig

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
_DEFAULT_TIMEOUT_SECONDS = 600.0
_DEFAULT_MAX_ATTEMPTS = 5
_DEFAULT_BACKOFF_SECONDS = 15.0
_MAX_RETRY_DELAY_SECONDS = 300.0

Runner = Callable[[Sequence[str], str, Path, dict[str, str], float], tuple[int, str, str]]


def _run_subprocess(
    args: Sequence[str], stdin: str, cwd: Path, env: dict[str, str], timeout: float
) -> tuple[int, str, str]:
    completed = subprocess.run(
        list(args),
        input=stdin,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


class ClaudeCliLLMClient(LLMClient):
    def __init__(
        self,
        model_config: ModelConfig,
        *,
        runner: Runner = _run_subprocess,
        sleep: Callable[[float], None] = time.sleep,
        workdir: Path | None = None,
    ) -> None:
        self._model_config = model_config
        params = model_config.parameters
        self._executable = _resolve_executable(str(params.get("executable", "claude")))
        self._system_prompt = str(params.get("system_prompt", _DEFAULT_SYSTEM_PROMPT))
        self._effort = params.get("effort")
        self._timeout = float(params.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
        self._max_attempts = max(1, int(params.get("max_attempts", _DEFAULT_MAX_ATTEMPTS)))
        self._backoff_seconds = float(params.get("retry_backoff_seconds", _DEFAULT_BACKOFF_SECONDS))
        self._runner = runner
        self._sleep = sleep
        self._workdir = workdir or Path(tempfile.mkdtemp(prefix="benchmark-agent-claude-"))
        self.backend = "claude_cli_subscription"

    def _args(self) -> list[str]:
        args = [
            self._executable,
            "-p",
            "--model",
            self._model_config.model_name,
            "--output-format",
            "json",
            "--tools",
            "",
            "--system-prompt",
            self._system_prompt,
            "--setting-sources",
            "",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--exclude-dynamic-system-prompt-sections",
        ]
        if self._effort:
            args += ["--effort", str(self._effort)]
        return args

    def complete(self, request: LLMRequest) -> LLMResponse:
        env = {key: value for key, value in os.environ.items() if key != "ANTHROPIC_API_KEY"}
        model_name = self._model_config.model_name
        last_error = ""
        for attempt in range(1, self._max_attempts + 1):
            try:
                exit_code, stdout, stderr = self._runner(
                    self._args(), request.prompt, self._workdir, env, self._timeout
                )
            except subprocess.TimeoutExpired as error:
                last_error = f"timeout after {error.timeout}s"
            else:
                body = _parse_json(stdout)
                if exit_code == 0 and body is not None and not body.get("is_error"):
                    return self._response(body)
                last_error = _error_text(exit_code, body, stdout, stderr)
            if attempt < self._max_attempts:
                delay = self._backoff_seconds * 2 ** (attempt - 1)
                self._sleep(min(delay, _MAX_RETRY_DELAY_SECONDS))
        raise RuntimeError(
            f"Claude CLI call failed ({model_name}) after {self._max_attempts} attempt(s): "
            f"{last_error}"
        )

    def _response(self, body: dict[str, Any]) -> LLMResponse:
        content = body.get("result")
        if not isinstance(content, str) or not content:
            raise ValueError("Claude CLI returned an empty response.")
        return LLMResponse(
            provider=self._model_config.provider,
            model_name=self._model_config.model_name,
            model_version=_served_model(body, self._model_config),
            content=content,
            finish_reason=_optional_str(body.get("stop_reason")),
            response_id=_optional_str(body.get("session_id")),
            usage=_usage(body),
        )


def _resolve_executable(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        return name
    path = Path(found)
    if path.suffix.lower() == ".cmd":
        native_executable = (
            path.parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        )
        if native_executable.exists():
            return str(native_executable)
    return found


def _parse_json(stdout: str) -> dict[str, Any] | None:
    try:
        body = json.loads(stdout)
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _error_text(code: int, body: dict[str, Any] | None, stdout: str, stderr: str) -> str:
    if body is not None:
        detail = body.get("result") or body.get("subtype") or body.get("api_error_status")
        return f"exit {code}: {detail}"
    return f"exit {code}: {(stderr or stdout).strip()[:500]}"


def _served_model(body: dict[str, Any], model_config: ModelConfig) -> str:
    usage = body.get("modelUsage")
    if isinstance(usage, dict) and model_config.model_name in usage:
        return model_config.model_name
    return model_config.model_version


def _usage(body: dict[str, Any]) -> dict[str, Any] | None:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    result: dict[str, Any] = {
        "prompt_token_count": input_tokens + cache_read + cache_write,
        "candidates_token_count": output_tokens,
        "total_token_count": input_tokens + cache_read + cache_write + output_tokens,
        "cached_content_token_count": cache_read,
    }
    if body.get("total_cost_usd") is not None:
        result["cli_list_price_usd"] = body["total_cost_usd"]
    return result


def _optional_str(value: object) -> str | None:
    return str(value) if value else None
