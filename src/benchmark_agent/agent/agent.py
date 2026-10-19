from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from benchmark_agent.agent.execution_feedback import ExecutionFeedback, execute_for_feedback
from benchmark_agent.agent.feedback import derive_feedback
from benchmark_agent.agent.tools import SutProbeToolkit
from benchmark_agent.config import AppConfig
from benchmark_agent.errors import GenerationError
from benchmark_agent.execution.sut import sut_environment
from benchmark_agent.execution.validator import validate_artifacts
from benchmark_agent.guidance import create_guidance_provider
from benchmark_agent.llm import create_llm_client
from benchmark_agent.measurement.proxy import MeasurementProxy
from benchmark_agent.models import (
    STATUS_GENERATION_FAILED,
    STATUS_OK,
    STATUS_VALIDATION_FAILED,
    ConditionConfig,
    GeneratedArtifactPayload,
    GeneratedArtifacts,
    GenerationResult,
    LLMRequest,
    LLMResponse,
    ProbeToolCall,
    PromptBuildResult,
    ValidationIssue,
    ValidationResult,
)
from benchmark_agent.orchestration.workspace import WorkspaceManager
from benchmark_agent.prompting import PromptBuilder

_JSON_CONTRACT = "JSON_CONTRACT"
_FEATURE_FILE = "benchmark.feature"
_STEPS_FILE = "steps.py"
_PROMPT_FILE = "prompt.txt"
_RESPONSE_FILE = "raw_response.json"
_METADATA_FILE = "metadata.json"
_BUDGET_EXHAUSTED_NOTICE = (
    "The probe budget is exhausted. Do not return another tool call; "
    "return the final artifact JSON now."
)


@dataclass
class _RunTrace:
    run_directory: Path
    feedback_log: list[dict[str, object]] = field(default_factory=list)
    probe_log: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    turn_count: int = 0

    def record_turn(self, entry: dict[str, object]) -> None:
        self.turn_count += 1
        with (self.run_directory / "turns.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"turn": self.turn_count, **entry}) + "\n")

    def record_usage(self, response: LLMResponse) -> None:
        previous = self.metadata.get("llm_calls")
        self.metadata["llm_calls"] = (previous if isinstance(previous, int) else 0) + 1
        if not response.usage:
            return
        previous_total = self.metadata.get("usage_total")
        summed: dict[str, int] = dict(previous_total) if isinstance(previous_total, dict) else {}
        for key, value in response.usage.items():
            if isinstance(value, int | float):
                summed[key] = int(summed.get(key, 0)) + int(value)
        self.metadata["usage_total"] = summed

    def flush(self) -> None:
        if self.feedback_log:
            _write_json(self.run_directory / "feedback_rounds.json", self.feedback_log)
        if self.probe_log:
            _write_json(self.run_directory / "probe_rounds.json", self.probe_log)
        _write_json(self.run_directory / _METADATA_FILE, self.metadata)


@dataclass
class _ModelTurn:
    response: LLMResponse
    payload: GeneratedArtifactPayload
    sent_prompt: str


class _ContractViolation(Exception):
    def __init__(self, message: str, *, response: LLMResponse | None, sent_prompt: str) -> None:
        super().__init__(message)
        self.response = response
        self.sent_prompt = sent_prompt


class GenerationAgent:
    def __init__(self, app_config: AppConfig) -> None:
        self._app_config = app_config
        self._workspace_manager = WorkspaceManager(app_config.paths.results_artifacts_dir)
        self._prompt_builder = PromptBuilder(app_config.paths.templates_dir)

    @property
    def workspace_manager(self) -> WorkspaceManager:
        return self._workspace_manager

    def generate(self, condition: ConditionConfig, *, repetition: int = 1) -> GenerationResult:
        provider = create_guidance_provider(condition.guidance_level)
        context = provider.build_context(self._app_config, condition)
        base_prompt = self._prompt_builder.build(
            condition,
            context,
            app_config=self._app_config,
        )
        model_config = self._app_config.resolve_model(condition.model)
        llm_client = create_llm_client(model_config, **self._app_config.llm_client_kwargs)

        run_directory = self._workspace_manager.create_run_directory(condition.name)
        trace = _RunTrace(run_directory=run_directory)
        seed = self._effective_seed(repetition)
        probe_facade = self._start_probe_facade() if context.probing_enabled else None
        project_root = self._app_config.paths.project_root
        trace.metadata = {
            "condition": condition.model_dump(mode="json"),
            "model": model_config.model_dump(mode="json"),
            "llm_client": type(llm_client).__name__,
            "llm_backend": getattr(llm_client, "backend", "api_key"),
            "template_name": base_prompt.template_name,
            "feedback_iterations": 0,
            "execution_feedback_rounds": 0,
            "repetition": repetition,
            "attempt": 0,
            "probe_calls": 0,
            "run_directory": _display_path(run_directory, project_root),
            "feature_path": _display_path(run_directory / _FEATURE_FILE, project_root),
            "steps_path": _display_path(run_directory / _STEPS_FILE, project_root),
            "seed": seed,
            "epoch": self._app_config.defaults.epoch,
            "workload": self._app_config.workload.name,
            "sut_api": self._app_config.sut.api,
            "timestamp": datetime.now(UTC).isoformat(),
            "prompt_sha256": _sha256(base_prompt.full_prompt),
            "status": STATUS_GENERATION_FAILED,
        }
        try:
            return self._generate_in(
                condition=condition,
                context_feedback_documents=context.feedback_documents,
                base_prompt=base_prompt,
                llm_client=llm_client,
                model_config_name=model_config.model_name,
                trace=trace,
                seed=seed,
            )
        except GenerationError as error:
            trace.metadata.setdefault("error", str(error))
            raise
        except Exception as error:
            trace.metadata["error"] = f"{type(error).__name__}: {error}"
            raise GenerationError(
                f"Generation failed for {condition.name}: {error}",
                run_directory=run_directory,
            ) from error
        finally:
            trace.flush()
            if probe_facade is not None:
                probe_facade.stop()

    def _probe_base_url(self) -> str:
        sut = self._app_config.sut
        return sut.influxdb_direct_url if sut.api == "influx" else sut.influxdb_url

    def _start_probe_facade(self) -> MeasurementProxy | None:
        sut = self._app_config.sut
        if sut.api == "influx":
            return None
        facade = MeasurementProxy(
            listen_host=self._app_config.measurement.proxy_listen_host,
            listen_port=self._app_config.measurement.proxy_listen_port,
            upstream_url=sut.influxdb_direct_url,
            upstream_timeout_s=self._app_config.probing.timeout_seconds + 5,
            api=sut.api,
            org=sut.org,
            bucket=sut.bucket,
        )
        facade.start()
        return facade

    def _create_probe_toolkit(self) -> SutProbeToolkit:
        probing = self._app_config.probing
        return SutProbeToolkit(
            base_url=self._probe_base_url(),
            max_calls=probing.max_calls,
            timeout_seconds=probing.timeout_seconds,
            max_response_chars=probing.max_response_chars,
            auth_token=self._app_config.sut.token,
            api=self._app_config.sut.api,
        )

    def _sut_reachable(self) -> bool:
        from benchmark_agent.execution import SutManager

        return SutManager(self._app_config.paths.project_root, self._app_config.sut).is_reachable()

    def _validate_run_artifacts(self, run_directory: Path) -> ValidationResult:
        sut = self._app_config.sut
        return validate_artifacts(
            run_directory / _FEATURE_FILE,
            run_directory / _STEPS_FILE,
            env=sut_environment(sut, influx_url=sut.influxdb_direct_url),
        )

    def _run_execution_feedback(
        self, run_directory: Path, *, probe_facade: MeasurementProxy | None
    ) -> ExecutionFeedback | None:
        try:
            return execute_for_feedback(
                self._app_config,
                run_directory / _FEATURE_FILE,
                run_directory / _STEPS_FILE,
                proxy=probe_facade,
            )
        except (OSError, RuntimeError, TimeoutError) as error:
            (run_directory / "execution_feedback_error.txt").write_text(
                f"{type(error).__name__}: {error}", encoding="utf-8"
            )
            return None

    def _effective_seed(self, repetition: int) -> int:
        base = self._app_config.defaults.seed
        if self._app_config.defaults.seed_per_repetition:
            return base + max(repetition, 1) - 1
        return base

    def _generate_in(
        self,
        *,
        condition: ConditionConfig,
        context_feedback_documents: list[Any],
        base_prompt: PromptBuildResult,
        llm_client: Any,
        model_config_name: str,
        trace: _RunTrace,
        seed: int | None,
    ) -> GenerationResult:
        run_directory = trace.run_directory
        guidance_level = self._app_config.resolve_guidance_level(condition.guidance_level.value)
        feedback_enabled = guidance_level.feedback_enabled
        max_iterations = (
            self._app_config.defaults.feedback.max_iterations if feedback_enabled else 0
        )
        execution_feedback_enabled = feedback_enabled and guidance_level.execution_feedback
        if execution_feedback_enabled and not self._sut_reachable():
            execution_feedback_enabled = False
            trace.metadata["execution_feedback_skipped"] = "SUT not reachable"
        probe_toolkit = self._create_probe_toolkit() if guidance_level.probing_enabled else None

        feedback_history: list[str] = []
        previous_payload: GeneratedArtifactPayload | None = None
        feedback_iterations = 0
        attempt = 0
        while True:
            trace.metadata["attempt"] = attempt
            trace.metadata["feedback_iterations"] = feedback_iterations
            prompt = _prompt_for_attempt(base_prompt, feedback_history, previous_payload)

            payload: GeneratedArtifactPayload | None = None
            response: LLMResponse | None = None
            try:
                turn = _complete_with_optional_probing(
                    llm_client=llm_client,
                    model_config_name=model_config_name,
                    condition_name=condition.name,
                    prompt_text=prompt.full_prompt,
                    probe_toolkit=probe_toolkit,
                    trace=trace,
                    seed=seed,
                )
            except _ContractViolation as violation:
                response = violation.response
                sent_prompt = violation.sent_prompt
                validation = _contract_violation_result(violation)
            else:
                payload, response, sent_prompt = turn.payload, turn.response, turn.sent_prompt
                _write_artifact_files(run_directory, sent_prompt, response.content, payload)
                validation = self._validate_run_artifacts(run_directory)

            trace.metadata["probe_calls"] = len(trace.probe_log)
            trace.metadata["prompt_sha256"] = _sha256(sent_prompt)
            trace.metadata["validation"] = validation.model_dump(mode="json")
            if response is not None:
                _record_response_metadata(trace.metadata, response)

            if validation.syntax_ok and payload is not None and response is not None:
                verdict: ExecutionFeedback | None = None
                if execution_feedback_enabled and attempt < max_iterations:
                    verdict = self._run_execution_feedback(run_directory, probe_facade=None)
                if verdict is not None:
                    log = trace.metadata.get("execution_feedback_log")
                    entries = list(log) if isinstance(log, list) else []
                    entries.append({"attempt": attempt, **verdict.log_entry()})
                    trace.metadata["execution_feedback_log"] = entries
                if verdict is not None and not verdict.ok:
                    previous_rounds = trace.metadata.get("execution_feedback_rounds")
                    trace.metadata["execution_feedback_rounds"] = (
                        previous_rounds if isinstance(previous_rounds, int) else 0
                    ) + 1
                    _archive_attempt(
                        run_directory,
                        attempt,
                        prompt=sent_prompt,
                        response=response,
                        payload=payload,
                        validation=validation,
                    )
                    runtime_feedback = derive_feedback(
                        mode=guidance_level.feedback_mode,
                        validation=validation,
                        feedback_documents=context_feedback_documents,
                        round_index=attempt,
                        execution=verdict.text,
                    )
                    feedback_iterations += 1
                    feedback_history.append(runtime_feedback)
                    previous_payload = payload
                    trace.feedback_log.append(
                        {
                            "round": feedback_iterations,
                            "attempt": attempt,
                            "feedback": runtime_feedback,
                            "issues": [],
                            **verdict.log_entry(),
                        }
                    )
                    attempt += 1
                    continue
                trace.metadata["status"] = STATUS_OK
                trace.metadata.pop("error", None)
                return GenerationResult(
                    condition=condition,
                    prompt=PromptBuildResult(
                        template_name=prompt.template_name,
                        system_prompt=prompt.system_prompt,
                        user_prompt=prompt.user_prompt,
                        full_prompt=sent_prompt,
                    ),
                    response=response,
                    payload=payload,
                    artifacts=_artifact_paths(run_directory),
                    feedback_iterations=feedback_iterations,
                    status=STATUS_OK,
                )

            _archive_attempt(
                run_directory,
                attempt,
                prompt=sent_prompt,
                response=response,
                payload=payload,
                validation=validation,
            )
            details = "; ".join(issue.message for issue in validation.issues)
            if not feedback_enabled or attempt >= max_iterations:
                if payload is None:
                    trace.metadata["status"] = STATUS_GENERATION_FAILED
                    trace.metadata["error"] = f"Model produced no artifacts: {details}"
                    raise GenerationError(
                        f"Model produced no artifacts after {attempt + 1} attempt(s): {details}",
                        run_directory=run_directory,
                    )
                trace.metadata["status"] = STATUS_VALIDATION_FAILED
                trace.metadata["error"] = f"Generated artifacts failed validation: {details}"
                raise GenerationError(
                    f"Generated artifacts failed validation after {attempt + 1} attempt(s): "
                    f"{details}",
                    run_directory=run_directory,
                )

            runtime_feedback = derive_feedback(
                mode=guidance_level.feedback_mode,
                validation=validation,
                feedback_documents=context_feedback_documents,
                round_index=attempt,
            )
            feedback_iterations += 1
            feedback_history.append(runtime_feedback)
            previous_payload = payload
            trace.feedback_log.append(
                {
                    "round": feedback_iterations,
                    "attempt": attempt,
                    "feedback": runtime_feedback,
                    "issues": [issue.model_dump(mode="json") for issue in validation.issues],
                }
            )
            attempt += 1


def _contract_violation_result(violation: _ContractViolation) -> ValidationResult:
    return ValidationResult(
        syntax_ok=False,
        issues=[
            ValidationIssue(
                code=_JSON_CONTRACT,
                message=(
                    f"{violation} Reply with exactly one JSON object holding the "
                    "keys `feature` and `steps_py` (no prose, no markdown fences)."
                ),
            )
        ],
    )


def _prompt_for_attempt(
    base_prompt: PromptBuildResult,
    feedback_history: list[str],
    previous_payload: GeneratedArtifactPayload | None,
) -> PromptBuildResult:
    if not feedback_history:
        return base_prompt

    sections: list[str] = []
    if previous_payload is not None:
        sections.append(
            "Previous attempt (rejected):\n"
            f"```gherkin\n{previous_payload.feature.rstrip()}\n```\n"
            f"```python\n{previous_payload.steps_py.rstrip()}\n```"
        )
    else:
        sections.append(
            "Previous attempt (rejected): the reply could not be parsed as artifact JSON."
        )
    if len(feedback_history) > 1:
        earlier = "\n\n".join(
            f"Round {index}:\n{text}" for index, text in enumerate(feedback_history[:-1], start=1)
        )
        sections.append(f"Developer feedback from earlier rounds (still applies):\n{earlier}")
    sections.append(
        "Developer feedback from the previous attempt:\n"
        f"{feedback_history[-1]}\n\n"
        "Regenerate corrected artifacts as valid JSON only."
    )
    user_prompt = f"{base_prompt.user_prompt}\n\n" + "\n\n".join(sections)
    return PromptBuildResult(
        template_name=base_prompt.template_name,
        system_prompt=base_prompt.system_prompt,
        user_prompt=user_prompt,
        full_prompt=f"{base_prompt.system_prompt}\n\n{user_prompt}",
    )


def _complete_with_optional_probing(
    *,
    llm_client: Any,
    model_config_name: str,
    condition_name: str,
    prompt_text: str,
    probe_toolkit: SutProbeToolkit | None,
    trace: _RunTrace,
    seed: int | None,
) -> _ModelTurn:
    working_prompt = prompt_text
    max_turns = 1 if probe_toolkit is None else probe_toolkit.remaining_calls + 1
    for _ in range(max_turns):
        response: LLMResponse = llm_client.complete(
            LLMRequest(
                condition_name=condition_name,
                model_name=model_config_name,
                prompt=working_prompt,
                seed=seed,
            )
        )
        trace.record_usage(response)
        turn_entry: dict[str, object] = {"prompt": working_prompt, "raw_response": response.content}
        try:
            parsed, json_repair = _parse_model_output_with_repair(response.content)
            if json_repair is not None:
                turn_entry["json_repair"] = json_repair
        except ValueError as error:
            turn_entry["parsed_kind"] = "invalid"
            trace.record_turn(turn_entry)
            raise _ContractViolation(
                f"Model output violated the JSON contract: {error}",
                response=response,
                sent_prompt=working_prompt,
            ) from error

        if isinstance(parsed, GeneratedArtifactPayload):
            turn_entry["parsed_kind"] = "artifacts"
            trace.record_turn(turn_entry)
            return _ModelTurn(response=response, payload=parsed, sent_prompt=working_prompt)

        turn_entry.update({"parsed_kind": "tool_call", "tool": parsed.tool, "args": parsed.args})
        if probe_toolkit is None:
            trace.record_turn(turn_entry)
            raise _ContractViolation(
                "Model returned a probe tool call, but probing is disabled for this level.",
                response=response,
                sent_prompt=working_prompt,
            )
        if probe_toolkit.remaining_calls == 0:
            trace.record_turn(turn_entry)
            raise _ContractViolation(
                "Model returned another tool call after the probe budget was exhausted.",
                response=response,
                sent_prompt=working_prompt,
            )

        tool_result = probe_toolkit.call(parsed.tool, parsed.args)
        turn_entry["result"] = tool_result
        trace.record_turn(turn_entry)
        trace.probe_log.append(
            {
                "tool": parsed.tool,
                "args": parsed.args,
                "result": tool_result,
                "remaining_calls": probe_toolkit.remaining_calls,
            }
        )
        remaining = probe_toolkit.remaining_calls
        if remaining == 0:
            continuation = _BUDGET_EXHAUSTED_NOTICE
        else:
            continuation = (
                f"Continue. {remaining} of {probe_toolkit.max_calls} probe calls remain. "
                "Return either another tool-call JSON or the final artifact JSON."
            )
        working_prompt = (
            f"{working_prompt}\n\nTool result for {parsed.tool}:\n{tool_result}\n\n{continuation}"
        )
    raise _ContractViolation(
        "L1 probing finished without producing benchmark artifacts.",
        response=None,
        sent_prompt=working_prompt,
    )


def _parse_model_output(text: str) -> ProbeToolCall | GeneratedArtifactPayload:
    parsed, _repair = _parse_model_output_with_repair(text)
    return parsed


def _parse_model_output_with_repair(
    text: str,
) -> tuple[ProbeToolCall | GeneratedArtifactPayload, str | None]:
    candidate = _extract_json_content(text)
    repair: str | None = None
    payload: object
    try:
        payload = json.loads(candidate, strict=False)
    except json.JSONDecodeError as error:
        whitespace_fixed = _unescape_whitespace_between_tokens(candidate)
        quotes_fixed = _escape_unescaped_quotes_in_strings(whitespace_fixed)
        escapes_fixed = _fix_invalid_escapes_in_strings(quotes_fixed)
        stages = (
            ("escaped_whitespace", whitespace_fixed),
            ("unescaped_quotes", quotes_fixed),
            ("invalid_escapes", escapes_fixed),
        )
        for stage_name, repaired in stages:
            try:
                payload = json.loads(repaired, strict=False)
            except json.JSONDecodeError:
                continue
            repair = stage_name
            break
        else:
            salvaged = _salvage_leading_string_members(escapes_fixed)
            if salvaged is None:
                raise ValueError(
                    f"Model output was not valid JSON ({error.msg} at char {error.pos})."
                ) from error
            payload = salvaged
            repair = "leading_members"
    return _payload_to_model(payload), repair


def _payload_to_model(payload: object) -> ProbeToolCall | GeneratedArtifactPayload:
    if not isinstance(payload, dict):
        raise ValueError("Model output must be a JSON object.")
    try:
        if "feature" in payload and "steps_py" in payload:
            return GeneratedArtifactPayload.model_validate(payload)
        if "tool" in payload:
            return ProbeToolCall.model_validate(payload)
    except ValidationError as error:
        raise ValueError(f"Model output did not match the artifact schema: {error}") from error
    raise ValueError("Model output must be artifact JSON or a probe tool call.")


_FENCE_RE = re.compile(r"```[A-Za-z0-9_-]*[ \t]*\r?\n(.*?)\r?\n[ \t]*```", re.DOTALL)


def _extract_json_content(text: str) -> str:
    stripped = text.strip().removeprefix("\ufeff").strip()
    if stripped.startswith("{") and _parses_as_json(stripped):
        return stripped
    fenced = _FENCE_RE.search(stripped)
    if fenced:
        return fenced.group(1).strip()
    if stripped.startswith("{"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _parses_as_json(text: str) -> bool:
    try:
        json.loads(text, strict=False)
    except ValueError:
        return False
    return True


_ESCAPED_WHITESPACE: dict[str, str] = {"n": "\n", "r": "\r", "t": "\t"}


def _unescape_whitespace_between_tokens(text: str) -> str:
    out: list[str] = []
    in_string = False
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if in_string:
            out.append(char)
            if char == "\\" and index + 1 < length:
                out.append(text[index + 1])
                index += 2
                continue
            if char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
        elif char == "\\" and index + 1 < length and text[index + 1] in _ESCAPED_WHITESPACE:
            out.append(_ESCAPED_WHITESPACE[text[index + 1]])
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


_JSON_WHITESPACE = " \t\r\n"
_LITERAL_WHITESPACE_ESCAPES = ("\\n", "\\r", "\\t")


def _escape_unescaped_quotes_in_strings(text: str) -> str:
    out: list[str] = []
    in_string = False
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if in_string:
            if char == "\\" and index + 1 < length:
                out.append(char)
                out.append(text[index + 1])
                index += 2
                continue
            if char == '"':
                if _closes_string(text, index + 1):
                    in_string = False
                    out.append(char)
                else:
                    out.append('\\"')
                index += 1
                continue
            out.append(char)
            index += 1
            continue
        if char == '"':
            in_string = True
        out.append(char)
        index += 1
    return "".join(out)


_JSON_ESCAPABLE = '"\\/bfnrtu'


def _fix_invalid_escapes_in_strings(text: str) -> str:
    out: list[str] = []
    in_string = False
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if in_string and char == "\\" and index + 1 < length:
            following = text[index + 1]
            if following in _JSON_ESCAPABLE:
                out.append(char)
                out.append(following)
            elif following == "'":
                out.append("'")
            else:
                out.append("\\\\")
                out.append(following)
            index += 2
            continue
        if char == '"':
            in_string = not in_string
        out.append(char)
        index += 1
    return "".join(out)


def _closes_string(text: str, index: int) -> bool:
    length = len(text)
    while index < length:
        if text[index] in _JSON_WHITESPACE:
            index += 1
        elif text[index : index + 2] in _LITERAL_WHITESPACE_ESCAPES:
            index += 2
        else:
            break
    return index >= length or text[index] in ",}]:"


def _salvage_leading_string_members(text: str) -> dict[str, str] | None:
    decoder = json.JSONDecoder(strict=False)
    index = _skip_json_whitespace(text, 0)
    if index >= len(text) or text[index] != "{":
        return None
    index = _skip_json_whitespace(text, index + 1)
    members: dict[str, str] = {}
    try:
        while index < len(text) and text[index] == '"':
            key, index = decoder.raw_decode(text, index)
            index = _skip_json_whitespace(text, index)
            if text[index] != ":":
                break
            index = _skip_json_whitespace(text, index + 1)
            if text[index] != '"':
                break
            value, index = decoder.raw_decode(text, index)
            index = _skip_json_whitespace(text, index)
            if index >= len(text) or text[index] not in ",}":
                break
            if isinstance(key, str) and isinstance(value, str):
                members[key] = value
            if text[index] == "}":
                break
            index = _skip_json_whitespace(text, index + 1)
    except ValueError, IndexError:
        pass
    if "feature" in members and "steps_py" in members:
        return members
    return None


def _skip_json_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index] in _JSON_WHITESPACE:
        index += 1
    return index


def validate_generated_artifacts(feature_path: Path, steps_path: Path) -> None:
    validation = validate_artifacts(feature_path, steps_path)
    if not validation.syntax_ok:
        details = "; ".join(issue.message for issue in validation.issues)
        raise ValueError(f"Generated artifacts failed validation: {details}")


def _write_artifact_files(
    run_directory: Path,
    prompt: str,
    response_content: str,
    payload: GeneratedArtifactPayload,
) -> None:
    (run_directory / _FEATURE_FILE).write_text(payload.feature, encoding="utf-8")
    (run_directory / _STEPS_FILE).write_text(payload.steps_py, encoding="utf-8")
    (run_directory / _PROMPT_FILE).write_text(prompt, encoding="utf-8")
    (run_directory / _RESPONSE_FILE).write_text(response_content, encoding="utf-8")


def _artifact_paths(run_directory: Path) -> GeneratedArtifacts:
    return GeneratedArtifacts(
        run_directory=run_directory,
        feature_path=run_directory / _FEATURE_FILE,
        steps_path=run_directory / _STEPS_FILE,
        prompt_path=run_directory / _PROMPT_FILE,
        response_path=run_directory / _RESPONSE_FILE,
        metadata_path=run_directory / _METADATA_FILE,
    )


def _archive_attempt(
    run_directory: Path,
    attempt: int,
    *,
    prompt: str,
    response: LLMResponse | None,
    payload: GeneratedArtifactPayload | None,
    validation: ValidationResult,
) -> None:
    attempt_dir = run_directory / "attempts" / f"attempt-{attempt:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / _PROMPT_FILE).write_text(prompt, encoding="utf-8")
    if response is not None:
        (attempt_dir / _RESPONSE_FILE).write_text(response.content, encoding="utf-8")
    if payload is not None:
        (attempt_dir / _FEATURE_FILE).write_text(payload.feature, encoding="utf-8")
        (attempt_dir / _STEPS_FILE).write_text(payload.steps_py, encoding="utf-8")
    (attempt_dir / "validation.json").write_text(
        validation.model_dump_json(indent=2), encoding="utf-8"
    )


def _record_response_metadata(metadata: dict[str, object], response: LLMResponse) -> None:
    metadata["response_model_version"] = response.model_version
    for key in ("finish_reason", "response_id", "usage"):
        value = getattr(response, key)
        if value is not None:
            metadata[key] = value


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
