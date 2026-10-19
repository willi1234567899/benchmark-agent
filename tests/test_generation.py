from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark_agent.errors import GenerationError
from benchmark_agent.llm.base import LLMClient
from benchmark_agent.models import STATUS_GENERATION_FAILED, STATUS_OK, STATUS_VALIDATION_FAILED
from benchmark_agent.orchestration.runner import GenerationRunner


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, client_type: type[Any]) -> list[Any]:
    created: list[Any] = []

    def _factory(
        model_config: Any, *, gemini_api_key: str | None = None, **_ignored: Any
    ) -> LLMClient:
        client: LLMClient = client_type(model_config)
        created.append(client)
        return client

    monkeypatch.setattr("benchmark_agent.agent.agent.create_llm_client", _factory)
    return created


def test_generation_runner_writes_auditable_artifacts(
    project_root: Path, isolated_artifacts: Path, patch_llm_client: None
) -> None:
    runner = GenerationRunner(project_root)

    result = runner.run_condition("L2_gemini_zeroshot")

    assert result.artifacts.run_directory.is_relative_to(isolated_artifacts)
    assert result.artifacts.feature_path.exists()
    assert result.artifacts.steps_path.exists()
    assert result.artifacts.prompt_path.exists()
    assert result.artifacts.response_path.exists()
    assert result.artifacts.metadata_path.exists()
    assert "Scenario Outline" in result.payload.feature
    assert "recorded_operations" in result.payload.steps_py
    assert "final-target benchmark artifacts" in result.prompt.user_prompt
    assert "API specification (" in result.prompt.user_prompt
    assert result.status == STATUS_OK

    metadata = json.loads(result.artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == STATUS_OK
    assert metadata["seed"] == 20260715
    assert len(metadata["prompt_sha256"]) == 64
    assert "timestamp" in metadata
    assert "\\" not in metadata["run_directory"]
    assert "Source: inputs/openapi/" in result.prompt.user_prompt


def test_l1_generation_prompt_has_tools_not_openapi(
    project_root: Path, isolated_artifacts: Path, patch_llm_client: None
) -> None:
    runner = GenerationRunner(project_root)

    result = runner.run_condition("L1_gemini_zeroshot")

    assert "http_request" in result.prompt.user_prompt
    assert "Probe budget: 8 tool calls" in result.prompt.user_prompt
    assert "Authorization: Token" in result.prompt.user_prompt
    assert any(k in result.prompt.user_prompt for k in ("TESSERA_KEY", "KESTREL_TOKEN"))
    assert "API specification (" not in result.prompt.user_prompt
    assert "/api/v2/write" not in result.prompt.user_prompt
    assert not (result.artifacts.run_directory / "probe_rounds.json").exists()


def test_l1_probe_loop_records_tool_round(
    project_root: Path, isolated_artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakes import ProbingFakeLLMClient

    _install_fake_client(monkeypatch, ProbingFakeLLMClient)
    runner = GenerationRunner(project_root)

    result = runner.run_condition("L1_gemini_zeroshot")

    probe_path = result.artifacts.run_directory / "probe_rounds.json"
    assert probe_path.exists()
    payload = json.loads(probe_path.read_text(encoding="utf-8"))
    assert payload[0]["tool"] == "sut_health"
    assert result.artifacts.feature_path.exists()

    prompt_text = result.artifacts.prompt_path.read_text(encoding="utf-8")
    assert "Tool result for sut_health:" in prompt_text
    assert payload[0]["result"] in prompt_text
    assert "7 of 8 probe calls remain" in prompt_text
    assert result.prompt.full_prompt == prompt_text

    turns = (result.artifacts.run_directory / "turns.jsonl").read_text(encoding="utf-8")
    assert len(turns.strip().splitlines()) == 2


def test_l1_budget_exhaustion_gives_final_artifact_turn(
    project_root: Path, isolated_artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakes import BudgetHungryFakeLLMClient

    monkeypatch.setenv("BENCHMARK_AGENT_PROBING__MAX_CALLS", "2")
    clients = _install_fake_client(monkeypatch, BudgetHungryFakeLLMClient)
    runner = GenerationRunner(project_root)

    result = runner.run_condition("L1_gemini_zeroshot")

    client = clients[0]
    assert client.calls == 3
    assert "probe budget is exhausted" in client.prompts[-1]
    assert "Probe budget: 2 tool calls" in result.prompt.user_prompt
    probes = json.loads(
        (result.artifacts.run_directory / "probe_rounds.json").read_text(encoding="utf-8")
    )
    assert len(probes) == 2
    metadata = json.loads(result.artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["probe_calls"] == 2


def test_generation_runner_supports_l3_multishot(
    project_root: Path, isolated_artifacts: Path, patch_llm_client: None
) -> None:
    runner = GenerationRunner(project_root)

    result = runner.run_condition("L3_gemini_multishot")

    assert result.condition.name == "L3_gemini_multishot"
    assert result.prompt.template_name == "multi_shot_v1.md"
    assert "kv_http_api.feature" in result.prompt.user_prompt
    assert "Use case" in result.prompt.user_prompt
    assert result.artifacts.feature_path.exists()


def test_l4_feedback_loop_recovers_from_invalid_draft(
    project_root: Path, isolated_artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakes import RecoveringFakeLLMClient

    clients = _install_fake_client(monkeypatch, RecoveringFakeLLMClient)
    runner = GenerationRunner(project_root)

    result = runner.run_condition("L4_gemini_zeroshot")

    assert result.feedback_iterations == 1
    assert (result.artifacts.run_directory / "feedback_rounds.json").exists()
    assert "Developer feedback from the previous attempt" in result.prompt.full_prompt
    assert result.artifacts.feature_path.exists()

    feedback_prompt = clients[0].prompts[1]
    assert "Previous attempt (rejected):" in feedback_prompt
    assert "Feature: Broken draft" in feedback_prompt
    assert "def broken(" in feedback_prompt
    assert "(steps.py)" in feedback_prompt
    assert str(result.artifacts.run_directory) not in feedback_prompt
    assert result.prompt.full_prompt == feedback_prompt
    assert "Developer feedback from the previous attempt" in result.prompt.user_prompt

    attempt_dir = result.artifacts.run_directory / "attempts" / "attempt-00"
    assert (attempt_dir / "steps.py").read_text(encoding="utf-8") == "def broken(\n"
    assert (attempt_dir / "validation.json").exists()
    assert "Feature: Broken draft" not in result.artifacts.feature_path.read_text(encoding="utf-8")


def test_l4_json_contract_failure_is_fed_back(
    project_root: Path, isolated_artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakes import ProseThenValidFakeLLMClient

    clients = _install_fake_client(monkeypatch, ProseThenValidFakeLLMClient)
    runner = GenerationRunner(project_root)

    result = runner.run_condition("L4_gemini_zeroshot")

    assert result.feedback_iterations == 1
    rounds = json.loads(
        (result.artifacts.run_directory / "feedback_rounds.json").read_text(encoding="utf-8")
    )
    assert rounds[0]["issues"][0]["code"] == "JSON_CONTRACT"
    assert "JSON_CONTRACT" in clients[0].prompts[1]
    assert "could not be parsed as artifact JSON" in clients[0].prompts[1]

    attempt_dir = result.artifacts.run_directory / "attempts" / "attempt-00"
    assert "Sure! Here are" in (attempt_dir / "raw_response.json").read_text(encoding="utf-8")
    assert not (attempt_dir / "steps.py").exists()
    assert result.artifacts.feature_path.exists()


def test_l4_feedback_exhaustion_raises_generation_error(
    project_root: Path, isolated_artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakes import AlwaysInvalidFakeLLMClient

    monkeypatch.setenv("BENCHMARK_AGENT_FEEDBACK__MAX_ITERATIONS", "2")
    clients = _install_fake_client(monkeypatch, AlwaysInvalidFakeLLMClient)
    runner = GenerationRunner(project_root)

    with pytest.raises(GenerationError) as excinfo:
        runner.run_condition("L4_gemini_zeroshot")

    run_directory = excinfo.value.run_directory
    assert run_directory is not None and run_directory.is_relative_to(isolated_artifacts)
    assert clients[0].calls == 3
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == STATUS_VALIDATION_FAILED
    assert "failed validation" in metadata["error"]
    assert metadata["feedback_iterations"] == 2
    assert metadata["validation"]["syntax_ok"] is False
    rounds = json.loads((run_directory / "feedback_rounds.json").read_text(encoding="utf-8"))
    assert [entry["round"] for entry in rounds] == [1, 2]
    assert sorted(path.name for path in (run_directory / "attempts").iterdir()) == [
        "attempt-00",
        "attempt-01",
        "attempt-02",
    ]
    assert "Developer feedback from earlier rounds" in clients[0].prompts[2]


def test_l2_contract_failure_without_feedback_raises_with_run_dir(
    project_root: Path, isolated_artifacts: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakes import ProseThenValidFakeLLMClient

    _install_fake_client(monkeypatch, ProseThenValidFakeLLMClient)
    runner = GenerationRunner(project_root)

    with pytest.raises(GenerationError) as excinfo:
        runner.run_condition("L2_gemini_zeroshot")

    run_directory = excinfo.value.run_directory
    assert run_directory is not None
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == STATUS_GENERATION_FAILED
    assert metadata["validation"]["issues"][0]["code"] == "JSON_CONTRACT"
    assert not (run_directory / "benchmark.feature").exists()
    assert (run_directory / "attempts" / "attempt-00" / "raw_response.json").exists()
    assert not (run_directory / "feedback_rounds.json").exists()


def test_l5_generation_includes_reference_in_prompt(
    project_root: Path, isolated_artifacts: Path, patch_llm_client: None
) -> None:
    runner = GenerationRunner(project_root)

    result = runner.run_condition("L5_gemini_zeroshot")

    assert "Reference (" in result.prompt.user_prompt
    assert "_benchmark.feature" in result.prompt.user_prompt
    assert "Feedback hints" not in result.prompt.user_prompt
    assert result.feedback_iterations == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("\ufeff" + '{"a": 1}', '{"a": 1}'),
        ('Here you go:\n```json\n{"a": 1}\n```\nDone.', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('Result: {"a": {"b": 2}} trailing', '{"a": {"b": 2}}'),
        ('{"a": 1}', '{"a": 1}'),
    ],
)
def test_extract_json_content_handles_common_shapes(text: str, expected: str) -> None:
    from benchmark_agent.agent.agent import _extract_json_content

    assert _extract_json_content(text) == expected


def test_workspace_numbering_skips_foreign_dirs_and_retries(tmp_path: Path) -> None:
    from benchmark_agent.orchestration.workspace import WorkspaceManager

    root = tmp_path / "artifacts"
    (root / "cond" / "run-007").mkdir(parents=True)
    (root / "cond" / "run-abc").mkdir()
    (root / "cond" / "run-0009").mkdir()

    created = WorkspaceManager(root).create_run_directory("cond")

    assert created.name == "run-010"


def test_extract_json_content_prefers_enclosing_object_over_inner_fence() -> None:
    from benchmark_agent.agent.agent import _parse_model_output

    text = (
        '{"feature": "Feature: x\n  Scenario: y\n    Given z\n", '
        '"steps_py": "# usage:\n```python\nx = 1\n```\nfrom behave import given\n"}'
    )
    payload = _parse_model_output(text)
    assert getattr(payload, "steps_py", "").startswith("# usage:")
