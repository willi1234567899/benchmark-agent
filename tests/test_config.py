from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

import pytest
import yaml
from conftest import PROJECT_ROOT
from pydantic import ValidationError

from benchmark_agent.config import load_app_config
from benchmark_agent.guidance.level2_openapi import Level2OpenApiGuidanceProvider
from benchmark_agent.prompting.builder import PromptBuilder

_DIALECTS: dict[str, dict[str, object]] = {
    "tessera": {
        "env_url": "TESSERA_URL",
        "write_endpoint": "POST /tessera/v1/ingest",
        "reference": [
            "features/tessera_ingest_benchmark.feature",
            "features/tessera_fetch_benchmark.feature",
            "features/tessera_purge_benchmark.feature",
        ],
    },
    "kestrel": {
        "env_url": "KESTREL_ENDPOINT",
        "write_endpoint": "/series/{series}/points",
        "reference": [
            "features/kestrel_write_benchmark.feature",
            "features/kestrel_read_benchmark.feature",
            "features/kestrel_delete_benchmark.feature",
        ],
    },
}


def _dialect(app_config) -> dict[str, object]:
    spec = app_config.resolve_guidance_level("L2").openapi_files[0]
    name = spec.split("-")[0]
    return {"name": name, "spec": spec, **_DIALECTS[name]}


def test_load_app_config_resolves_l2_condition(project_root: Path) -> None:
    app_config = load_app_config(project_root)

    condition = app_config.resolve_condition("L2_gemini_zeroshot")
    model = app_config.resolve_model("gemini")

    assert condition.guidance_level.value == "L2"
    assert condition.model == "gemini"
    assert app_config.defaults.feedback.max_iterations == 5
    assert model.provider == "google"
    assert model.model_name == "gemini-3.5-flash"
    assert model.model_version == "gemini-3.5-flash"
    assert model.client_adapter == "gemini"


def test_guidance_levels_l3_to_l5_are_fully_populated(project_root: Path) -> None:
    app_config = load_app_config(project_root)

    level3 = app_config.resolve_guidance_level("L3")
    level4 = app_config.resolve_guidance_level("L4")
    level5 = app_config.resolve_guidance_level("L5")

    dialect = _dialect(app_config)
    name = dialect["name"]
    assert level3.use_case_files == [f"{name}_write.md", f"{name}_read.md", f"{name}_delete.md"]
    assert level4.feedback_enabled is True
    assert level4.feedback_mode == "hybrid"
    assert level4.feedback_files == [f"{name}_feedback_hints.md"]
    assert level5.reference_enabled is True
    assert level5.reference_files == dialect["reference"]


def test_level2_guidance_contains_only_workload_and_openapi(project_root: Path) -> None:
    app_config = load_app_config(project_root)
    condition = app_config.resolve_condition("L2_gemini_zeroshot")

    context = Level2OpenApiGuidanceProvider().build_context(app_config, condition)

    assert context.workload_name == app_config.workload.name
    assert len(context.openapi_documents) == 1
    assert context.use_case_documents == []
    assert context.example_documents == []
    assert context.reference_documents == []
    assert str(_dialect(app_config)["env_url"]) in context.workload_prompt


def test_level1_guidance_has_minimal_task_and_no_static_openapi(project_root: Path) -> None:
    from benchmark_agent.guidance.level1_probe import Level1ProbeGuidanceProvider

    app_config = load_app_config(project_root)
    condition = app_config.resolve_condition("L1_gemini_zeroshot")

    context = Level1ProbeGuidanceProvider().build_context(app_config, condition)

    assert context.openapi_documents == []
    assert context.use_case_documents == []
    assert context.probe_documents == []
    assert context.probing_enabled is True
    assert "write" in context.workload_prompt.lower()
    assert "/api/v2/write" not in context.workload_prompt
    assert "influxdb-oss" not in context.workload_prompt

    prompt = PromptBuilder(app_config.paths.templates_dir).build(condition, context)
    assert "http_request" in prompt.user_prompt
    assert "sut_health" in prompt.user_prompt
    assert prompt.user_prompt.find("You may probe") < prompt.user_prompt.find(
        "After probing, the final artifact JSON"
    )
    assert "tool-call JSON" in prompt.system_prompt
    assert (
        "Return exactly one JSON object that validates against this Pydantic schema:"
        not in prompt.user_prompt
    )
    assert "API specification (tessera-v1.yaml)" not in prompt.user_prompt
    assert "(none provided for this guidance level)" not in prompt.user_prompt


def test_level3_guidance_includes_use_cases(project_root: Path) -> None:
    from benchmark_agent.guidance.level3_usecases import Level3UseCasesGuidanceProvider

    app_config = load_app_config(project_root)
    condition = app_config.resolve_condition("L3_gemini_zeroshot")

    context = Level3UseCasesGuidanceProvider().build_context(app_config, condition)

    assert len(context.openapi_documents) == 1
    assert len(context.use_case_documents) == 3
    name = _dialect(app_config)["name"]
    assert {doc.title for doc in context.use_case_documents} == {
        f"Use case ({name}_write.md)",
        f"Use case ({name}_read.md)",
        f"Use case ({name}_delete.md)",
    }
    assert context.reference_documents == []


def test_zero_shot_prompt_contains_schema_and_openapi_reference(project_root: Path) -> None:
    app_config = load_app_config(project_root)
    condition = app_config.resolve_condition("L2_gemini_zeroshot")
    context = Level2OpenApiGuidanceProvider().build_context(app_config, condition)

    prompt = PromptBuilder(app_config.paths.templates_dir).build(condition, context)

    assert prompt.template_name == "zero_shot_v1.md"
    assert "steps_py" in prompt.user_prompt
    dialect = _dialect(app_config)
    assert f"API specification ({dialect['spec']})" in prompt.user_prompt
    assert app_config.workload.name in prompt.user_prompt
    assert str(dialect["env_url"]) in prompt.user_prompt
    assert "(none provided for this guidance level)" not in prompt.user_prompt
    assert "Use-case documents:" not in prompt.user_prompt
    assert "You may probe" not in prompt.user_prompt
    assert "probe protocol" not in prompt.user_prompt


def test_multi_shot_prompt_includes_contamination_safe_examples(project_root: Path) -> None:
    from benchmark_agent.guidance.level3_usecases import Level3UseCasesGuidanceProvider

    app_config = load_app_config(project_root)
    condition = app_config.resolve_condition("L3_gemini_multishot")
    context = Level3UseCasesGuidanceProvider().build_context(app_config, condition)

    prompt = PromptBuilder(app_config.paths.templates_dir).build(
        condition,
        context,
        app_config=app_config,
    )

    assert prompt.template_name == "multi_shot_v1.md"
    assert "kv_http_api.feature" in prompt.user_prompt
    assert "different domain" in prompt.user_prompt
    assert str(_dialect(app_config)["write_endpoint"]) in prompt.user_prompt
    assert "key-value" in prompt.user_prompt.lower() or "Key-value" in prompt.user_prompt


def test_level4_keeps_hints_off_the_first_prompt(project_root: Path) -> None:
    from benchmark_agent.guidance.level4_feedback import Level4FeedbackGuidanceProvider

    app_config = load_app_config(project_root)
    condition = app_config.resolve_condition("L4_gemini_zeroshot")
    context = Level4FeedbackGuidanceProvider().build_context(app_config, condition)

    assert len(context.use_case_documents) == 3
    assert len(context.feedback_documents) == 1
    assert context.reference_documents == []
    assert "Hint 1" in context.feedback_documents[0].content

    prompt = PromptBuilder(app_config.paths.templates_dir).build(condition, context)
    assert "Hint 1" not in prompt.user_prompt
    assert "Feedback hints" not in prompt.user_prompt
    assert "Use-case documents:" in prompt.user_prompt


def test_level5_guidance_includes_bddbench_features(project_root: Path) -> None:
    from benchmark_agent.guidance.level5_reference import Level5ReferenceGuidanceProvider

    app_config = load_app_config(project_root)
    condition = app_config.resolve_condition("L5_gemini_zeroshot")
    context = Level5ReferenceGuidanceProvider().build_context(app_config, condition)

    assert len(context.feedback_documents) == 1
    assert len(context.reference_documents) == 3
    dialect = _dialect(app_config)
    reference_files = cast(list[str], dialect["reference"])
    reference_name = reference_files[0].split("/")[-1]
    assert any(reference_name in doc.title for doc in context.reference_documents)

    prompt = PromptBuilder(app_config.paths.templates_dir).build(condition, context)
    assert reference_name in prompt.user_prompt
    assert str(dialect["write_endpoint"]) in prompt.user_prompt
    assert "Hint 1" not in prompt.user_prompt


def test_hybrid_feedback_combines_auto_and_scripted(project_root: Path) -> None:
    from benchmark_agent.agent.feedback import derive_feedback
    from benchmark_agent.guidance.level4_feedback import Level4FeedbackGuidanceProvider
    from benchmark_agent.models import ValidationIssue, ValidationResult

    app_config = load_app_config(project_root)
    condition = app_config.resolve_condition("L4_gemini_zeroshot")
    context = Level4FeedbackGuidanceProvider().build_context(app_config, condition)
    validation = ValidationResult(
        syntax_ok=False,
        issues=[ValidationIssue(code="PY_COMPILE", message="unexpected EOF", path="steps.py")],
    )

    feedback = derive_feedback(
        mode="hybrid",
        validation=validation,
        feedback_documents=context.feedback_documents,
        round_index=0,
    )

    assert "Automatic validator feedback" in feedback
    assert "PY_COMPILE" in feedback
    assert "Curated developer hint" in feedback


def test_load_app_config_is_anchored_at_project_root_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("BENCHMARK_AGENT_SEED=1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BENCHMARK_AGENT_SEED", raising=False)

    app_config = load_app_config(PROJECT_ROOT)

    assert app_config.defaults.seed == 20260715
    assert app_config.paths.project_root == PROJECT_ROOT.resolve()


def _copy_config_tree(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    shutil.copytree(PROJECT_ROOT / "config", root / "config")
    return root


def test_duplicate_condition_name_is_rejected(tmp_path: Path) -> None:
    root = _copy_config_tree(tmp_path)
    conditions_path = root / "config" / "conditions.yaml"
    payload = yaml.safe_load(conditions_path.read_text(encoding="utf-8"))
    payload["conditions"][1]["name"] = payload["conditions"][0]["name"]
    conditions_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="Duplicate condition name 'L1_gemini_zeroshot'"):
        load_app_config(root)


def test_condition_id_must_match_matrix(tmp_path: Path) -> None:
    root = _copy_config_tree(tmp_path)
    conditions_path = root / "config" / "conditions.yaml"
    payload = yaml.safe_load(conditions_path.read_text(encoding="utf-8"))
    payload["conditions"][0]["id"] = 99
    conditions_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="must match its matrix entry"):
        load_app_config(root)


def test_condition_name_must_exist_in_matrix(tmp_path: Path) -> None:
    root = _copy_config_tree(tmp_path)
    conditions_path = root / "config" / "conditions.yaml"
    payload = yaml.safe_load(conditions_path.read_text(encoding="utf-8"))
    payload["conditions"][0]["name"] = "L2_gemini_oneshot"
    conditions_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="not part of the level x model x strategy"):
        load_app_config(root)


def test_configured_ids_agree_with_expanded_matrix(project_root: Path) -> None:
    from benchmark_agent.conditions import matrix_by_name

    app_config = load_app_config(project_root)
    matrix = matrix_by_name(app_config.models.models)
    for condition in app_config.conditions.conditions:
        assert matrix[condition.name] == condition


def test_unknown_model_reference_is_rejected(tmp_path: Path) -> None:
    root = _copy_config_tree(tmp_path)
    conditions_path = root / "config" / "conditions.yaml"
    payload = yaml.safe_load(conditions_path.read_text(encoding="utf-8"))
    payload["conditions"][0]["model"] = "no_such_model"
    conditions_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="unknown model 'no_such_model'"):
        load_app_config(root)


def test_unknown_guidance_key_and_feedback_mode_are_rejected(tmp_path: Path) -> None:
    root = _copy_config_tree(tmp_path)
    guidance_path = root / "config" / "guidance.yaml"
    payload = yaml.safe_load(guidance_path.read_text(encoding="utf-8"))
    payload["levels"]["L4"]["feedback_mode"] = "hybird"
    guidance_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="feedback_mode"):
        load_app_config(root)

    payload["levels"]["L4"]["feedback_mode"] = "hybrid"
    payload["levels"]["L4"]["probing_enabeld"] = True
    guidance_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="probing_enabeld"):
        load_app_config(root)


def test_proxy_port_must_match_influxdb_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCHMARK_AGENT_MEASUREMENT__PROXY_LISTEN_PORT", "9999")
    with pytest.raises(ValidationError, match="must point at the measurement proxy"):
        load_app_config(PROJECT_ROOT)


def test_direct_url_must_differ_from_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCHMARK_AGENT_SUT__INFLUXDB_DIRECT_URL", "http://127.0.0.1:8086")
    with pytest.raises(ValidationError, match="must not equal the measurement proxy"):
        load_app_config(PROJECT_ROOT)


def test_resolve_condition_by_id(project_root: Path) -> None:
    app_config = load_app_config(project_root)
    assert app_config.resolve_condition_by_id(14).name == "L3_gemini_multishot"
    assert app_config.resolve_condition_by_id(7).name == "L2_gemini_zeroshot"
    with pytest.raises(KeyError, match="Unknown condition id: 99"):
        app_config.resolve_condition_by_id(99)
