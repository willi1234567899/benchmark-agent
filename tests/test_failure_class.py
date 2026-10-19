from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmark_agent.evaluation.failure_class import (
    CLASS_API_REJECTED,
    CLASS_NONCONFORMANT,
    CLASS_NONE,
    CLASS_QUOTED_LITERAL,
    CLASS_SCENARIO_FAILED,
    CLASS_STEP_ERROR,
    CLASS_SYNTAX,
    CLASS_TIMEOUT,
    classify_failure,
    failure_class_counts,
)
from benchmark_agent.evaluation.runs import RunRecord, collect_runs
from benchmark_agent.evaluation.statistics import (
    TestResult,
    apply_holm,
    compute_statistics,
    fisher_control_vs_rest,
    pairwise_fisher_vs_reference,
    permutation_success_by_level,
)


def _run(**fields: Any) -> RunRecord:
    base: dict[str, Any] = {
        "run_id": "L2_gemini_zeroshot/run-001",
        "condition_name": "L2_gemini_zeroshot",
        "guidance_level": "L2",
        "model": "gemini",
        "prompting": "zero_shot",
        "status": "execution_failed",
        "syntax_ok": True,
        "executed_ok": False,
        "semantics_ok": False,
        "success": False,
    }
    base.update(fields)
    return RunRecord(**base)


def test_classify_failure_precedence() -> None:
    assert (
        classify_failure(_run(status="ok", executed_ok=True, semantics_ok=True, success=True))[0]
        == CLASS_NONE
    )
    assert classify_failure(_run(status="validation_failed", syntax_ok=False))[0] == CLASS_SYNTAX
    assert classify_failure(_run(timed_out=True))[0] == CLASS_TIMEOUT
    failure_class, detail = classify_failure(_run(api_rejections=["invalid_since x20"]))
    assert (failure_class, detail) == (CLASS_API_REJECTED, "invalid_since x20")
    assert classify_failure(_run(quoted_literal_requests=3, scenarios_failed=1))[0] == (
        CLASS_QUOTED_LITERAL
    )
    assert classify_failure(_run(scenarios_error=2))[0] == CLASS_STEP_ERROR
    assert classify_failure(_run(scenarios_failed=1))[0] == CLASS_SCENARIO_FAILED
    failure_class, detail = classify_failure(
        _run(executed_ok=True, conformance_issues=["read/labels: pinned label filter"])
    )
    assert (failure_class, detail) == (CLASS_NONCONFORMANT, "read/labels")


def test_failure_class_counts_keeps_canonical_order() -> None:
    runs = [
        _run(failure_class=CLASS_SCENARIO_FAILED),
        _run(failure_class=CLASS_SYNTAX),
        _run(failure_class=CLASS_SYNTAX),
        _run(failure_class=CLASS_NONE),
    ]
    assert list(failure_class_counts(runs).items()) == [
        (CLASS_SYNTAX, 2),
        (CLASS_SCENARIO_FAILED, 1),
    ]


def test_collect_runs_derives_failure_class_from_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "L3_gemini_lite_multishot" / "run-004"
    (run_dir / "evaluation").mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "condition": {"name": "L3_gemini_lite_multishot", "guidance_level": "L3"},
                "status": "ok",
                "usage_total": {"prompt_token_count": 10, "candidates_token_count": 5},
                "llm_calls": 2,
                "execution_feedback_rounds": 1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "evaluation" / "summary.json").write_text(
        json.dumps(
            {
                "status": "execution_failed",
                "success": {"syntax_ok": True, "executed_ok": False, "semantics_ok": False},
                "behave": {
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                    "scenarios_failed": 1,
                    "scenarios_error": 0,
                },
                "adherence": {"api": "kestrel", "rejections": [], "ok": True},
                "proxy_stats": {
                    "window_duration_s": 1.0,
                    "by_operation": {},
                    "requests": [
                        {
                            "operation": "write",
                            "method": "PUT",
                            "path": "/v1/tenants/t/stores/s/series/%22benchmark%22/points",
                            "status_code": 201,
                            "latency_ms": 1.0,
                            "error": False,
                        },
                        {
                            "operation": "delete",
                            "method": "DELETE",
                            "path": "/v1/tenants/t/stores/s/series/%22benchmark%22/points",
                            "status_code": 400,
                            "latency_ms": 1.0,
                            "error": True,
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    (run,) = collect_runs(csv_path=None, artifacts_dir=tmp_path).runs

    assert run.quoted_literal_requests == 2
    assert run.failure_class == CLASS_QUOTED_LITERAL
    assert "2 request(s)" in run.failure_detail
    assert run.llm_calls == 2 and run.usage_total["prompt_token_count"] == 10
    assert run.execution_feedback_rounds == 1


def _ladder(successes: dict[str, tuple[int, int]]) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for level, (ok, fail) in successes.items():
        for index in range(ok + fail):
            success = index < ok
            runs.append(
                _run(
                    run_id=f"{level}_gemini_lite_zeroshot/run-{index + 1:03d}",
                    condition_name=f"{level}_gemini_lite_zeroshot",
                    guidance_level=level,
                    model="gemini_lite",
                    status="ok" if success else "execution_failed",
                    executed_ok=success,
                    semantics_ok=success,
                    success=success,
                )
            )
    return runs


def test_fisher_control_vs_rest_uses_l1_as_control() -> None:
    runs = _ladder({"L1": (0, 10), "L2": (10, 0), "L3": (6, 4), "L4": (8, 2), "L5": (7, 3)})
    result = fisher_control_vs_rest(runs)
    assert result.applicable and result.groups == ["L1", "L2–L5"]
    assert result.group_sizes == [10, 40]
    assert result.p_value is not None and result.p_value < 1e-4
    assert result.effect == pytest.approx(0.0 - 31 / 40)
    assert "0/10 vs. 31/40" in result.note


def test_permutation_test_excludes_control_and_is_reproducible() -> None:
    runs = _ladder({"L1": (0, 10), "L2": (10, 0), "L3": (6, 4), "L4": (8, 2), "L5": (7, 3)})
    first = permutation_success_by_level(runs, permutations=2000)
    second = permutation_success_by_level(runs, permutations=2000)
    assert first.groups == ["L2", "L3", "L4", "L5"]
    assert first.p_value == second.p_value
    assert first.p_value is not None and 0 < first.p_value < 0.5
    assert first.statistic == pytest.approx(5.018, abs=0.01)
    flat = _ladder({"L2": (8, 2), "L3": (8, 2), "L4": (8, 2), "L5": (8, 2)})
    assert permutation_success_by_level(flat, permutations=500).p_value == 1.0
    ceiling = _ladder({"L2": (10, 0), "L3": (10, 0)})
    assert not permutation_success_by_level(ceiling).applicable


def test_pairwise_fisher_vs_l2_is_holm_corrected() -> None:
    runs = _ladder({"L2": (10, 0), "L3": (6, 4), "L4": (8, 2), "L5": (7, 3)})
    results = pairwise_fisher_vs_reference(runs)
    apply_holm(results)
    assert [result.groups for result in results] == [["L2", "L3"], ["L2", "L4"], ["L2", "L5"]]
    assert all(result.family == "stage_a_pairwise" and result.applicable for result in results)
    raw = [result.p_value for result in results]
    adjusted = [result.p_adjusted for result in results]
    assert raw[0] == pytest.approx(0.0867, abs=0.001)
    assert all(
        adjusted_p is not None and raw_p is not None and adjusted_p >= raw_p
        for adjusted_p, raw_p in zip(adjusted, raw, strict=True)
    )
    assert max(adjusted_p for adjusted_p in adjusted if adjusted_p is not None) <= 1.0
    assert results[0].adjust_method == "Holm (m=3)"
    assert results[0].effect == pytest.approx(-0.4)


def test_apply_holm_handles_families_and_singletons() -> None:
    tests = [
        TestResult(name="a", description="", family="f", applicable=True, p_value=0.01),
        TestResult(name="b", description="", family="f", applicable=True, p_value=0.04),
        TestResult(name="c", description="", family="f", applicable=True, p_value=0.03),
        TestResult(name="d", description="", applicable=True, p_value=0.2),
        TestResult(name="e", description="", family="f", applicable=False, p_value=None),
    ]
    apply_holm(tests)
    by_name = {test.name: test for test in tests}
    assert by_name["a"].p_adjusted == pytest.approx(0.03)
    assert by_name["c"].p_adjusted == pytest.approx(0.06)
    assert by_name["b"].p_adjusted == pytest.approx(0.06)
    assert by_name["d"].p_adjusted == 0.2 and by_name["d"].adjust_method == ""
    assert by_name["e"].p_adjusted is None


def test_compute_statistics_includes_exact_tests_and_failure_classes() -> None:
    runs = _ladder({"L1": (0, 4), "L2": (4, 0), "L3": (2, 2), "L4": (3, 1), "L5": (3, 1)})
    for run in runs:
        if not run.success:
            run.failure_class = CLASS_SCENARIO_FAILED
    report = compute_statistics(runs)
    names = [test.name for test in report.tests]
    assert "Fisher exakt (Kontrolle)" in names and "Permutationstest L2–L5" in names
    assert "Fisher exakt L2 vs. L3" in names
    stage_b = [test for test in report.tests if test.family == "stage_b"]
    assert stage_b and all(not test.applicable for test in stage_b)
    l3 = next(cell for cell in report.cells if cell.guidance_level == "L3")
    assert l3.failure_classes == {CLASS_SCENARIO_FAILED: 2}
