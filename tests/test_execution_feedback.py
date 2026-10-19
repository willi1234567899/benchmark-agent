from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark_agent.agent.execution_feedback import (
    ExecutionFeedback,
    format_execution_feedback,
    judge_execution,
)
from benchmark_agent.agent.feedback import derive_feedback
from benchmark_agent.config import load_app_config
from benchmark_agent.evaluation.contamination import scan_text
from benchmark_agent.models import (
    BehaveRunResult,
    ContextDocument,
    OperationWindowStats,
    ProxyWindowStats,
    RequestSample,
    ValidationResult,
)
from benchmark_agent.orchestration.runner import GenerationRunner


def _sample(
    operation: str, path: str, *, rejection: str | None = None, error: bool = False
) -> RequestSample:
    return RequestSample(
        operation=operation,
        method="GET" if operation == "read" else "PUT",
        path=path,
        status_code=400 if error else 200,
        latency_ms=1.0,
        error=error,
        rejection=rejection,
        api="kestrel",
    )


def _stats(samples: list[RequestSample]) -> ProxyWindowStats:
    by_operation = {
        op: OperationWindowStats(
            n_requests=sum(1 for sample in samples if sample.operation == op),
            error_count=sum(1 for sample in samples if sample.operation == op and sample.error),
            n_points=sum(sample.n_points for sample in samples if sample.operation == op),
        )
        for op in ("write", "read", "delete")
    }
    return ProxyWindowStats(window_duration_s=1.0, by_operation=by_operation, requests=samples)


def _behave(*, failed: int = 1, stdout: str = "") -> BehaveRunResult:
    return BehaveRunResult(
        exit_code=1 if failed else 0,
        stdout=stdout,
        stderr="",
        timed_out=False,
        scenarios_passed=0 if failed else 1,
        scenarios_failed=failed,
        scenarios_error=0,
    )


def test_judge_execution_reports_rejections_and_workload_gaps_neutrally(
    project_root: Path,
) -> None:
    workload = load_app_config(project_root).workload
    stats = _stats(
        [
            *(
                _sample(
                    "read",
                    "/v1/tenants/t/stores/s/series/benchmark/points?since=PT2HOURS",
                    rejection="invalid_since",
                    error=True,
                )
                for _ in range(20)
            ),
        ]
    )
    stdout = (
        "Assertion Failed: read returned 400\n"
        "some noise line\n"
        "AssertionError: expected precision=ns in the query\n"
    )
    verdict = judge_execution(workload, "kestrel", _behave(stdout=stdout), stats)

    assert not verdict.ok and not verdict.executed_ok
    assert verdict.rejections == [
        "20 request(s) rejected with `invalid_since` "
        "(GET /v1/tenants/t/stores/s/series/benchmark/points)"
    ]
    assert "write/points" in verdict.failed_checks and "read/labels" in verdict.failed_checks
    text = verdict.text
    assert "Execution feedback" in text
    assert "rejected with `invalid_since`" in text
    assert "did not accept exactly 1000 written points" in text
    assert "last 2 hour(s)" in text and "at most 25 points" in text
    assert "host = h1" in text
    assert "Assertion Failed: read returned 400" in text
    assert "precision" not in text and verdict.withheld_lines == 1
    assert "1 output line(s) withheld" in text
    assert scan_text(text).total == 0
    entry = verdict.log_entry()
    assert entry["kind"] == "execution" and entry["behave"]["scenarios_failed"] == 1


def test_judge_execution_accepts_a_conforming_run(project_root: Path) -> None:
    workload = load_app_config(project_root).workload
    read_query = json.dumps(
        {
            "query": 'from(bucket: "benchmark") |> range(start: -2h) '
            '|> filter(fn: (r) => r._measurement == "benchmark" and r.host == "h1") '
            "|> limit(n: 25)"
        }
    )
    delete_body = json.dumps(
        {
            "start": "1970-01-01T00:00:00Z",
            "stop": "2100-01-01T00:00:00Z",
            "predicate": '_measurement="benchmark" AND host="h1"',
        }
    )
    samples = [
        *(
            RequestSample(
                operation="write",
                method="PUT",
                path="/v1/x/points?unit=ms",
                status_code=201,
                latency_ms=1.0,
                error=False,
                n_points=100,
                upstream_path="/api/v2/write?precision=ms",
                api="kestrel",
            )
            for _ in range(10)
        ),
        *(
            RequestSample(
                operation="read",
                method="GET",
                path="/v1/x/points",
                status_code=200,
                latency_ms=1.0,
                error=False,
                body_excerpt=read_query,
                api="kestrel",
            )
            for _ in range(20)
        ),
        RequestSample(
            operation="delete",
            method="DELETE",
            path="/v1/x/points",
            status_code=200,
            latency_ms=1.0,
            error=False,
            body_excerpt=delete_body,
            api="kestrel",
        ),
    ]
    verdict = judge_execution(workload, "kestrel", _behave(failed=0), _stats(samples))
    assert verdict.ok and verdict.text == "" and verdict.failed_checks == []


def test_format_execution_feedback_without_any_request(project_root: Path) -> None:
    workload = load_app_config(project_root).workload
    text, withheld = format_execution_feedback(
        workload,
        _behave(stdout="Traceback (most recent call last):\nKeyError: 'config'"),
        _stats([]),
        None,
        [],
        adherence_ok=True,
    )
    assert "No request reached the service" in text
    assert "KeyError: 'config'" in text and withheld == 0


def test_derive_feedback_carries_execution_text_and_hint() -> None:
    validation = ValidationResult(syntax_ok=True, issues=[])
    hints = [ContextDocument(title="h", source_path="h.md", content="## Hint 1\nUse PUT.")]
    text = derive_feedback(
        mode="hybrid",
        validation=validation,
        feedback_documents=hints,
        round_index=0,
        execution="Execution feedback: the service rejected 20 request(s).",
    )
    assert text.startswith("Execution feedback: the service rejected")
    assert "Hint 1: Use PUT." in text
    assert "failed validation" not in text


def test_l4_regenerates_after_failed_execution_feedback(
    project_root: Path,
    isolated_artifacts: Path,
    patch_llm_client: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verdicts = [
        ExecutionFeedback(
            ok=False,
            executed_ok=False,
            semantics_ok=False,
            text="Execution feedback: 1 request(s) rejected with `unsupported_parameter`.",
            rejections=["1 request(s) rejected with `unsupported_parameter` (GET /x)"],
            failed_checks=["read/labels"],
            behave=_behave(),
        ),
        ExecutionFeedback(
            ok=True, executed_ok=True, semantics_ok=True, text="", behave=_behave(failed=0)
        ),
    ]
    calls: list[Path] = []

    def _fake_execute(app_config, feature_path, steps_path, *, proxy=None):
        calls.append(feature_path)
        return verdicts[len(calls) - 1]

    monkeypatch.setattr("benchmark_agent.agent.agent.execute_for_feedback", _fake_execute)
    monkeypatch.setattr(
        "benchmark_agent.agent.agent.GenerationAgent._sut_reachable", lambda self: True
    )

    result = GenerationRunner(project_root).run_condition("L4_gemini_zeroshot")

    assert len(calls) == 2
    assert result.feedback_iterations == 1
    assert "Execution feedback: 1 request(s) rejected" in result.prompt.full_prompt
    metadata = json.loads(result.artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["execution_feedback_rounds"] == 1
    assert metadata["feedback_iterations"] == 1 and metadata["status"] == "ok"
    rounds = json.loads(
        (result.artifacts.run_directory / "feedback_rounds.json").read_text(encoding="utf-8")
    )
    assert rounds[0]["kind"] == "execution" and rounds[0]["failed_checks"] == ["read/labels"]
    assert (result.artifacts.run_directory / "attempts" / "attempt-00" / "steps.py").exists()


def test_l4_skips_execution_feedback_without_sut(
    project_root: Path,
    isolated_artifacts: Path,
    patch_llm_client: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "benchmark_agent.agent.agent.GenerationAgent._sut_reachable", lambda self: False
    )
    result = GenerationRunner(project_root).run_condition("L4_gemini_zeroshot")
    metadata = json.loads(result.artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["execution_feedback_skipped"] == "SUT not reachable"
    assert metadata["execution_feedback_rounds"] == 0 and result.feedback_iterations == 0
