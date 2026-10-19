from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark_agent.evaluation.runs import RunRecord

CLASS_NONE = "none"
CLASS_GENERATION = "generation_failed"
CLASS_SYNTAX = "syntax_error"
CLASS_INFRASTRUCTURE = "infrastructure"
CLASS_NOT_MEASURED = "not_measured"
CLASS_NOT_EVALUATED = "not_evaluated"
CLASS_TIMEOUT = "timeout"
CLASS_API_REJECTED = "api_rejected"
CLASS_QUOTED_LITERAL = "quoted_literal"
CLASS_STEP_ERROR = "step_error"
CLASS_SCENARIO_FAILED = "scenario_failed"
CLASS_NONCONFORMANT = "nonconformant"
CLASS_UNKNOWN = "unknown"

FAILURE_CLASS_ORDER: tuple[str, ...] = (
    CLASS_NONE,
    CLASS_GENERATION,
    CLASS_SYNTAX,
    CLASS_INFRASTRUCTURE,
    CLASS_NOT_MEASURED,
    CLASS_NOT_EVALUATED,
    CLASS_TIMEOUT,
    CLASS_API_REJECTED,
    CLASS_QUOTED_LITERAL,
    CLASS_STEP_ERROR,
    CLASS_SCENARIO_FAILED,
    CLASS_NONCONFORMANT,
    CLASS_UNKNOWN,
)

FAILURE_CLASS_LABELS: dict[str, str] = {
    CLASS_NONE: "Erfolg",
    CLASS_GENERATION: "keine Artefakte (JSON-Vertrag)",
    CLASS_SYNTAX: "Syntaxfehler (Gherkin/Python)",
    CLASS_INFRASTRUCTURE: "Harness/Infrastruktur",
    CLASS_NOT_MEASURED: "nicht gemessen",
    CLASS_NOT_EVALUATED: "nicht ausgewertet",
    CLASS_TIMEOUT: "Timeout",
    CLASS_API_REJECTED: "API-Ablehnung (Fassade)",
    CLASS_QUOTED_LITERAL: "zitiertes Literal im Pfad",
    CLASS_STEP_ERROR: "Step-Fehler (Exception)",
    CLASS_SCENARIO_FAILED: "Szenario fehlgeschlagen (Assertion)",
    CLASS_NONCONFORMANT: "Workload nicht konform",
    CLASS_UNKNOWN: "unklar",
}

HARNESS_CLASSES: frozenset[str] = frozenset({CLASS_INFRASTRUCTURE})


def classify_failure(run: RunRecord) -> tuple[str, str]:
    if run.success:
        return CLASS_NONE, ""
    status = run.status or ""
    generation_status = run.generation_status or ""
    if status == "generation_failed" or generation_status == "generation_failed":
        return CLASS_GENERATION, _short(run.error)
    if not run.syntax_ok or status == "validation_failed":
        return CLASS_SYNTAX, _short("; ".join(run.validation_issues[:2]))
    if status == "infrastructure_failed":
        return CLASS_INFRASTRUCTURE, _short(run.skip_reason or run.error)
    if status == "not_measured":
        return CLASS_NOT_MEASURED, _short(run.skip_reason)
    if status == "not_evaluated":
        return CLASS_NOT_EVALUATED, ""
    if run.timed_out:
        return CLASS_TIMEOUT, "behave timed out"
    if run.api_rejections:
        return CLASS_API_REJECTED, ", ".join(run.api_rejections)
    if run.quoted_literal_requests:
        return (
            CLASS_QUOTED_LITERAL,
            f"{run.quoted_literal_requests} request(s) with a quoted literal in the URL path",
        )
    if (run.scenarios_error or 0) > 0:
        return CLASS_STEP_ERROR, f"{run.scenarios_error} scenario(s) with an exception"
    if (run.scenarios_failed or 0) > 0:
        return CLASS_SCENARIO_FAILED, f"{run.scenarios_failed} failed scenario(s)"
    if run.executed_ok and not run.semantics_ok:
        checks = [issue.split(":", 1)[0] for issue in run.conformance_issues]
        return CLASS_NONCONFORMANT, ", ".join(checks) if checks else "coverage incomplete"
    return CLASS_UNKNOWN, _short(run.error or run.skip_reason)


def failure_class_counts(runs: list[RunRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        if run.failure_class and run.failure_class != CLASS_NONE:
            counts[run.failure_class] = counts.get(run.failure_class, 0) + 1
    known = {key: counts[key] for key in FAILURE_CLASS_ORDER if key in counts}
    unknown = {key: value for key, value in counts.items() if key not in FAILURE_CLASS_ORDER}
    return known | unknown


def _short(text: str | None, limit: int = 160) -> str:
    if not text:
        return ""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
