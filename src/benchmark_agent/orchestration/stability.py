from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from benchmark_agent.models import STATUS_INFRASTRUCTURE_FAILED

Verdict = Literal["success", "model_failure", "harness_failure", "skipped"]

BUDGET_STOP_STATUS = "budget_stop"
_HARNESS_ERROR_RE = re.compile(
    r"Gemini API call failed|RESOURCE_EXHAUSTED|\b(429|500|502|503|504)\b|"
    r"quota|rate limit|connection (refused|reset|error)|Max retries|"
    r"proxy|docker|compose|SUT (is )?not reachable|needs a reachable SUT|"
    r"budget guard|Permission denied|WinError",
    re.IGNORECASE,
)


class MatrixStability(BaseModel):
    file: str
    timestamp: str | None = None
    n_runs: int = 0
    n_success: int = 0
    n_model_failures: int = 0
    n_harness_failures: int = 0
    n_skipped: int = 0
    conditions: list[str] = Field(default_factory=list)
    harness_errors: list[str] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        return self.n_runs > 0 and self.n_harness_failures == 0 and self.n_skipped == 0


def classify_outcome(outcome: dict[str, object]) -> Verdict:
    if outcome.get("success") is True:
        return "success"
    evaluation_status = str(outcome.get("evaluation_status") or "")
    error = str(outcome.get("error") or "")
    if evaluation_status == BUDGET_STOP_STATUS:
        return "skipped"
    if evaluation_status == STATUS_INFRASTRUCTURE_FAILED:
        return "harness_failure"
    if error and _HARNESS_ERROR_RE.search(error):
        return "harness_failure"
    return "model_failure"


def summarize_matrix(path: Path) -> MatrixStability | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError, ValueError:
        return None
    outcomes = payload.get("outcomes") if isinstance(payload, dict) else None
    if not isinstance(outcomes, list):
        return None
    summary = MatrixStability(file=path.name, timestamp=_text_or_none(payload.get("timestamp")))
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            continue
        summary.n_runs += 1
        name = _text_or_none(outcome.get("condition_name"))
        if name and name not in summary.conditions:
            summary.conditions.append(name)
        verdict = classify_outcome(outcome)
        if verdict == "success":
            summary.n_success += 1
        elif verdict == "model_failure":
            summary.n_model_failures += 1
        elif verdict == "skipped":
            summary.n_skipped += 1
        else:
            summary.n_harness_failures += 1
            error = (
                _text_or_none(outcome.get("error"))
                or _text_or_none(outcome.get("evaluation_status"))
                or "?"
            )
            summary.harness_errors.append(f"{name} rep {outcome.get('repetition')}: {error[:160]}")
    summary.conditions.sort()
    return summary


def load_stability(raw_dir: Path) -> list[MatrixStability]:
    reports: list[MatrixStability] = []
    if not raw_dir.is_dir():
        return reports
    for path in sorted(raw_dir.glob("matrix_*.json")):
        summary = summarize_matrix(path)
        if summary is not None:
            reports.append(summary)
    reports.reverse()
    return reports


def consecutive_clean(reports: Sequence[MatrixStability]) -> int:
    count = 0
    for report in reports:
        if not report.clean:
            break
        count += 1
    return count


def format_stability(reports: Sequence[MatrixStability], *, required: int = 2) -> str:
    if not reports:
        return "No matrix reports found."
    header = ("matrix", "runs", "success", "model_fail", "harness_fail", "skipped", "clean")
    rows: list[tuple[str, ...]] = [header]
    for report in reports:
        rows.append(
            (
                report.file,
                str(report.n_runs),
                str(report.n_success),
                str(report.n_model_failures),
                str(report.n_harness_failures),
                str(report.n_skipped),
                "yes" if report.clean else "no",
            )
        )
    widths = [max(len(row[col]) for row in rows) for col in range(len(header))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    lines.insert(1, "  ".join("-" * width for width in widths))
    clean = consecutive_clean(reports)
    lines.append(
        f"Consecutive clean matrices (newest first): {clean} of {required} required -> "
        + ("STABLE" if clean >= required else "not yet stable")
    )
    for report in reports[:3]:
        for error in report.harness_errors:
            lines.append(f"  {report.file}: {error}")
    return "\n".join(lines)


def _text_or_none(value: object) -> str | None:
    return str(value) if value else None
