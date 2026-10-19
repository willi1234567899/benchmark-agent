from __future__ import annotations

import re
from pathlib import PurePath

from benchmark_agent.models import ContextDocument, ValidationResult

_HINT_RE = re.compile(r"^##\s+(Hint\s+\d+)\s*$", re.MULTILINE)


def derive_feedback(
    *,
    mode: str | None,
    validation: ValidationResult,
    feedback_documents: list[ContextDocument],
    round_index: int,
    execution: str | None = None,
) -> str:
    normalized = (mode or "hybrid").lower()
    parts: list[str] = []

    if normalized in {"auto", "hybrid"}:
        auto = _auto_feedback(validation)
        if auto:
            parts.append(auto)
        if execution:
            parts.append(execution)

    if normalized in {"scripted", "hybrid"}:
        hint = _scripted_hint(feedback_documents, round_index)
        if hint:
            parts.append(hint)

    if not parts:
        parts.append(
            "Developer feedback: previous artifacts failed validation. "
            "Regenerate syntactically valid Gherkin and Python steps."
        )
    return "\n\n".join(parts)


def extract_hint_pool(feedback_documents: list[ContextDocument]) -> list[str]:
    hints: list[str] = []
    for document in feedback_documents:
        parts = _HINT_RE.split(document.content)
        if len(parts) < 3:
            continue
        for index in range(1, len(parts), 2):
            title = parts[index].strip()
            body = parts[index + 1].strip() if index + 1 < len(parts) else ""
            if body:
                hints.append(f"{title}: {body}")
    return hints


def _auto_feedback(validation: ValidationResult) -> str | None:
    if validation.syntax_ok or not validation.issues:
        return None
    lines = ["Automatic validator feedback:"]
    for issue in validation.issues:
        location = f" ({PurePath(issue.path).name})" if issue.path else ""
        lines.append(f"- [{issue.code}]{location} {issue.message}")
    return "\n".join(lines)


def _scripted_hint(feedback_documents: list[ContextDocument], round_index: int) -> str | None:
    pool = extract_hint_pool(feedback_documents)
    if not pool:
        if not feedback_documents or round_index > 0:
            return None
        return f"Curated feedback hint:\n{feedback_documents[0].content.strip()}"
    if round_index >= len(pool):
        return None
    hint = pool[round_index]
    return f"Curated developer hint (round {round_index + 1}):\n{hint}"
