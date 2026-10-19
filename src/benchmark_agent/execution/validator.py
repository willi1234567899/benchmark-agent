from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from behave.parser import Parser

from benchmark_agent.execution.behave_runner import (
    UTF8_CHILD_ENV,
    kill_process_tree,
    sanitized_environment,
    stage_behave_layout,
)
from benchmark_agent.models import ValidationIssue, ValidationResult

_GHERLINT_LINE_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+):(?P<column>\d+):\s+(?P<text>.+?)\s+\((?P<name>[^)]+)\)\s*$"
)

BLOCKING_GHERLINT_RULES: frozenset[str] = frozenset(
    {
        "missing-parameter",
        "unused-parameter",
        "examples-outside-scenario-outline",
        "file-has-no-feature",
        "empty-feature",
        "empty-scenario",
        "empty-background",
        "missing-feature-name",
        "missing-scenario-name",
    }
)

_DRY_RUN_TIMEOUT_S = 60
_UNDEFINED_RE = re.compile(r"(\d+)\s+undefined", re.IGNORECASE)
_EXCEPTION_RE = re.compile(r"^(?:[A-Za-z_][\w.]*(?:Error|Exception))\b.*$", re.MULTILINE)


def validate_artifacts(
    feature_path: Path,
    steps_path: Path,
    *,
    dry_run: bool = True,
    env: Mapping[str, str] | None = None,
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    if not feature_path.is_file():
        issues.append(
            ValidationIssue(
                code="MISSING_FEATURE",
                message=f"Feature file not found: {feature_path}",
                path=str(feature_path),
            )
        )
    else:
        issues.extend(_parse_feature(feature_path))
        lint_issues, lint_warnings = _lint_feature_with_gherlint(feature_path)
        issues.extend(lint_issues)
        warnings.extend(lint_warnings)

    if not steps_path.is_file():
        issues.append(
            ValidationIssue(
                code="MISSING_STEPS",
                message=f"Steps file not found: {steps_path}",
                path=str(steps_path),
            )
        )
    else:
        issues.extend(_compile_steps(steps_path))

    if dry_run and not issues:
        issues.extend(_dry_run(feature_path, steps_path, env=env))

    return ValidationResult(syntax_ok=not issues, issues=issues, warnings=warnings)


def _parse_feature(feature_path: Path) -> list[ValidationIssue]:
    try:
        Parser().parse(feature_path.read_text(encoding="utf-8"), filename=str(feature_path))
    except Exception as error:
        return [
            ValidationIssue(
                code="E001",
                message=f"Feature could not be parsed: {error}",
                path=str(feature_path),
            )
        ]
    return []


def _compile_steps(steps_path: Path) -> list[ValidationIssue]:
    try:
        compile(steps_path.read_text(encoding="utf-8"), str(steps_path), "exec")
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        return [
            ValidationIssue(
                code="PY_COMPILE",
                message=f"{type(error).__name__}: {error}",
                path=str(steps_path),
            )
        ]
    return []


def _lint_feature_with_gherlint(
    feature_path: Path,
) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
    completed = subprocess.run(
        [sys.executable, "-m", "gherlint", "lint", str(feature_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env={**os.environ, **UTF8_CHILD_ENV},
    )
    issues: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    for line in f"{completed.stdout}\n{completed.stderr}".splitlines():
        match = _GHERLINT_LINE_RE.match(line.strip())
        if match is None:
            continue
        issue = ValidationIssue(
            code=match.group("name"),
            message=f"line {match.group('line')}: {match.group('text')}",
            path=match.group("path"),
        )
        if issue.code in BLOCKING_GHERLINT_RULES:
            issues.append(issue)
        else:
            warnings.append(issue)
    if "Traceback" in (completed.stderr or ""):
        issues.append(
            ValidationIssue(
                code="GHERLINT_CRASH",
                message=_tail(completed.stderr),
                path=str(feature_path),
            )
        )
    return issues, warnings


def _dry_run(
    feature_path: Path,
    steps_path: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> list[ValidationIssue]:
    child_env = {**sanitized_environment(os.environ), **(env or {}), **UTF8_CHILD_ENV}
    root = Path(tempfile.mkdtemp(prefix="benchmark-agent-dryrun-"))
    try:
        features_dir = stage_behave_layout(root, feature_path, steps_path)
        command = [
            sys.executable,
            "-m",
            "behave",
            str(features_dir),
            "--dry-run",
            "--format",
            "plain",
        ]
        proc = subprocess.Popen(
            command,
            cwd=root,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            raw_out, raw_err = proc.communicate(timeout=_DRY_RUN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            kill_process_tree(proc)
            return [
                ValidationIssue(
                    code="DRY_RUN_TIMEOUT",
                    message=f"behave --dry-run did not finish within {_DRY_RUN_TIMEOUT_S}s",
                    path=str(steps_path),
                )
            ]
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if proc.returncode == 0:
        return []
    stdout = raw_out.decode("utf-8", errors="replace")
    stderr = raw_err.decode("utf-8", errors="replace")
    combined = f"{stdout}\n{stderr}"
    exception = _EXCEPTION_RE.search(stderr) or _EXCEPTION_RE.search(stdout)
    if exception is not None:
        return [
            ValidationIssue(
                code="IMPORT_ERROR",
                message=f"steps module failed to load: {exception.group(0).strip()}",
                path=str(steps_path),
            )
        ]
    undefined = _UNDEFINED_RE.search(combined)
    if undefined is not None and int(undefined.group(1)) > 0:
        return [
            ValidationIssue(
                code="UNDEFINED_STEP",
                message=f"{undefined.group(1)} step(s) have no matching step definition",
                path=str(feature_path),
            )
        ]
    return [
        ValidationIssue(
            code="DRY_RUN",
            message=f"behave --dry-run exited with {proc.returncode}: {_tail(combined)}",
            path=str(feature_path),
        )
    ]


def _tail(text: str, line_count: int = 5) -> str:
    non_empty = [line for line in text.splitlines() if line.strip()]
    return "\n".join(non_empty[-line_count:])
