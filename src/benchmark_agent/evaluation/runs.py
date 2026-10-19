from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from benchmark_agent.evaluation.coverage import REQUIRED_OPERATIONS
from benchmark_agent.evaluation.failure_class import classify_failure
from benchmark_agent.models import (
    STATUS_EXECUTION_FAILED,
    STATUS_GENERATION_FAILED,
    STATUS_INFRASTRUCTURE_FAILED,
    STATUS_NOT_MEASURED,
    STATUS_OK,
    STATUS_VALIDATION_FAILED,
    MetricRow,
)
from benchmark_agent.orchestration.workspace import QUARANTINE_DIR_NAME, QUARANTINE_FILE_NAME
from benchmark_agent.results_io import read_metric_rows

STATUS_NOT_EVALUATED = "not_evaluated"
STATUS_ORDER: tuple[str, ...] = (
    STATUS_OK,
    STATUS_EXECUTION_FAILED,
    STATUS_VALIDATION_FAILED,
    STATUS_GENERATION_FAILED,
    STATUS_INFRASTRUCTURE_FAILED,
    STATUS_NOT_MEASURED,
    STATUS_NOT_EVALUATED,
)
SOURCE_PROXY = "proxy"
SOURCE_SERVER = "server"
_MEASURED_SOURCES = (SOURCE_PROXY, SOURCE_SERVER)
_RUN_DIR_RE = re.compile(r"^run-(\d{3,})$")


class OpStats(BaseModel):
    n_requests: int = 0
    error_count: int = 0
    n_points: int = 0
    throughput_ops_s: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    covered: bool = False

    @property
    def exercised(self) -> bool:
        return (self.n_requests - self.error_count) > 0


class RunRecord(BaseModel):
    run_id: str
    condition_id: int | None = None
    condition_name: str
    guidance_level: str | None = None
    model: str | None = None
    prompting: str | None = None
    repetition: int = 1
    seed: int | None = None
    model_version: str | None = None
    status: str = STATUS_NOT_EVALUATED
    generation_status: str | None = None
    syntax_ok: bool = False
    executed_ok: bool = False
    semantics_ok: bool = False
    success: bool = False
    coverage_ratio: float | None = None
    covered_operations: list[str] = Field(default_factory=list)
    feedback_iterations: int = 0
    attempts: int | None = None
    probe_calls: int | None = None
    generated_at: str | None = None
    evaluated_at: str | None = None
    scenarios_passed: int | None = None
    scenarios_failed: int | None = None
    scenarios_error: int | None = None
    behave_exit_code: int | None = None
    timed_out: bool | None = None
    error: str | None = None
    skip_reason: str | None = None
    validation_issues: list[str] = Field(default_factory=list)
    conformance_ok: bool | None = None
    conformance_issues: list[str] = Field(default_factory=list)
    conformance_deviations: list[str] = Field(default_factory=list)
    spec_coverage: float | None = None
    criteria_version: int | None = None
    epoch: int | None = None
    prompt_sha256: str | None = None
    temperature: float | None = None
    llm_backend: str | None = None
    sut_api: str | None = None
    adherence_ok: bool | None = None
    api_rejections: list[str] = Field(default_factory=list)
    api_legacy_requests: int | None = None
    vocab_leak: int | None = None
    vocab_hits: dict[str, int] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    usage_total: dict[str, Any] = Field(default_factory=dict)
    llm_calls: int | None = None
    quoted_literal_requests: int = 0
    failure_class: str = "none"
    failure_detail: str = ""
    execution_feedback_rounds: int = 0
    measurements: dict[str, dict[str, OpStats]] = Field(default_factory=dict)
    run_directory: str | None = None
    is_fixture: bool = False
    synthetic: bool = False

    def op(self, source: str, operation: str) -> OpStats | None:
        return self.measurements.get(source, {}).get(operation)

    def stage_b_value(self, source: str, operation: str, metric: str) -> float | None:
        if not self.executed_ok:
            return None
        stats = self.op(source, operation)
        if stats is None or not stats.exercised:
            return None
        value = getattr(stats, metric)
        return float(value) if value is not None else None


class RunCollection(BaseModel):
    runs: list[RunRecord] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    csv_path: str | None = None
    artifacts_dir: str | None = None
    csv_row_count: int = 0
    quarantined: list[dict[str, str]] = Field(default_factory=list)

    @property
    def measured(self) -> list[RunRecord]:
        return [run for run in self.runs if run.status != STATUS_NOT_EVALUATED]


def collect_runs(
    *,
    csv_path: Path | None,
    artifacts_dir: Path | None,
    project_root: Path | None = None,
) -> RunCollection:
    collection = RunCollection(
        csv_path=_display(csv_path, project_root),
        artifacts_dir=_display(artifacts_dir, project_root),
    )
    by_id: dict[str, RunRecord] = {}

    if csv_path is not None:
        try:
            rows = read_metric_rows(csv_path)
        except ValueError as error:
            collection.issues.append(f"metrics.csv unreadable: {error}")
            rows = []
        collection.csv_row_count = len(rows)
        for run_id, run_rows in _group_rows(rows).items():
            by_id[run_id] = _record_from_rows(run_id, run_rows)

    if artifacts_dir is not None and artifacts_dir.is_dir():
        for run_directory in _iter_run_directories(artifacts_dir):
            run_id = f"{run_directory.parent.name}/{run_directory.name}"
            try:
                trace = _read_trace(run_directory)
            except ValueError as error:
                collection.issues.append(f"{run_id}: {error}")
                continue
            record = by_id.get(run_id)
            if record is None:
                record = _record_from_trace(run_id, run_directory, trace)
                by_id[run_id] = record
            _enrich_from_trace(record, trace)
            record.run_directory = _display(run_directory, project_root)
        collection.quarantined = _list_quarantined(artifacts_dir / QUARANTINE_DIR_NAME)

    for record in by_id.values():
        record.failure_class, record.failure_detail = classify_failure(record)
    collection.runs = sorted(by_id.values(), key=_sort_key)
    return collection


def _list_quarantined(quarantine_dir: Path) -> list[dict[str, str]]:
    if not quarantine_dir.is_dir():
        return []
    listed: list[dict[str, str]] = []
    for run_directory in _iter_run_directories(quarantine_dir):
        run_id = f"{run_directory.parent.name}/{run_directory.name}"
        try:
            note = _read_json(run_directory / QUARANTINE_FILE_NAME) or {}
        except ValueError:
            note = {}
        listed.append(
            {
                "run_id": run_id,
                "reason": str(note.get("reason") or f"(no {QUARANTINE_FILE_NAME})"),
                "quarantined_at": str(note.get("quarantined_at") or ""),
            }
        )
    return listed


def _group_rows(rows: list[MetricRow]) -> dict[str, list[MetricRow]]:
    grouped: dict[str, list[MetricRow]] = defaultdict(list)
    for row in rows:
        key = row.run_id or f"{row.condition_name}/rep-{row.repetition}"
        grouped[key].append(row)
    return grouped


def _record_from_rows(run_id: str, rows: list[MetricRow]) -> RunRecord:
    latest_eval = max(row.evaluation_id or row.timestamp for row in rows)
    rows = [row for row in rows if (row.evaluation_id or row.timestamp) == latest_eval]
    head = rows[0]
    measurements: dict[str, dict[str, OpStats]] = {}
    for row in rows:
        if row.source not in _MEASURED_SOURCES:
            continue
        measurements.setdefault(row.source, {})[row.operation] = OpStats(
            n_requests=row.n_requests,
            error_count=row.error_count,
            n_points=row.n_points,
            throughput_ops_s=row.throughput_ops_s,
            latency_p50_ms=row.latency_p50_ms,
            latency_p95_ms=row.latency_p95_ms,
            latency_p99_ms=row.latency_p99_ms,
            covered=row.covered,
        )
    proxy = measurements.get(SOURCE_PROXY, {})
    covered = [
        operation for operation in REQUIRED_OPERATIONS if proxy.get(operation, OpStats()).covered
    ]
    coverage = len(covered) / len(REQUIRED_OPERATIONS) if proxy else None
    success = head.syntax_ok and head.executed_ok and head.semantics_ok
    return RunRecord(
        run_id=run_id,
        condition_id=head.condition_id,
        condition_name=head.condition_name,
        guidance_level=head.guidance_level,
        model=head.model,
        prompting=head.prompting,
        repetition=head.repetition,
        seed=head.seed,
        model_version=head.model_version,
        status=head.status,
        syntax_ok=head.syntax_ok,
        executed_ok=head.executed_ok,
        semantics_ok=head.semantics_ok,
        success=success,
        coverage_ratio=coverage,
        covered_operations=covered,
        feedback_iterations=head.feedback_iterations,
        evaluated_at=head.evaluation_id or head.timestamp,
        measurements=measurements,
        is_fixture=head.condition_name.startswith("_"),
    )


class _Trace:
    def __init__(self, run_directory: Path) -> None:
        self.run_directory = run_directory
        self.metadata: dict[str, Any] = _read_json(run_directory / "metadata.json") or {}
        evaluation_dir = run_directory / "evaluation"
        self.summary: dict[str, Any] = _read_json(evaluation_dir / "summary.json") or {}
        self.behave: dict[str, Any] = _read_json(evaluation_dir / "behave.json") or {}
        self.has_artifacts = (run_directory / "benchmark.feature").is_file() and (
            run_directory / "steps.py"
        ).is_file()


def _iter_run_directories(artifacts_dir: Path) -> list[Path]:
    found: list[Path] = []
    for condition_dir in sorted(artifacts_dir.iterdir()):
        if not condition_dir.is_dir():
            continue
        for run_directory in sorted(condition_dir.iterdir()):
            if run_directory.is_dir() and _RUN_DIR_RE.match(run_directory.name):
                found.append(run_directory)
    return found


def _read_trace(run_directory: Path) -> _Trace:
    return _Trace(run_directory)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise ValueError(f"{path.name} unreadable: {error}") from error
    return payload if isinstance(payload, dict) else None


def _record_from_trace(run_id: str, run_directory: Path, trace: _Trace) -> RunRecord:
    condition = trace.metadata.get("condition")
    condition = condition if isinstance(condition, dict) else {}
    model = trace.metadata.get("model")
    model = model if isinstance(model, dict) else {}
    condition_name = str(condition.get("name") or run_directory.parent.name)
    return RunRecord(
        run_id=run_id,
        condition_id=_as_int(condition.get("id")),
        condition_name=condition_name,
        guidance_level=_as_str(condition.get("guidance_level")),
        model=_as_str(condition.get("model")),
        prompting=_as_str(condition.get("prompting_strategy")),
        repetition=_as_int(trace.metadata.get("repetition"))
        or _repetition_from_name(run_directory.name),
        seed=_as_int(trace.metadata.get("seed")),
        model_version=_as_str(model.get("model_version")),
        status=STATUS_NOT_EVALUATED,
        is_fixture=condition_name.startswith("_"),
    )


def _enrich_from_trace(record: RunRecord, trace: _Trace) -> None:
    _enrich_from_metadata(record, trace.metadata)

    summary = trace.summary
    if summary:
        _enrich_from_summary(record, summary)
    elif record.status == STATUS_NOT_EVALUATED and record.generation_status in {
        STATUS_GENERATION_FAILED,
        STATUS_VALIDATION_FAILED,
    }:
        record.status = record.generation_status
        record.syntax_ok = False

    behave = trace.behave or summary.get("behave")
    if isinstance(behave, dict):
        record.scenarios_passed = _as_int(behave.get("scenarios_passed"))
        record.scenarios_failed = _as_int(behave.get("scenarios_failed"))
        record.scenarios_error = _as_int(behave.get("scenarios_error"))
        record.behave_exit_code = _as_int(behave.get("exit_code"))
        timed_out = behave.get("timed_out")
        record.timed_out = timed_out if isinstance(timed_out, bool) else None


def _enrich_from_metadata(record: RunRecord, metadata: dict[str, Any]) -> None:
    record.generation_status = _as_str(metadata.get("status"))
    record.generated_at = _as_str(metadata.get("timestamp")) or record.generated_at
    attempt = _as_int(metadata.get("attempt"))
    record.attempts = attempt + 1 if attempt is not None else None
    record.probe_calls = _as_int(metadata.get("probe_calls"))
    if record.feedback_iterations == 0:
        record.feedback_iterations = _as_int(metadata.get("feedback_iterations")) or 0
    usage = metadata.get("usage")
    if isinstance(usage, dict):
        record.usage = dict(usage)
    usage_total = metadata.get("usage_total")
    if isinstance(usage_total, dict):
        record.usage_total = dict(usage_total)
    record.llm_calls = _as_int(metadata.get("llm_calls"))
    record.execution_feedback_rounds = _as_int(metadata.get("execution_feedback_rounds")) or 0
    if record.model_version is None:
        record.model_version = _as_str(metadata.get("response_model_version"))
    if record.error is None:
        record.error = _as_str(metadata.get("error"))
    record.epoch = _as_int(metadata.get("epoch"))
    record.prompt_sha256 = _as_str(metadata.get("prompt_sha256"))
    record.llm_backend = _as_str(metadata.get("llm_backend"))
    model = metadata.get("model")
    parameters = model.get("parameters") if isinstance(model, dict) else None
    temperature = parameters.get("temperature") if isinstance(parameters, dict) else None
    record.temperature = float(temperature) if isinstance(temperature, int | float) else None
    record.sut_api = _as_str(metadata.get("sut_api"))


def _enrich_from_summary(record: RunRecord, summary: dict[str, Any]) -> None:
    record.criteria_version = _as_int(summary.get("criteria_version")) or 1
    adherence = summary.get("adherence")
    if isinstance(adherence, dict):
        record.adherence_ok = bool(adherence.get("ok"))
        record.api_legacy_requests = _as_int(adherence.get("legacy"))
        rejections = adherence.get("rejections")
        if isinstance(rejections, list):
            record.api_rejections = [str(item) for item in rejections]
        if record.sut_api is None:
            record.sut_api = _as_str(adherence.get("api"))
    vocabulary = summary.get("vocabulary")
    if isinstance(vocabulary, dict):
        record.vocab_leak = _as_int(vocabulary.get("total"))
        hits = vocabulary.get("hits")
        if isinstance(hits, dict):
            record.vocab_hits = {
                str(label): int(count) for label, count in hits.items() if _as_int(count)
            }
    proxy_stats = summary.get("proxy_stats")
    if isinstance(proxy_stats, dict):
        requests = proxy_stats.get("requests")
        if isinstance(requests, list):
            record.quoted_literal_requests = sum(
                1
                for sample in requests
                if isinstance(sample, dict) and "%22" in str(sample.get("path") or "")
            )
    conformance = summary.get("conformance")
    if isinstance(conformance, dict):
        record.conformance_ok = bool(conformance.get("ok"))
        checks = conformance.get("checks")
        if isinstance(checks, list):
            record.conformance_issues = [
                f"{check.get('operation')}/{check.get('name')}: {check.get('detail')}"
                for check in checks
                if isinstance(check, dict) and not check.get("ok")
            ]
        deviations = conformance.get("deviations")
        if isinstance(deviations, list):
            record.conformance_deviations = [str(item) for item in deviations]
        spec_coverage = conformance.get("spec_coverage")
        record.spec_coverage = (
            float(spec_coverage) if isinstance(spec_coverage, int | float) else None
        )
    if record.status == STATUS_NOT_EVALUATED:
        record.status = _as_str(summary.get("status")) or STATUS_NOT_EVALUATED
        success = summary.get("success")
        if isinstance(success, dict):
            record.syntax_ok = bool(success.get("syntax_ok"))
            record.executed_ok = bool(success.get("executed_ok"))
            record.semantics_ok = bool(success.get("semantics_ok"))
            record.success = record.syntax_ok and record.executed_ok and record.semantics_ok
        coverage = summary.get("coverage")
        if isinstance(coverage, dict):
            ratio = coverage.get("coverage_ratio")
            record.coverage_ratio = float(ratio) if isinstance(ratio, int | float) else None
            operations = coverage.get("covered_operations")
            record.covered_operations = (
                [str(operation) for operation in operations] if isinstance(operations, list) else []
            )
    record.skip_reason = _as_str(summary.get("skip_reason"))
    validation = summary.get("validation")
    if isinstance(validation, dict):
        issues = validation.get("issues")
        if isinstance(issues, list):
            record.validation_issues = [
                str(issue.get("message") if isinstance(issue, dict) else issue) for issue in issues
            ]


def _sort_key(record: RunRecord) -> tuple[str, str, int, str]:
    return (
        record.guidance_level or "~",
        record.condition_name,
        record.repetition,
        record.run_id,
    )


def _repetition_from_name(name: str) -> int:
    match = _RUN_DIR_RE.match(name)
    return int(match.group(1)) if match else 1


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _as_str(value: object) -> str | None:
    return None if value is None else str(value)


def _display(path: Path | None, project_root: Path | None) -> str | None:
    if path is None:
        return None
    if project_root is not None:
        try:
            return path.resolve().relative_to(project_root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()
