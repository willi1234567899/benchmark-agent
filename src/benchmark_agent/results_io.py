from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from benchmark_agent.models import MetricRow

RAW_CSV_COLUMNS: tuple[str, ...] = (
    "condition_id",
    "condition_name",
    "guidance_level",
    "model",
    "prompting",
    "repetition",
    "operation",
    "source",
    "executed_ok",
    "syntax_ok",
    "semantics_ok",
    "throughput_ops_s",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "n_requests",
    "error_count",
    "covered",
    "feedback_iterations",
    "model_version",
    "seed",
    "timestamp",
    "run_id",
    "evaluation_id",
    "status",
    "n_points",
)

_INT_COLUMNS = (
    "condition_id",
    "repetition",
    "n_requests",
    "error_count",
    "feedback_iterations",
    "seed",
    "n_points",
)
_FLOAT_COLUMNS = ("throughput_ops_s", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms")
_BOOL_COLUMNS = ("executed_ok", "syntax_ok", "semantics_ok", "covered")
_OPTIONAL_STR_COLUMNS = ("guidance_level", "model", "prompting", "model_version")
_MISSING = {"", "None", None}


def append_metric_rows(csv_path: Path, rows: list[MetricRow]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not write_header:
        _check_header(csv_path)
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RAW_CSV_COLUMNS))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def replace_metric_rows(csv_path: Path, run_id: str, rows: list[MetricRow]) -> None:
    existing = [row for row in read_metric_rows(csv_path) if row.run_id != run_id]
    write_metric_rows(csv_path, [*existing, *rows])


def remove_metric_rows(csv_path: Path, run_id: str) -> list[MetricRow]:
    rows = read_metric_rows(csv_path)
    removed = [row for row in rows if row.run_id == run_id]
    if removed:
        write_metric_rows(csv_path, [row for row in rows if row.run_id != run_id])
    return removed


def write_metric_rows(csv_path: Path, rows: list[MetricRow]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RAW_CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def read_metric_rows(csv_path: Path) -> list[MetricRow]:
    if not csv_path.is_file():
        return []
    rows: list[MetricRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            try:
                rows.append(MetricRow.model_validate(_normalize_row(row)))
            except (ValidationError, ValueError) as error:
                raise ValueError(f"{csv_path}:{line_number}: {error}") from error
    return rows


def utc_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _check_header(csv_path: Path) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        existing = next(csv.reader(handle), [])
    if tuple(existing) != RAW_CSV_COLUMNS:
        raise ValueError(
            f"{csv_path} header {existing} does not match RAW_CSV_COLUMNS "
            f"{list(RAW_CSV_COLUMNS)}; rotate or migrate the file"
        )


def _normalize_row(row: dict[str, str]) -> dict[str, object]:
    normalized: dict[str, object] = dict(row)
    for key in _INT_COLUMNS:
        value = row.get(key, "")
        normalized[key] = None if value in _MISSING else int(value)
    for key in _FLOAT_COLUMNS:
        value = row.get(key, "")
        normalized[key] = None if value in _MISSING else float(value)
    for key in _BOOL_COLUMNS:
        normalized[key] = str(row.get(key, "")).lower() in {"1", "true", "yes"}
    for key in _OPTIONAL_STR_COLUMNS:
        text = row.get(key)
        normalized[key] = None if text in _MISSING else text
    return normalized
