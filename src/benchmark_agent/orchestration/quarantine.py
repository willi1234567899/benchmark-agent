from __future__ import annotations

import json
import shutil
from pathlib import Path

from pydantic import BaseModel

from benchmark_agent.orchestration.workspace import (
    QUARANTINE_DIR_NAME,
    QUARANTINE_FILE_NAME,
    RUN_DIR_RE,
)
from benchmark_agent.results_io import remove_metric_rows, utc_timestamp, write_metric_rows

ARCHIVED_ROWS_NAME = "metrics_rows.csv"


class QuarantineResult(BaseModel):
    run_id: str
    source: Path
    target: Path
    rows_removed: int
    reason: str


def quarantine_run(
    run_directory: Path,
    *,
    artifacts_dir: Path,
    csv_path: Path,
    reason: str,
) -> QuarantineResult:
    if not reason.strip():
        raise ValueError("A quarantine reason is required (what did the harness do wrong?).")
    source = run_directory.resolve()
    artifacts_root = artifacts_dir.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"{run_directory} is not a directory")
    try:
        condition, run_name = source.relative_to(artifacts_root).parts
    except ValueError as error:
        raise ValueError(
            f"{source} is not a <condition>/run-NNN directory inside {artifacts_root}"
        ) from error
    if not RUN_DIR_RE.match(run_name):
        raise ValueError(f"{source} is not a run-NNN directory")
    if condition.startswith("_"):
        raise ValueError(f"{condition} is a fixture/reference condition, not a benchmark run")

    run_id = f"{condition}/{run_name}"
    target = artifacts_root / QUARANTINE_DIR_NAME / condition / run_name
    if target.exists():
        raise FileExistsError(f"{target} already exists")

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    removed_rows = remove_metric_rows(csv_path, run_id)
    if removed_rows:
        write_metric_rows(target / ARCHIVED_ROWS_NAME, removed_rows)
    note = {
        "run_id": run_id,
        "condition": condition,
        "run": run_name,
        "reason": reason.strip(),
        "quarantined_at": utc_timestamp(),
        "original_directory": str(source),
        "metrics_rows_removed": len(removed_rows),
    }
    (target / QUARANTINE_FILE_NAME).write_text(json.dumps(note, indent=2), encoding="utf-8")
    return QuarantineResult(
        run_id=run_id, source=source, target=target, rows_removed=len(removed_rows), reason=reason
    )
