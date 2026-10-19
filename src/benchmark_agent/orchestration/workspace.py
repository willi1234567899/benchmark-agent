from __future__ import annotations

import re
from pathlib import Path

RUN_DIR_RE = re.compile(r"^run-(\d{3,})$")
QUARANTINE_DIR_NAME = "_quarantine"
QUARANTINE_FILE_NAME = "QUARANTINE.json"


class WorkspaceManager:
    def __init__(self, artifacts_root: Path) -> None:
        self._artifacts_root = artifacts_root

    def create_run_directory(self, condition_name: str) -> Path:
        condition_root = self._artifacts_root / condition_name
        condition_root.mkdir(parents=True, exist_ok=True)

        existing_numbers = _run_numbers(condition_root) + _run_numbers(
            self._artifacts_root / QUARANTINE_DIR_NAME / condition_name
        )
        next_number = max(existing_numbers, default=0) + 1
        while True:
            candidate = condition_root / f"run-{next_number:03d}"
            try:
                candidate.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                next_number += 1
                continue
            return candidate


def _run_numbers(directory: Path) -> list[int]:
    if not directory.is_dir():
        return []
    numbers: list[int] = []
    for path in directory.iterdir():
        match = RUN_DIR_RE.match(path.name)
        if match and path.is_dir():
            numbers.append(int(match.group(1)))
    return numbers
