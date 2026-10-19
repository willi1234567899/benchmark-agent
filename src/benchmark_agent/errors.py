from __future__ import annotations

from pathlib import Path


class GenerationError(RuntimeError):
    def __init__(self, message: str, *, run_directory: Path | None = None) -> None:
        super().__init__(message)
        self.run_directory = run_directory
