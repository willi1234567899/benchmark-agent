from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from benchmark_agent.models import BehaveRunResult

_SUMMARY_RE = re.compile(
    r"(\d+)\s+scenarios?\s+passed,\s*(\d+)\s+failed(?:,\s*(\d+)\s+error)?",
    re.IGNORECASE,
)

UTF8_CHILD_ENV: dict[str, str] = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
_SECRET_MARKERS = ("API_KEY", "APIKEY", "SECRET", "PASSWORD", "_TOKEN")


def sanitized_environment(base: Mapping[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in base.items()
        if not any(marker in key.upper() for marker in _SECRET_MARKERS)
    }


class BehaveRunner:
    def __init__(self, timeout_seconds: int) -> None:
        self._timeout_seconds = timeout_seconds

    def run(
        self,
        feature_path: Path,
        steps_path: Path,
        *,
        env: dict[str, str] | None = None,
        work_root: Path | None = None,
    ) -> BehaveRunResult:
        cleanup = work_root is None
        root = work_root or Path(tempfile.mkdtemp(prefix="benchmark-agent-behave-"))
        try:
            features_dir = stage_behave_layout(root, feature_path, steps_path)
            command = [
                sys.executable,
                "-m",
                "behave",
                str(features_dir),
                "--no-capture",
                "--format",
                "plain",
            ]
            base_env = env if env is not None else os.environ
            child_env = {**base_env, **UTF8_CHILD_ENV}
            proc = _spawn(command, cwd=root, env=child_env)
            timed_out = False
            try:
                raw_out, raw_err = proc.communicate(timeout=self._timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_process_tree(proc)
                try:
                    raw_out, raw_err = proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    raw_out, raw_err = b"", b""

            stdout = _decode(raw_out)
            stderr = _decode(raw_err)
            passed, failed, errored = _parse_summary(f"{stdout}\n{stderr}")
            return BehaveRunResult(
                exit_code=124 if timed_out else proc.returncode,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
                scenarios_passed=passed,
                scenarios_failed=failed,
                scenarios_error=errored,
            )
        finally:
            if cleanup:
                shutil.rmtree(root, ignore_errors=True)


def stage_behave_layout(root: Path, feature_path: Path, steps_path: Path) -> Path:
    features_dir = root / "features"
    steps_dir = features_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(feature_path, features_dir / "benchmark.feature")
    shutil.copy2(steps_path, steps_dir / "steps.py")
    return features_dir


def _spawn(command: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.Popen[bytes]:
    if sys.platform == "win32":
        return subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    return subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def kill_process_tree(proc: subprocess.Popen[bytes]) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            check=False,
            timeout=15,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass


def _decode(raw: bytes | None) -> str:
    if not raw:
        return ""
    return raw.decode("utf-8", errors="replace")


def _parse_summary(text: str) -> tuple[int | None, int | None, int | None]:
    match = _SUMMARY_RE.search(text)
    if match is None:
        return None, None, None
    passed = int(match.group(1))
    failed = int(match.group(2))
    errored = int(match.group(3)) if match.group(3) is not None else 0
    return passed, failed, errored
