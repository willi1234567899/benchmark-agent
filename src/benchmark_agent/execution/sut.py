from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from pathlib import Path

import httpx

from benchmark_agent.config import SutSettings

_LP_TIMESTAMP_RE = re.compile(r"^(?P<body>.+\S)\s+(?P<ts>-?\d{1,19})\s*$")


def sut_environment(settings: SutSettings, *, influx_url: str) -> dict[str, str]:
    if settings.api == "tessera":
        return {
            "TESSERA_URL": influx_url,
            "TESSERA_KEY": settings.token,
            "TESSERA_SPACE": settings.org,
            "TESSERA_STORE": settings.bucket,
        }
    if settings.api == "kestrel":
        return {
            "KESTREL_ENDPOINT": influx_url,
            "KESTREL_TOKEN": settings.token,
            "KESTREL_TENANT": settings.org,
            "KESTREL_STORE": settings.bucket,
        }
    return {
        "INFLUX_URL": influx_url,
        "INFLUXDB_URL": influx_url,
        "INFLUX_TOKEN": settings.token,
        "INFLUX_ORG": settings.org,
        "INFLUX_BUCKET": settings.bucket,
    }


def compose_project_name(project_root: Path) -> str:
    digest = hashlib.sha1(str(project_root.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"benchmark-agent-{digest}"


def render_seed_lines(text: str, *, now_ns: int) -> str:
    parsed: list[tuple[str, int] | None] = []
    lines = text.splitlines()
    for line in lines:
        match = _LP_TIMESTAMP_RE.match(line)
        if line.lstrip().startswith("#") or not line.strip() or match is None:
            parsed.append(None)
        else:
            parsed.append((match.group("body"), int(match.group("ts"))))
    timestamps = [entry[1] for entry in parsed if entry is not None]
    if not timestamps:
        return text
    offset = (now_ns - 1_000_000_000) - max(timestamps)
    rendered: list[str] = []
    for line, entry in zip(lines, parsed, strict=True):
        if entry is None:
            rendered.append(line)
        else:
            body, timestamp = entry
            rendered.append(f"{body} {timestamp + offset}")
    return "\n".join(rendered) + "\n"


class SutManager:
    def __init__(
        self,
        project_root: Path,
        settings: SutSettings,
        *,
        seed_dir: Path | None = None,
    ) -> None:
        self._project_root = project_root
        self._settings = settings
        self._compose_file = project_root / settings.compose_file
        self._seed_dir = seed_dir if seed_dir is not None else self._compose_file.parent / "seed"

    @property
    def compose_file(self) -> Path:
        return self._compose_file

    @property
    def seed_dir(self) -> Path:
        return self._seed_dir

    def up(self) -> None:
        self._run_compose(["up", "-d", "--wait"])
        self.wait_until_ready()

    def down(self) -> None:
        self._run_compose(["down", "--remove-orphans", "--volumes"])

    def reset(self) -> None:
        self.down()
        self.up()

    def is_reachable(self, *, timeout_seconds: float = 2.0) -> bool:
        try:
            response = httpx.get(self._health_url(), timeout=timeout_seconds)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def seed_files(self) -> list[Path]:
        if not self._seed_dir.is_dir():
            return []
        return sorted(path for path in self._seed_dir.glob("*.lp") if path.is_file())

    def seed(self, *, timeout_seconds: float = 30.0) -> int:
        files = self.seed_files()
        if not files:
            return 0
        base = self._settings.influxdb_direct_url.rstrip("/")
        params = {
            "org": self._settings.org,
            "bucket": self._settings.bucket,
            "precision": "ns",
        }
        headers = {
            "Authorization": f"Token {self._settings.token}",
            "Content-Type": "text/plain; charset=utf-8",
        }
        now_ns = time.time_ns()
        for path in files:
            body = render_seed_lines(path.read_text(encoding="utf-8"), now_ns=now_ns).encode(
                "utf-8"
            )
            try:
                response = httpx.post(
                    f"{base}/api/v2/write",
                    params=params,
                    headers=headers,
                    content=body,
                    timeout=timeout_seconds,
                )
            except httpx.HTTPError as error:
                raise RuntimeError(f"seeding {path.name} failed: {error}") from error
            if response.status_code >= 300:
                detail = response.text.strip()[:500]
                raise RuntimeError(
                    f"seeding {path.name} rejected with HTTP {response.status_code}: {detail}"
                )
        return len(files)

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._settings.ready_timeout_seconds
        health_url = self._health_url()
        last_error = "SUT did not become ready"
        while time.monotonic() < deadline:
            try:
                response = httpx.get(health_url, timeout=2.0)
                if response.status_code == 200:
                    return
                last_error = f"health status {response.status_code}"
            except httpx.HTTPError as error:
                last_error = str(error)
            time.sleep(1.0)
        raise TimeoutError(f"InfluxDB SUT not ready: {last_error}")

    def _health_url(self) -> str:
        return f"{self._settings.influxdb_direct_url.rstrip('/')}/health"

    def _run_compose(self, args: list[str]) -> None:
        if not self._compose_file.is_file():
            raise RuntimeError(f"compose file missing: {self._compose_file}")
        command = ["docker", "compose", "-f", str(self._compose_file), *args]
        env = {
            **os.environ,
            "COMPOSE_PROJECT_NAME": os.environ.get(
                "COMPOSE_PROJECT_NAME", compose_project_name(self._project_root)
            ),
        }
        try:
            completed = subprocess.run(
                command,
                cwd=self._project_root,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                "docker CLI not found on PATH (is Docker Desktop installed and PATH refreshed?)"
            ) from error
        if completed.returncode != 0:
            detail = (completed.stderr or "").strip() or (completed.stdout or "").strip()
            raise RuntimeError(f"docker compose {' '.join(args)} failed: {detail}")
