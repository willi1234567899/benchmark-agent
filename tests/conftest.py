from __future__ import annotations

import socket
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fake_influx import FakeInfluxServer, start_fake_influx

from benchmark_agent.measurement.proxy import MeasurementProxy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
TESTS_PATH = PROJECT_ROOT / "tests"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(TESTS_PATH) not in sys.path:
    sys.path.insert(0, str(TESTS_PATH))


@pytest.fixture
def project_root(monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(PROJECT_ROOT)
    return PROJECT_ROOT


@pytest.fixture
def isolated_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from benchmark_agent.orchestration.workspace import WorkspaceManager

    artifacts_root = tmp_path / "artifacts"
    monkeypatch.setenv("BENCHMARK_AGENT_PATHS__RESULTS_ARTIFACTS_DIR", str(artifacts_root))

    def _factory(_artifacts_root: Path) -> WorkspaceManager:
        return WorkspaceManager(artifacts_root)

    monkeypatch.setattr("benchmark_agent.agent.agent.WorkspaceManager", _factory)
    return artifacts_root


@pytest.fixture
def patch_llm_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from fakes import FakeLLMClient

    def _factory(model_config, *, gemini_api_key=None, **_ignored):
        return FakeLLMClient(model_config)

    monkeypatch.setattr("benchmark_agent.agent.agent.create_llm_client", _factory)


@pytest.fixture(autouse=True)
def isolated_llm_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCHMARK_AGENT_GEMINI_BACKEND", "api_key")
    monkeypatch.delenv("BENCHMARK_AGENT_VERTEX_PROJECT", raising=False)
    monkeypatch.delenv("BENCHMARK_AGENT_VERTEX_LOCATION", raising=False)
    monkeypatch.setenv("BENCHMARK_AGENT_SUT__API", "influx")
    monkeypatch.setattr(
        "benchmark_agent.agent.agent.GenerationAgent._sut_reachable", lambda self: False
    )


@pytest.fixture(autouse=True)
def isolated_results_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    raw_dir = tmp_path / "raw"
    monkeypatch.setenv("BENCHMARK_AGENT_PATHS__RESULTS_RAW_DIR", str(raw_dir))
    monkeypatch.setenv("BENCHMARK_AGENT_PATHS__RESULTS_REPORTS_DIR", str(tmp_path / "reports"))
    return raw_dir


@pytest.fixture
def fake_influx() -> Iterator[FakeInfluxServer]:
    server, thread = start_fake_influx()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def proxy(fake_influx: FakeInfluxServer) -> Iterator[tuple[MeasurementProxy, str]]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    instance = MeasurementProxy(
        listen_host="127.0.0.1",
        listen_port=port,
        upstream_url=f"http://127.0.0.1:{fake_influx.port}",
        upstream_timeout_s=10.0,
    )
    instance.start()
    try:
        yield instance, f"http://127.0.0.1:{port}"
    finally:
        instance.stop()
