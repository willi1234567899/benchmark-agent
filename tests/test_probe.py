from __future__ import annotations

import pytest

from benchmark_agent.agent.tools import SutProbeToolkit


def test_probe_toolkit_rejects_absolute_urls() -> None:
    toolkit = SutProbeToolkit(base_url="http://127.0.0.1:8086", max_calls=2)
    result = toolkit.call(
        "http_request",
        {"method": "GET", "path": "http://example.com/secret"},
    )
    assert "Absolute URLs" in result
    assert toolkit.remaining_calls == 1


def test_probe_toolkit_enforces_budget() -> None:
    toolkit = SutProbeToolkit(base_url="http://127.0.0.1:9", max_calls=1)
    first = toolkit.call("sut_health")
    assert "failed" in first.lower() or "HTTP" in first
    second = toolkit.call("sut_health")
    assert "budget exhausted" in second


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("DELETE", "/api/v2/delete"),
        ("PUT", "/api/v2/buckets/x"),
        ("PATCH", "/api/v2/orgs/x"),
        ("POST", "/api/v2/write?org=benchmark&bucket=benchmark"),
        ("POST", "/api/v2/delete"),
    ],
)
def test_probe_toolkit_rejects_mutating_requests(method: str, path: str) -> None:
    toolkit = SutProbeToolkit(base_url="http://127.0.0.1:9", max_calls=5)

    result = toolkit.call("http_request", {"method": method, "path": path, "body": "m v=1"})

    assert result.startswith("Rejected")
    assert toolkit.remaining_calls == 4


@pytest.mark.parametrize(
    "body",
    [
        'from(bucket:"b") |> range(start:-1h) |> to(bucket:"other")',
        '{"query": "from(bucket:\\"b\\") |> TO (bucket:\\"x\\")"}',
    ],
)
def test_probe_toolkit_rejects_flux_to_in_query(body: str) -> None:
    toolkit = SutProbeToolkit(base_url="http://127.0.0.1:9", max_calls=5)

    result = toolkit.call(
        "http_request", {"method": "POST", "path": "/api/v2/query?org=b", "body": body}
    )

    assert result.startswith("Rejected")
    assert "to()" in result


def test_probe_toolkit_injects_auth_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Response:
        status_code = 200
        text = "ok"

    def _fake_request(method, url, *, content, headers, timeout):
        captured.update({"method": method, "url": url, "headers": headers})
        return _Response()

    monkeypatch.setattr("benchmark_agent.agent.tools.httpx.request", _fake_request)
    toolkit = SutProbeToolkit(base_url="http://127.0.0.1:8087", max_calls=2, auth_token="secret")

    result = toolkit.call(
        "http_request",
        {"method": "POST", "path": "/api/v2/query?org=benchmark", "headers": {"X-Extra": "1"}},
    )

    assert "HTTP 200" in result
    assert captured["method"] == "POST"
    assert captured["headers"] == {"Authorization": "Token secret", "X-Extra": "1"}
