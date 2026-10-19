from __future__ import annotations

import gzip
import socket

import httpx
import pytest
from fake_influx import FakeInfluxServer

from benchmark_agent.measurement.proxy import MeasurementProxy, count_line_protocol_points
from benchmark_agent.measurement.server_metrics import (
    ServerMetricsCollector,
    _parse_prometheus_text,
)
from benchmark_agent.models import MetricsSnapshot


def test_proxy_forwards_gzip_body_untouched(
    proxy: tuple[MeasurementProxy, str], fake_influx: FakeInfluxServer
) -> None:
    instance, base = proxy
    raw = b"benchmark,source=t value=1i 1\nbenchmark,source=t value=2i 2\n"
    compressed = gzip.compress(raw)
    instance.open_window()
    response = httpx.post(
        f"{base}/api/v2/write?org=o&bucket=b",
        content=compressed,
        headers={"Content-Encoding": "gzip", "Content-Type": "text/plain"},
    )
    stats = instance.close_window()
    assert response.status_code == 204
    assert fake_influx.last_body[:2] == b"\x1f\x8b"
    assert fake_influx.last_body == compressed
    assert fake_influx.last_headers.get("Content-Encoding") == "gzip"
    assert stats.by_operation["write"].n_points == 2


def test_proxy_percentiles_exclude_error_samples(
    proxy: tuple[MeasurementProxy, str], fake_influx: FakeInfluxServer
) -> None:
    instance, base = proxy
    instance.open_window()
    for _ in range(3):
        httpx.post(f"{base}/api/v2/write", content=b"m v=1i 1\n")
    httpx.post(f"{base}/api/v2/write/unknown", content=b"m v=1i 1\n")
    httpx.get(f"{base}/health")
    stats = instance.close_window()
    write = stats.by_operation["write"]
    assert write.n_requests == 4
    assert write.error_count == 1
    assert write.n_points == 3
    assert write.latency_p50_ms is not None and write.latency_p99_ms is not None
    assert write.active_duration_s > 0
    assert write.throughput_ops_s == pytest.approx(4 / write.active_duration_s)
    assert stats.unmapped_requests == 1
    assert stats.unmapped_paths == ["/health"]

    instance.open_window()
    httpx.post(f"{base}/api/v2/delete/nope", content=b"")
    stats = instance.close_window()
    assert stats.by_operation["delete"].n_requests == 1
    assert stats.by_operation["delete"].error_count == 1
    assert stats.by_operation["delete"].latency_p50_ms is None


def test_proxy_close_window_is_idempotent(proxy: tuple[MeasurementProxy, str]) -> None:
    instance, base = proxy
    httpx.post(f"{base}/api/v2/write", content=b"m v=1i 1\n")
    instance.open_window()
    httpx.post(f"{base}/api/v2/write", content=b"m v=1i 1\n")
    first = instance.close_window()
    second = instance.close_window()
    assert first.by_operation["write"].n_requests == 1
    assert second.by_operation["write"].n_requests == 0
    assert second.window_duration_s == 0.0


def test_proxy_start_failure_resets_thread(fake_influx: FakeInfluxServer) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        port = int(blocker.getsockname()[1])
        instance = MeasurementProxy(
            listen_host="127.0.0.1",
            listen_port=port,
            upstream_url=f"http://127.0.0.1:{fake_influx.port}",
        )
        with pytest.raises(RuntimeError, match="failed to start"):
            instance.start()
        assert instance._loop_thread is None
    instance.stop()


def test_count_line_protocol_points_handles_comments_and_gzip() -> None:
    body = b"# comment\nm v=1i 1\n\nm v=2i 2\nm v=3i 3"
    assert count_line_protocol_points(body, None) == 3
    assert count_line_protocol_points(gzip.compress(body), "gzip") == 3
    assert count_line_protocol_points(b"", None) == 0
    assert count_line_protocol_points(b"not gzip", "gzip") == 0


def _snapshot(text: str) -> MetricsSnapshot:
    return MetricsSnapshot(samples=_parse_prometheus_text(text))


def _series(path: str, code: str, count: float, buckets: dict[str, float]) -> str:
    labels = (
        f'handler="platform",method="POST",path="{path}",response_code="{code}",'
        f'status="{code[0]}XX",user_agent="x, y"'
    )
    lines = [f"http_api_requests_total{{{labels}}} {count}"]
    for le, cumulative in buckets.items():
        lines.append(f'http_api_request_duration_seconds_bucket{{{labels},le="{le}"}} {cumulative}')
    return "\n".join(lines)


def test_server_metrics_error_count_and_histogram_percentiles() -> None:
    start = _snapshot(
        "\n".join(
            [
                _series("/api/v2/write", "204", 10, {"0.005": 8, "0.01": 10, "+Inf": 10}),
                _series("/api/v2/write", "400", 1, {"0.005": 1, "0.01": 1, "+Inf": 1}),
            ]
        )
    )
    end = _snapshot(
        "\n".join(
            [
                _series("/api/v2/write", "204", 20, {"0.005": 13, "0.01": 19, "+Inf": 20}),
                _series("/api/v2/write", "400", 3, {"0.005": 3, "0.01": 3, "+Inf": 3}),
                _series("/api/v2/write", "500", 1, {"0.005": 1, "0.01": 1, "+Inf": 1}),
            ]
        )
    )
    delta = ServerMetricsCollector().delta(start, end)
    write = delta.by_operation["write"]
    assert write.n_requests == 13
    assert write.error_count == 3
    assert write.latency_p50_ms == pytest.approx(6.5 / 8 * 5.0)
    assert write.latency_p95_ms == pytest.approx(10.0)
    assert write.latency_p99_ms == pytest.approx(10.0)
    assert delta.by_operation["read"].n_requests == 0
    assert delta.by_operation["read"].latency_p50_ms is None


def test_server_metrics_without_buckets_only_fills_p50_from_mean() -> None:
    start = _snapshot(
        'http_api_requests_total{path="/api/v2/query",status="2XX"} 4\n'
        'http_api_request_duration_seconds_sum{path="/api/v2/query"} 1.0\n'
        'http_api_request_duration_seconds_count{path="/api/v2/query"} 4\n'
    )
    end = _snapshot(
        'http_api_requests_total{path="/api/v2/query",status="2XX"} 8\n'
        'http_api_request_duration_seconds_sum{path="/api/v2/query"} 1.4\n'
        'http_api_request_duration_seconds_count{path="/api/v2/query"} 8\n'
    )
    read = ServerMetricsCollector().delta(start, end).by_operation["read"]
    assert read.n_requests == 4
    assert read.error_count == 0
    assert read.latency_p50_ms == pytest.approx(100.0)
    assert read.latency_p95_ms is None
    assert read.latency_p99_ms is None


def test_fake_influx_metrics_parse_like_real_influx(fake_influx: FakeInfluxServer) -> None:
    collector = ServerMetricsCollector()
    base = f"http://127.0.0.1:{fake_influx.port}"
    start = collector.snapshot(f"{base}/metrics")
    httpx.post(f"{base}/api/v2/write", content=b"m v=1i 1\n")
    httpx.post(f"{base}/api/v2/query", content=b'from(bucket:"b")')
    end = collector.snapshot(f"{base}/metrics")
    delta = collector.delta(start, end)
    assert delta.by_operation["write"].n_requests == 1
    assert delta.by_operation["read"].n_requests == 1
    assert delta.by_operation["write"].latency_p99_ms == pytest.approx(4.95)
