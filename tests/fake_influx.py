from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from typing import Any
from urllib.parse import urlsplit

_BUCKET_BOUNDS = (
    "0.005",
    "0.01",
    "0.025",
    "0.05",
    "0.1",
    "0.25",
    "0.5",
    "1",
    "2.5",
    "5",
    "10",
    "+Inf",
)


class FakeInfluxHandler(BaseHTTPRequestHandler):
    server: FakeInfluxServer

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/health":
            self._send(200, b'{"name":"influxdb","status":"pass"}', "application/json")
            return
        if path == "/metrics":
            body = self.server.metrics_text().encode("utf-8")
            self._send(200, body, "text/plain; version=0.0.4")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        server = self.server
        path = urlsplit(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        server.record_request(path, dict(self.headers.items()), body)
        if path == "/api/v2/write":
            server.write_count += 1
            self._send(204, b"", "text/plain")
            return
        if path == "/api/v2/query":
            server.query_count += 1
            csv = (
                "#datatype,string,long,dateTime:RFC3339,double\n"
                ",result,table,_time,_value\n"
                ",_result,0,2026-08-20T00:00:00Z,1\n"
            )
            self._send(200, csv.encode("utf-8"), "text/csv")
            return
        if path == "/api/v2/delete":
            server.delete_count += 1
            self._send(204, b"", "application/json")
            return
        self._send(404, b"not found", "text/plain")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)


class FakeInfluxServer(ThreadingHTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), FakeInfluxHandler)
        self.write_count = 0
        self.query_count = 0
        self.delete_count = 0
        self.last_path: str | None = None
        self.last_headers: dict[str, str] = {}
        self.last_body: bytes = b""
        self.requests: list[tuple[str, dict[str, str], bytes]] = []
        self._record_lock = Lock()

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def record_request(self, path: str, headers: dict[str, str], body: bytes) -> None:
        with self._record_lock:
            self.last_path = path
            self.last_headers = headers
            self.last_body = body
            self.requests.append((path, headers, body))

    def metrics_text(self) -> str:
        lines = [
            "# HELP http_api_requests_total Count of HTTP API requests.",
            "# TYPE http_api_requests_total counter",
        ]
        series = (
            ("/api/v2/write", "204", self.write_count),
            ("/api/v2/query", "200", self.query_count),
            ("/api/v2/delete", "204", self.delete_count),
        )
        for path, code, count in series:
            lines.append(f"http_api_requests_total{_labels(path, code)} {count}")
        lines += [
            "# HELP http_api_request_duration_seconds Time taken to respond to HTTP request",
            "# TYPE http_api_request_duration_seconds histogram",
        ]
        for path, code, count in series:
            for bound in _BUCKET_BOUNDS:
                lines.append(
                    f"http_api_request_duration_seconds_bucket{_labels(path, code, le=bound)} "
                    f"{count}"
                )
            lines.append(
                f"http_api_request_duration_seconds_sum{_labels(path, code)} {count * 0.002:.6f}"
            )
            lines.append(f"http_api_request_duration_seconds_count{_labels(path, code)} {count}")
        return "\n".join(lines) + "\n"


def _labels(path: str, code: str, *, le: str | None = None) -> str:
    parts = [
        'handler="platform"',
        'method="POST"',
        f'path="{path}"',
        f'response_code="{code}"',
        f'status="{code[0]}XX"',
        'user_agent="x"',
    ]
    if le is not None:
        parts.append(f'le="{le}"')
    return "{" + ",".join(parts) + "}"


def start_fake_influx() -> tuple[FakeInfluxServer, Thread]:
    server = FakeInfluxServer()
    thread = Thread(target=server.serve_forever, name="fake-influx", daemon=True)
    thread.start()
    return server, thread
