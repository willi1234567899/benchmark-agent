from __future__ import annotations

import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HtmlBuilder = Callable[[], str]
PayloadBuilder = Callable[[], dict[str, Any]]


def make_handler(
    build_html: HtmlBuilder, build_payload: PayloadBuilder | None = None
) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "BenchmarkAgentDashboard/0.1"

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            try:
                if path in {"/", "/index.html", "/dashboard.html"}:
                    body = build_html().encode("utf-8")
                    self._send(200, "text/html; charset=utf-8", body)
                elif path == "/data.json" and build_payload is not None:
                    body = json.dumps(build_payload(), ensure_ascii=False).encode("utf-8")
                    self._send(200, "application/json; charset=utf-8", body)
                elif path == "/healthz":
                    self._send(200, "text/plain; charset=utf-8", b"ok")
                else:
                    self._send(404, "text/plain; charset=utf-8", b"not found")
            except Exception as error:
                message = f"dashboard build failed: {type(error).__name__}: {error}"
                self._send(500, "text/plain; charset=utf-8", message.encode("utf-8"))

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return DashboardHandler


def create_server(
    build_html: HtmlBuilder,
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
    build_payload: PayloadBuilder | None = None,
) -> ThreadingHTTPServer:
    handler = make_handler(build_html, build_payload)
    return ThreadingHTTPServer((host, port), handler)


def serve_dashboard(
    build_html: HtmlBuilder,
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
    build_payload: PayloadBuilder | None = None,
) -> None:
    server = create_server(build_html, host=host, port=port, build_payload=build_payload)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
