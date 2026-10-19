from __future__ import annotations

import asyncio
import gzip
import threading
import time
import warnings

from aiohttp import ClientSession, ClientTimeout, web

from benchmark_agent.measurement.facade import (
    API_TESSERA,
    LocalResponse,
    TesseraFacade,
    UpstreamRequest,
)
from benchmark_agent.measurement.kestrel import API_KESTREL, KestrelFacade
from benchmark_agent.measurement.op_mapping import map_path_to_operation
from benchmark_agent.models import OperationWindowStats, ProxyWindowStats, RequestSample

API_INFLUX = "influx"

_OPERATIONS = ("write", "read", "delete")
_BODY_EXCERPT_CHARS = 4000
_BODY_OPERATIONS = ("read", "delete")
_HOP_BY_HOP_REQUEST = {"host", "content-length", "transfer-encoding"}
_HOP_BY_HOP_RESPONSE = {"content-encoding", "transfer-encoding", "content-length"}


class MeasurementProxy:
    def __init__(
        self,
        *,
        listen_host: str,
        listen_port: int,
        upstream_url: str,
        upstream_timeout_s: float | None = None,
        inflight_grace_s: float = 2.0,
        api: str = API_INFLUX,
        org: str | None = None,
        bucket: str | None = None,
    ) -> None:
        self._api = api
        self._facade: TesseraFacade | KestrelFacade | None = None
        if api == API_TESSERA:
            self._facade = TesseraFacade(org=org or "", bucket=bucket or "")
        elif api == API_KESTREL:
            self._facade = KestrelFacade(org=org or "", bucket=bucket or "")
        self._listen_host = listen_host
        self._listen_port = listen_port
        self._upstream_url = upstream_url.rstrip("/")
        self._upstream_timeout_s = upstream_timeout_s
        self._inflight_grace_s = inflight_grace_s
        self._samples: list[RequestSample] = []
        self._window_open = False
        self._window_started_at: float | None = None
        self._in_flight = 0
        self._lock = threading.Lock()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._session: ClientSession | None = None
        self._loop_thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._start_error: BaseException | None = None

    def start(self) -> None:
        if self._loop_thread is not None:
            return
        self._stop.clear()
        self._ready.clear()
        self._start_error = None
        self._loop_thread = threading.Thread(
            target=self._run_server, name="measurement-proxy", daemon=True
        )
        self._loop_thread.start()
        if not self._ready.wait(timeout=10):
            self._stop.set()
            self._loop_thread = None
            raise RuntimeError("Measurement proxy failed to start in time")
        if self._start_error is not None:
            error = self._start_error
            self._loop_thread.join(timeout=5)
            self._loop_thread = None
            raise RuntimeError(f"Measurement proxy failed to start: {error}")

    def stop(self) -> None:
        self._stop.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10)
            if self._loop_thread.is_alive():
                warnings.warn(
                    "measurement proxy thread did not stop within 10 s",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self._loop_thread = None

    def open_window(self) -> None:
        with self._lock:
            self._samples = []
            self._window_open = True
            self._window_started_at = time.perf_counter()

    def close_window(self) -> ProxyWindowStats:
        with self._lock:
            started = self._window_started_at
            self._window_open = False
            self._window_started_at = None
        if started is None:
            return _empty_window()

        deadline = time.monotonic() + self._inflight_grace_s
        while time.monotonic() < deadline:
            with self._lock:
                if self._in_flight == 0:
                    break
            time.sleep(0.01)

        with self._lock:
            ended = time.perf_counter()
            samples = list(self._samples)
            self._samples = []

        duration = max(ended - started, 1e-9)
        by_operation: dict[str, OperationWindowStats] = {}
        for operation in _OPERATIONS:
            operation_samples = [sample for sample in samples if sample.operation == operation]
            by_operation[operation] = _stats_for_samples(operation_samples, duration)
        unmapped = [sample.path for sample in samples if sample.operation is None]
        return ProxyWindowStats(
            window_duration_s=duration,
            by_operation=by_operation,
            unmapped_requests=len(unmapped),
            unmapped_paths=sorted(set(unmapped)),
            requests=samples,
        )

    def _run_server(self) -> None:
        async def _serve() -> None:
            app = web.Application()
            app.router.add_route("*", "/{path:.*}", self._handle)
            self._session = ClientSession(timeout=ClientTimeout(total=self._upstream_timeout_s))
            self._runner = web.AppRunner(app, auto_decompress=False, shutdown_timeout=2.0)
            try:
                await self._runner.setup()
                self._site = web.TCPSite(self._runner, self._listen_host, self._listen_port)
                await self._site.start()
            except BaseException:
                await self._runner.cleanup()
                await self._session.close()
                raise
            self._ready.set()
            while not self._stop.is_set():
                await asyncio.sleep(0.05)
            if self._site is not None:
                await self._site.stop()
            if self._runner is not None:
                await self._runner.cleanup()
            if self._session is not None:
                await self._session.close()

        try:
            asyncio.run(_serve())
        except BaseException as error:
            self._start_error = error
            self._ready.set()

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        assert self._session is not None
        body = await request.read()
        operation: str | None = map_path_to_operation(request.path)
        api = API_INFLUX
        rejection: str | None = None
        upstream_method = request.method
        upstream_url = f"{self._upstream_url}{request.path_qs}"
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _HOP_BY_HOP_REQUEST
        }
        upstream_body = body
        content_encoding = request.headers.get("Content-Encoding")
        translated: UpstreamRequest | None = None
        local: LocalResponse | None = None
        if self._facade is not None:
            outcome = self._facade.handle(
                request.method, request.path, dict(request.query), request.headers, body
            )
            if isinstance(outcome, LocalResponse):
                local = outcome
                operation = outcome.operation
                api = outcome.api
                rejection = outcome.rejection
            else:
                translated = outcome
                operation = outcome.operation
                api = self._api
                upstream_method = outcome.method
                upstream_url = f"{self._upstream_url}{outcome.path_qs}"
                headers = dict(outcome.headers)
                upstream_body = outcome.body
                content_encoding = None

        with self._lock:
            in_window = self._window_open
            self._in_flight += 1
        started = time.perf_counter()
        error = False
        status_code = 502
        response_body = b""
        response_headers: dict[str, str] = {}
        try:
            if local is not None:
                status_code = local.status
                response_body = local.body
                response_headers = {"Content-Type": local.content_type}
            else:
                async with self._session.request(
                    upstream_method,
                    upstream_url,
                    headers=headers,
                    data=upstream_body,
                    allow_redirects=False,
                ) as upstream_response:
                    response_body = await upstream_response.read()
                    status_code = upstream_response.status
                    response_headers = {
                        key: value
                        for key, value in upstream_response.headers.items()
                        if key.lower() not in _HOP_BY_HOP_RESPONSE
                    }
                if translated is not None and self._facade is not None:
                    status_code, response_body, content_type = self._facade.translate_response(
                        translated, status_code, response_body
                    )
                    response_headers = {"Content-Type": content_type}
        except Exception:
            error = True
            response_body = b"upstream proxy error"
            status_code = 502
        finally:
            ended = time.perf_counter()
            measured = local is None
            sample = RequestSample(
                operation=operation,
                method=request.method,
                path=request.path_qs,
                status_code=status_code,
                latency_ms=(ended - started) * 1000.0,
                error=error or status_code >= 400,
                started_at=started,
                ended_at=ended,
                n_points=(
                    count_line_protocol_points(upstream_body, content_encoding)
                    if measured and operation == "write"
                    else 0
                ),
                body_excerpt=(
                    body_excerpt(upstream_body, content_encoding)
                    if measured and operation in _BODY_OPERATIONS
                    else None
                ),
                api=api,
                rejection=rejection,
                upstream_path=translated.path_qs if translated is not None else None,
            )
            with self._lock:
                self._in_flight -= 1
                if in_window:
                    self._samples.append(sample)

        return web.Response(status=status_code, body=response_body, headers=response_headers)


def count_line_protocol_points(body: bytes, content_encoding: str | None) -> int:
    if not body:
        return 0
    if content_encoding and "gzip" in content_encoding.lower():
        try:
            body = gzip.decompress(body)
        except OSError, EOFError:
            return 0
    return sum(
        1 for line in body.splitlines() if line.strip() and not line.lstrip().startswith(b"#")
    )


def body_excerpt(body: bytes, content_encoding: str | None) -> str | None:
    if not body:
        return None
    if content_encoding and "gzip" in content_encoding.lower():
        try:
            body = gzip.decompress(body)
        except OSError, EOFError:
            return None
    return body[: _BODY_EXCERPT_CHARS * 4].decode("utf-8", errors="replace")[:_BODY_EXCERPT_CHARS]


def _empty_window() -> ProxyWindowStats:
    return ProxyWindowStats(
        window_duration_s=0.0,
        by_operation={operation: OperationWindowStats() for operation in _OPERATIONS},
    )


def _stats_for_samples(
    samples: list[RequestSample], window_duration_s: float
) -> OperationWindowStats:
    if not samples:
        return OperationWindowStats(active_duration_s=window_duration_s)
    ok_latencies = [sample.latency_ms for sample in samples if not sample.error]
    error_count = sum(1 for sample in samples if sample.error)
    active = max(
        max(sample.ended_at for sample in samples) - min(sample.started_at for sample in samples),
        1e-9,
    )
    return OperationWindowStats(
        n_requests=len(samples),
        error_count=error_count,
        n_points=sum(sample.n_points for sample in samples if not sample.error),
        throughput_ops_s=len(samples) / active,
        active_duration_s=active,
        latency_p50_ms=_percentile(ok_latencies, 50),
        latency_p95_ms=_percentile(ok_latencies, 95),
        latency_p99_ms=_percentile(ok_latencies, 99),
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (percentile / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight
