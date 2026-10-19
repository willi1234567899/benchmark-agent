from __future__ import annotations

import csv
import gzip
import io
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from benchmark_agent.measurement.op_mapping import map_path_to_operation

API_PREFIX = "/tessera/v1"
KEY_HEADER = "X-Tessera-Key"
TS_UNITS: tuple[str, ...] = ("ns", "us", "ms", "s")
API_TESSERA = "tessera"
API_LEGACY = "legacy"
API_UNKNOWN = "unknown"

FACADE_OPERATIONS: dict[str, str] = {
    f"{API_PREFIX}/ingest": "write",
    f"{API_PREFIX}/fetch": "read",
    f"{API_PREFIX}/purge": "delete",
}
_INFLUX_QUERY_PARAMS: dict[str, str] = {
    "org": "space",
    "bucket": "store",
    "precision": "ts_unit",
    "db": "store",
}
_FORBIDDEN_BODY_FIELDS: dict[str, dict[str, str]] = {
    "write": {"lines": "records", "points": "records", "line_protocol": "records"},
    "read": {
        "query": "series + window",
        "flux": "series + window",
        "limit": "max_rows",
        "n": "max_rows",
        "bucket": "store",
        "measurement": "series",
        "range": "window",
        "start": "window.from",
        "stop": "window.to",
    },
    "delete": {
        "predicate": "series (+ labels)",
        "start": "from",
        "stop": "until",
        "measurement": "series",
        "bucket": "store",
    },
}
_ALLOWED_BODY_FIELDS: dict[str, set[str]] = {
    "write": {"records"},
    "read": {"store", "series", "window", "max_rows", "labels"},
    "delete": {"series", "from", "until", "labels"},
}
_DURATION_RE = re.compile(r"^-\d+(ns|us|ms|s|m|h|d|w)$")
_RFC3339_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")
_LEGACY_PATHS = ("/api/", "/health", "/ping", "/metrics", "/write", "/query", "/ready")


@dataclass(frozen=True)
class UpstreamRequest:
    operation: str | None
    method: str
    path_qs: str
    headers: dict[str, str]
    body: bytes
    n_records: int = 0
    series: str | None = None


@dataclass(frozen=True)
class LocalResponse:
    status: int
    body: bytes
    content_type: str = "application/json"
    operation: str | None = None
    rejection: str | None = None
    api: str = API_TESSERA


class TesseraFacade:
    def __init__(self, *, org: str, bucket: str) -> None:
        self._org = org
        self._bucket = bucket

    def handle(
        self,
        method: str,
        path: str,
        query: Mapping[str, str],
        headers: Mapping[str, str],
        body: bytes,
    ) -> UpstreamRequest | LocalResponse:
        method = method.upper()
        path = path.rstrip("/") or "/"
        if path in ("/", API_PREFIX):
            if method not in ("GET", "HEAD"):
                return _reject(405, "method_not_allowed", f"{method} is not allowed on {path}")
            return LocalResponse(status=200, body=_json(index_document()))
        if path == f"{API_PREFIX}/health":
            if method not in ("GET", "HEAD"):
                return _reject(405, "method_not_allowed", f"{method} is not allowed on {path}")
            return UpstreamRequest(
                operation=None, method="GET", path_qs="/health", headers={}, body=b""
            )

        operation = FACADE_OPERATIONS.get(path)
        if operation is None:
            return _unknown_endpoint(path)
        if method != "POST":
            return _reject(
                405, "method_not_allowed", f"{method} is not allowed on {path}; use POST", operation
            )

        key = _header(headers, KEY_HEADER)
        if not key:
            if _header(headers, "Authorization"):
                return _reject(
                    401,
                    "influx_auth_header",
                    "the Authorization header is not supported; send the API key in the "
                    f"{KEY_HEADER} header",
                    operation,
                )
            return _reject(401, "missing_key", f"missing {KEY_HEADER} header", operation)

        for influx_name, tessera_name in _INFLUX_QUERY_PARAMS.items():
            if influx_name in query:
                return _reject(
                    400,
                    f"influx_param_{influx_name}",
                    f"unknown query parameter '{influx_name}'; Tessera uses '{tessera_name}'",
                    operation,
                )
        space = query.get("space")
        if not space:
            return _reject(400, "missing_space", "query parameter 'space' is required", operation)

        if "gzip" in (_header(headers, "Content-Encoding") or "").lower():
            try:
                body = gzip.decompress(body)
            except OSError, EOFError:
                return _reject(
                    400, "invalid_gzip_body", "the gzip body could not be decoded", operation
                )
        payload = _parse_json_object(body)
        if payload is None:
            return _reject(
                400,
                "non_json_body",
                "the request body must be a JSON object (see GET /tessera/v1)",
                operation,
            )
        for field_name in payload:
            if field_name in _FORBIDDEN_BODY_FIELDS[operation]:
                hint = _FORBIDDEN_BODY_FIELDS[operation][field_name]
                return _reject(
                    400,
                    f"influx_field_{field_name}",
                    f"unknown field '{field_name}' in request body; Tessera uses '{hint}'",
                    operation,
                )
            if field_name not in _ALLOWED_BODY_FIELDS[operation]:
                return _reject(
                    400,
                    f"unknown_field_{field_name}",
                    f"unknown field '{field_name}' in request body; allowed: "
                    + ", ".join(sorted(_ALLOWED_BODY_FIELDS[operation])),
                    operation,
                )

        auth = {"Authorization": f"Token {key}"}
        if operation == "write":
            return self._ingest(query, payload, auth, space)
        if operation == "read":
            return self._fetch(payload, auth, space)
        return self._purge(query, payload, auth, space)

    def _ingest(
        self, query: Mapping[str, str], payload: dict[str, Any], auth: dict[str, str], space: str
    ) -> UpstreamRequest | LocalResponse:
        store = query.get("store")
        if not store:
            return _reject(400, "missing_store", "query parameter 'store' is required", "write")
        ts_unit = query.get("ts_unit", "ns")
        if ts_unit not in TS_UNITS:
            return _reject(
                400,
                "invalid_ts_unit",
                f"ts_unit must be one of {', '.join(TS_UNITS)}",
                "write",
            )
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            return _reject(
                400, "missing_field_records", "'records' must be a non-empty array", "write"
            )
        lines: list[str] = []
        for index, record in enumerate(records):
            try:
                lines.append(_record_to_line(record))
            except ValueError as error:
                return _reject(400, "invalid_record", f"records[{index}]: {error}", "write")
        body = ("\n".join(lines) + "\n").encode("utf-8")
        params = urlencode({"org": space, "bucket": store, "precision": ts_unit})
        return UpstreamRequest(
            operation="write",
            method="POST",
            path_qs=f"/api/v2/write?{params}",
            headers={**auth, "Content-Type": "text/plain; charset=utf-8"},
            body=body,
            n_records=len(lines),
        )

    def _fetch(
        self, payload: dict[str, Any], auth: dict[str, str], space: str
    ) -> UpstreamRequest | LocalResponse:
        store = payload.get("store")
        series = payload.get("series")
        if not isinstance(store, str) or not store:
            return _reject(400, "missing_field_store", "'store' (string) is required", "read")
        if not isinstance(series, str) or not series:
            return _reject(400, "missing_field_series", "'series' (string) is required", "read")
        window = payload.get("window")
        if not isinstance(window, dict) or not isinstance(window.get("from"), str):
            return _reject(
                400,
                "missing_field_window",
                "'window' must be an object with 'from' (duration like -1h or RFC3339 time)",
                "read",
            )
        start = _time_literal(window["from"])
        if start is None:
            return _reject(400, "invalid_window_from", "'window.from' is not a valid time", "read")
        stop = None
        if "to" in window:
            stop = _time_literal(str(window["to"]))
            if stop is None:
                return _reject(400, "invalid_window_to", "'window.to' is not a valid time", "read")
        max_rows = payload.get("max_rows")
        if max_rows is not None and (
            isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1
        ):
            return _reject(400, "invalid_max_rows", "'max_rows' must be a positive integer", "read")
        labels = payload.get("labels") or {}
        if not isinstance(labels, dict):
            return _reject(400, "invalid_labels", "'labels' must be an object", "read")

        flux = f'from(bucket: "{_flux_string(store)}") |> range(start: {start}'
        flux += f", stop: {stop})" if stop else ")"
        flux += f' |> filter(fn: (r) => r._measurement == "{_flux_string(series)}")'
        for key, value in labels.items():
            label = _flux_string(str(key))
            wanted = _flux_string(str(value))
            flux += f' |> filter(fn: (r) => r["{label}"] == "{wanted}")'
        flux += " |> group()"
        if max_rows is not None:
            flux += f" |> limit(n: {int(max_rows)})"
        body = json.dumps(
            {
                "query": flux,
                "type": "flux",
                "dialect": {"header": True, "annotations": [], "delimiter": ","},
            }
        ).encode("utf-8")
        params = urlencode({"org": space})
        return UpstreamRequest(
            operation="read",
            method="POST",
            path_qs=f"/api/v2/query?{params}",
            headers={**auth, "Content-Type": "application/json", "Accept": "application/csv"},
            body=body,
            series=series,
        )

    def _purge(
        self, query: Mapping[str, str], payload: dict[str, Any], auth: dict[str, str], space: str
    ) -> UpstreamRequest | LocalResponse:
        store = query.get("store")
        if not store:
            return _reject(400, "missing_store", "query parameter 'store' is required", "delete")
        series = payload.get("series")
        if not isinstance(series, str) or not series:
            return _reject(400, "missing_field_series", "'series' (string) is required", "delete")
        start = payload.get("from")
        until = payload.get("until")
        if not isinstance(start, str) or not _RFC3339_RE.match(start):
            return _reject(400, "invalid_from", "'from' must be an RFC3339 time", "delete")
        if not isinstance(until, str) or not _RFC3339_RE.match(until):
            return _reject(400, "invalid_until", "'until' must be an RFC3339 time", "delete")
        labels = payload.get("labels") or {}
        if not isinstance(labels, dict):
            return _reject(400, "invalid_labels", "'labels' must be an object", "delete")
        predicate = f'_measurement="{series}"'
        for key, value in labels.items():
            predicate += f' AND {key}="{value}"'
        body = json.dumps({"start": start, "stop": until, "predicate": predicate}).encode("utf-8")
        params = urlencode({"org": space, "bucket": store})
        return UpstreamRequest(
            operation="delete",
            method="POST",
            path_qs=f"/api/v2/delete?{params}",
            headers={**auth, "Content-Type": "application/json"},
            body=body,
            series=series,
        )

    def translate_response(
        self, upstream: UpstreamRequest, status: int, body: bytes
    ) -> tuple[int, bytes, str]:
        if upstream.operation is None:
            if status == 200:
                return 200, _json({"status": "ready"}), "application/json"
            return (
                503,
                _json({"status": "unavailable", "upstream_status": status}),
                "application/json",
            )
        if status >= 400:
            return (
                status,
                _json({"error": _text(body), "upstream_status": status}),
                "application/json",
            )
        if upstream.operation == "write":
            return 200, _json({"accepted": upstream.n_records}), "application/json"
        if upstream.operation == "delete":
            return 200, _json({"purged": True, "series": upstream.series}), "application/json"
        rows = parse_csv_rows(_text(body))
        return (
            200,
            _json({"series": upstream.series, "row_count": len(rows), "rows": rows}),
            "application/json",
        )


def index_document() -> dict[str, Any]:
    return {
        "service": "Tessera time-series store",
        "api": "v1",
        "base_path": API_PREFIX,
        "auth": {"header": KEY_HEADER, "description": "API key, required on every data endpoint"},
        "endpoints": [
            {"method": "GET", "path": f"{API_PREFIX}/health", "summary": "readiness probe"},
            {
                "method": "POST",
                "path": f"{API_PREFIX}/ingest",
                "query": ["space", "store", "ts_unit (ns|us|ms|s, default ns)"],
                "body": "JSON object",
                "summary": "append records to a series; 200 with {accepted: n}",
            },
            {
                "method": "POST",
                "path": f"{API_PREFIX}/fetch",
                "query": ["space"],
                "body": "JSON object",
                "summary": "read rows of a series inside a time window; 200 with {rows, row_count}",
            },
            {
                "method": "POST",
                "path": f"{API_PREFIX}/purge",
                "query": ["space", "store"],
                "body": "JSON object",
                "summary": "remove the records of a series inside a time range; 200 with {purged}",
            },
        ],
        "errors": "4xx replies carry {error: <message>} naming the expected parameter or field",
    }


_COLUMN_NAMES: dict[str, str | None] = {
    "": None,
    "result": None,
    "table": None,
    "_start": None,
    "_stop": None,
    "_time": "ts",
    "_value": "value",
    "_field": "metric",
    "_measurement": "series",
}


def parse_csv_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    header: list[str] | None = None
    for cells in csv.reader(io.StringIO(text)):
        if not cells or all(cell == "" for cell in cells):
            header = None
            continue
        if cells[0].startswith("#"):
            continue
        if header is None:
            header = cells
            continue
        row: dict[str, Any] = {}
        for column, value in zip(header, cells, strict=False):
            name = _COLUMN_NAMES.get(column, column)
            if name is None:
                continue
            row[name] = _float_or_text(value) if name == "value" else value
        rows.append(row)
    return rows


def _record_to_line(record: object) -> str:
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    series = record.get("series")
    if not isinstance(series, str) or not series:
        raise ValueError("'series' (string) is required")
    values = record.get("values")
    if not isinstance(values, dict) or not values:
        raise ValueError("'values' must be a non-empty object")
    labels = record.get("labels") or {}
    if not isinstance(labels, dict):
        raise ValueError("'labels' must be an object")
    unknown = set(record) - {"series", "labels", "values", "ts"}
    if unknown:
        raise ValueError(f"unknown record field(s) {sorted(unknown)}")
    line = _escape_key(series)
    for key, value in labels.items():
        line += f",{_escape_key(str(key))}={_escape_key(str(value))}"
    fields: list[str] = []
    for key, value in values.items():
        fields.append(f"{_escape_key(str(key))}={_field_value(value)}")
    line += " " + ",".join(fields)
    timestamp = record.get("ts")
    if timestamp is not None:
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise ValueError("'ts' must be an integer timestamp in the requested ts_unit")
        line += f" {timestamp}"
    return line


def _field_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(float(value))
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    raise ValueError("values must be numbers, booleans or strings")


def _escape_key(text: str) -> str:
    return text.replace("\\", "\\\\").replace(",", "\\,").replace(" ", "\\ ").replace("=", "\\=")


def _flux_string(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _time_literal(text: str) -> str | None:
    text = text.strip()
    if _DURATION_RE.match(text):
        return text
    if _RFC3339_RE.match(text):
        return text
    if text == "now()":
        return "now()"
    return None


def _parse_json_object(body: bytes) -> dict[str, Any] | None:
    if not body:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except ValueError, UnicodeDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _header(headers: Mapping[str, str], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        if key.lower() == wanted:
            return value
    return None


_WIRE_CODES: tuple[tuple[str, str], ...] = (
    ("influx_param_", "unsupported_parameter"),
    ("influx_field_", "unknown_field"),
    ("influx_auth_header", "unsupported_auth_header"),
    ("legacy_endpoint", "unknown_endpoint"),
    ("foreign_param_", "unsupported_parameter"),
    ("foreign_field_", "unknown_field"),
    ("foreign_auth_header", "unsupported_auth_header"),
    ("foreign_endpoint", "unknown_endpoint"),
)


def wire_code(internal: str) -> str:
    for prefix, public in _WIRE_CODES:
        if internal.startswith(prefix):
            return public
    return internal


def _reject(status: int, code: str, message: str, operation: str | None = None) -> LocalResponse:
    return LocalResponse(
        status=status,
        body=_json({"error": message, "code": wire_code(code)}),
        operation=operation,
        rejection=code,
    )


def _unknown_endpoint(path: str) -> LocalResponse:
    legacy = path.startswith(_LEGACY_PATHS)
    return LocalResponse(
        status=404,
        body=_json(
            {
                "error": (
                    "unknown endpoint; this service speaks the Tessera API v1 — "
                    f"GET {API_PREFIX} lists the endpoints"
                ),
                "code": "unknown_endpoint",
            }
        ),
        operation=map_path_to_operation(path),
        rejection="legacy_endpoint" if legacy else "unknown_endpoint",
        api=API_LEGACY if legacy else API_UNKNOWN,
    )


def _json(payload: object) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _text(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def _float_or_text(value: str) -> Any:
    try:
        return float(value)
    except ValueError:
        return value
