from __future__ import annotations

import gzip
import json
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import unquote, urlencode

from benchmark_agent.measurement.facade import (
    API_LEGACY,
    API_UNKNOWN,
    LocalResponse,
    UpstreamRequest,
    _escape_key,
    _field_value,
    _flux_string,
    _header,
    _json,
    _text,
    parse_csv_rows,
    wire_code,
)
from benchmark_agent.measurement.op_mapping import map_path_to_operation

API_KESTREL = "kestrel"
API_PREFIX = "/v1"
TOKEN_HEADER = "X-Api-Token"
UNITS: tuple[str, ...] = ("ns", "us", "ms", "s")
_POINTS_RE = re.compile(r"^/v1/tenants/([^/]+)/stores/([^/]+)/series/([^/]+)/points$")
_RFC3339_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")
_ISO_DURATION_RE = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")
_LEGACY_PATHS = ("/api/", "/health", "/ping", "/metrics", "/write", "/query", "/ready")
_METHOD_OPERATION = {"PUT": "write", "GET": "read", "DELETE": "delete"}
_FOREIGN_PARAMS: dict[str, tuple[str, str]] = {
    "precision": ("unit", "influx"),
    "org": ("the tenant path segment", "influx"),
    "orgID": ("the tenant path segment", "influx"),
    "bucket": ("the store path segment", "influx"),
    "start": ("since (read) or from (delete)", "influx"),
    "stop": ("until (read) or to (delete)", "influx"),
    "predicate": ("label.<name> parameters", "influx"),
    "ts_unit": ("unit", "foreign"),
    "space": ("the tenant path segment", "foreign"),
    "store": ("the store path segment", "foreign"),
    "max_rows": ("limit", "foreign"),
}
_FOREIGN_FIELDS: dict[str, tuple[str, str]] = {
    "lines": ("columns + rows", "influx"),
    "line_protocol": ("columns + rows", "influx"),
    "records": ("columns + rows", "foreign"),
}
_ALLOWED_WRITE_FIELDS = {"columns", "rows", "labels"}


class KestrelFacade:
    api = API_KESTREL

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
                return self._reject(405, "method_not_allowed", f"{method} is not allowed on {path}")
            return LocalResponse(status=200, body=_json(index_document()), api=API_KESTREL)
        if path == f"{API_PREFIX}/status":
            if method not in ("GET", "HEAD"):
                return self._reject(405, "method_not_allowed", f"{method} is not allowed on {path}")
            return UpstreamRequest(
                operation=None, method="GET", path_qs="/health", headers={}, body=b""
            )
        match = _POINTS_RE.match(path)
        if match is None:
            return _unknown_endpoint(path)
        tenant, store, series = (unquote(part) for part in match.groups())
        operation = _METHOD_OPERATION.get(method)
        if operation is None:
            return self._reject(
                405,
                "method_not_allowed",
                "use PUT to write points, GET to read them and DELETE to remove them",
                "write" if method == "POST" else None,
            )

        token = _header(headers, TOKEN_HEADER)
        if not token:
            if _header(headers, "Authorization"):
                return self._reject(
                    401,
                    "influx_auth_header",
                    f"the Authorization header is not supported; send the token in {TOKEN_HEADER}",
                    operation,
                )
            if _header(headers, "X-Tessera-Key"):
                return self._reject(
                    401,
                    "foreign_auth_header",
                    f"unknown authentication header; send the token in {TOKEN_HEADER}",
                    operation,
                )
            return self._reject(401, "missing_token", f"missing {TOKEN_HEADER} header", operation)

        for name, (hint, origin) in _FOREIGN_PARAMS.items():
            if name in query:
                return self._reject(
                    400,
                    f"{origin}_param_{name}",
                    f"unknown query parameter '{name}'; Kestrel uses {hint}",
                    operation,
                )
        auth = {"Authorization": f"Token {token}"}
        if operation == "write":
            return self._write(query, headers, body, auth, tenant, store, series)
        if operation == "read":
            return self._read(query, auth, tenant, store, series)
        return self._delete(query, auth, tenant, store, series)

    def _write(
        self,
        query: Mapping[str, str],
        headers: Mapping[str, str],
        body: bytes,
        auth: dict[str, str],
        tenant: str,
        store: str,
        series: str,
    ) -> UpstreamRequest | LocalResponse:
        unit = query.get("unit", "ns")
        if unit not in UNITS:
            return self._reject(
                400, "invalid_unit", f"unit must be one of {', '.join(UNITS)}", "write"
            )
        if "gzip" in (_header(headers, "Content-Encoding") or "").lower():
            try:
                body = gzip.decompress(body)
            except OSError, EOFError:
                return self._reject(
                    400, "invalid_gzip_body", "the gzip body could not be decoded", "write"
                )
        payload = _parse_json_object(body)
        if payload is None:
            return self._reject(
                400,
                "non_json_body",
                "the request body must be a JSON object (see GET /v1)",
                "write",
            )
        for field_name in payload:
            if field_name in _FOREIGN_FIELDS:
                hint, origin = _FOREIGN_FIELDS[field_name]
                return self._reject(
                    400,
                    f"{origin}_field_{field_name}",
                    f"unknown field '{field_name}' in request body; Kestrel uses {hint}",
                    "write",
                )
            if field_name not in _ALLOWED_WRITE_FIELDS:
                return self._reject(
                    400,
                    f"unknown_field_{field_name}",
                    f"unknown field '{field_name}' in request body; allowed: columns, rows, labels",
                    "write",
                )
        columns = payload.get("columns")
        rows = payload.get("rows")
        labels = payload.get("labels") or []
        if (
            not isinstance(columns, list)
            or not columns
            or not all(isinstance(column, str) for column in columns)
        ):
            return self._reject(
                400, "missing_field_columns", "'columns' must be a list of names", "write"
            )
        if "t" not in columns:
            return self._reject(
                400, "missing_time_column", "'columns' must contain the time column 't'", "write"
            )
        if not isinstance(rows, list) or not rows:
            return self._reject(
                400, "missing_field_rows", "'rows' must be a non-empty list", "write"
            )
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            return self._reject(
                400, "invalid_labels", "'labels' must be a list of column names", "write"
            )
        unknown_labels = [label for label in labels if label not in columns]
        if unknown_labels:
            return self._reject(
                400,
                "invalid_labels",
                f"label columns {unknown_labels} are not in 'columns'",
                "write",
            )
        metric_columns = [column for column in columns if column != "t" and column not in labels]
        if not metric_columns:
            return self._reject(
                400, "missing_metric_column", "at least one metric column is required", "write"
            )
        lines: list[str] = []
        for index, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != len(columns):
                return self._reject(
                    400, "invalid_row", f"rows[{index}] must have {len(columns)} values", "write"
                )
            values = dict(zip(columns, row, strict=True))
            timestamp = values["t"]
            if isinstance(timestamp, bool) or not isinstance(timestamp, int):
                return self._reject(
                    400, "invalid_row", f"rows[{index}]: 't' must be an integer", "write"
                )
            line = _escape_key(series)
            for label in labels:
                line += f",{_escape_key(label)}={_escape_key(str(values[label]))}"
            try:
                fields = ",".join(
                    f"{_escape_key(name)}={_field_value(values[name])}" for name in metric_columns
                )
            except ValueError as error:
                return self._reject(400, "invalid_row", f"rows[{index}]: {error}", "write")
            lines.append(f"{line} {fields} {timestamp}")
        params = urlencode({"org": tenant, "bucket": store, "precision": unit})
        return UpstreamRequest(
            operation="write",
            method="POST",
            path_qs=f"/api/v2/write?{params}",
            headers={**auth, "Content-Type": "text/plain; charset=utf-8"},
            body=("\n".join(lines) + "\n").encode("utf-8"),
            n_records=len(lines),
            series=series,
        )

    def _read(
        self, query: Mapping[str, str], auth: dict[str, str], tenant: str, store: str, series: str
    ) -> UpstreamRequest | LocalResponse:
        since = query.get("since")
        if not since:
            return self._reject(
                400,
                "missing_since",
                "query parameter 'since' is required (ISO-8601 duration like PT2H or RFC3339 time)",
                "read",
            )
        start = _time_literal(since)
        if start is None:
            return self._reject(
                400, "invalid_since", "'since' is not a valid duration or time", "read"
            )
        stop = None
        if "until" in query:
            stop = _time_literal(query["until"])
            if stop is None:
                return self._reject(
                    400, "invalid_until", "'until' is not a valid duration or time", "read"
                )
        limit: int | None = None
        if "limit" in query:
            if not query["limit"].isdigit() or int(query["limit"]) < 1:
                return self._reject(
                    400, "invalid_limit", "'limit' must be a positive integer", "read"
                )
            limit = int(query["limit"])
        labels = _label_params(query)
        flux = f'from(bucket: "{_flux_string(store)}") |> range(start: {start}'
        flux += f", stop: {stop})" if stop else ")"
        flux += f' |> filter(fn: (r) => r._measurement == "{_flux_string(series)}")'
        for key, value in labels.items():
            flux += f' |> filter(fn: (r) => r["{_flux_string(key)}"] == "{_flux_string(value)}")'
        flux += " |> group()"
        if limit is not None:
            flux += f" |> limit(n: {limit})"
        body = json.dumps(
            {
                "query": flux,
                "type": "flux",
                "dialect": {"header": True, "annotations": [], "delimiter": ","},
            }
        ).encode("utf-8")
        return UpstreamRequest(
            operation="read",
            method="POST",
            path_qs=f"/api/v2/query?{urlencode({'org': tenant})}",
            headers={**auth, "Content-Type": "application/json", "Accept": "application/csv"},
            body=body,
            series=series,
        )

    def _delete(
        self, query: Mapping[str, str], auth: dict[str, str], tenant: str, store: str, series: str
    ) -> UpstreamRequest | LocalResponse:
        start = query.get("from")
        stop = query.get("to")
        if not start or not _RFC3339_RE.match(start):
            return self._reject(400, "invalid_from", "'from' must be an RFC3339 time", "delete")
        if not stop or not _RFC3339_RE.match(stop):
            return self._reject(400, "invalid_to", "'to' must be an RFC3339 time", "delete")
        predicate = f'_measurement="{series}"'
        for key, value in _label_params(query).items():
            predicate += f' AND {key}="{value}"'
        body = json.dumps({"start": start, "stop": stop, "predicate": predicate}).encode("utf-8")
        return UpstreamRequest(
            operation="delete",
            method="POST",
            path_qs=f"/api/v2/delete?{urlencode({'org': tenant, 'bucket': store})}",
            headers={**auth, "Content-Type": "application/json"},
            body=body,
            series=series,
        )

    def translate_response(
        self, upstream: UpstreamRequest, status: int, body: bytes
    ) -> tuple[int, bytes, str]:
        if upstream.operation is None:
            if status == 200:
                return 200, _json({"state": "up"}), "application/json"
            return 503, _json({"state": "down", "upstream_status": status}), "application/json"
        if status >= 400:
            return (
                status,
                _json({"error": _text(body), "upstream_status": status}),
                "application/json",
            )
        if upstream.operation == "write":
            return (
                201,
                _json({"written": upstream.n_records, "series": upstream.series}),
                "application/json",
            )
        if upstream.operation == "delete":
            return 200, _json({"removed": True, "series": upstream.series}), "application/json"
        points = []
        for row in parse_csv_rows(_text(body)):
            point: dict[str, Any] = {
                "t": row.get("ts"),
                "metric": row.get("metric"),
                "v": row.get("value"),
            }
            for key, value in row.items():
                if key not in ("ts", "metric", "value", "series"):
                    point[key] = value
            points.append(point)
        return (
            200,
            _json({"series": upstream.series, "count": len(points), "points": points}),
            "application/json",
        )

    def _reject(
        self, status: int, code: str, message: str, operation: str | None = None
    ) -> LocalResponse:
        return LocalResponse(
            status=status,
            body=_json({"error": message, "code": wire_code(code)}),
            operation=operation,
            rejection=code,
            api=API_KESTREL,
        )


def index_document() -> dict[str, Any]:
    points = "/v1/tenants/{tenant}/stores/{store}/series/{series}/points"
    return {
        "service": "Kestrel metrics service",
        "api": "v1",
        "auth": {
            "header": TOKEN_HEADER,
            "description": "API token, required on every points endpoint",
        },
        "endpoints": [
            {"method": "GET", "path": f"{API_PREFIX}/status", "summary": "service state"},
            {
                "method": "PUT",
                "path": points,
                "query": ["unit (ns|us|ms|s, default ns)"],
                "body": "JSON object",
                "summary": "append points to a series; 201 with {written: n}",
            },
            {
                "method": "GET",
                "path": points,
                "query": ["since", "until", "limit", "label.<name>"],
                "summary": "read points of a series; 200 with {count, points}",
            },
            {
                "method": "DELETE",
                "path": points,
                "query": ["from", "to", "label.<name>"],
                "summary": "remove points of a series in a time range; 200 with {removed}",
            },
        ],
        "errors": "4xx replies carry {error: <message>} naming the expected parameter or field",
    }


def iso_duration_seconds(text: str) -> int | None:
    match = _ISO_DURATION_RE.match(text)
    if not match or text in ("P", "PT"):
        return None
    days, hours, minutes, seconds = (int(part) if part else 0 for part in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _time_literal(text: str) -> str | None:
    text = text.strip()
    seconds = iso_duration_seconds(text)
    if seconds is not None:
        return f"-{seconds}s"
    if _RFC3339_RE.match(text):
        return text
    return None


def _label_params(query: Mapping[str, str]) -> dict[str, str]:
    return {
        key.removeprefix("label."): value
        for key, value in query.items()
        if key.startswith("label.")
    }


def _parse_json_object(body: bytes) -> dict[str, Any] | None:
    if not body:
        return None
    try:
        payload = json.loads(body.decode("utf-8"))
    except ValueError, UnicodeDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _unknown_endpoint(path: str) -> LocalResponse:
    legacy = path.startswith(_LEGACY_PATHS)
    if legacy:
        rejection = "legacy_endpoint"
    elif path.startswith("/tessera/"):
        rejection = "foreign_endpoint"
    else:
        rejection = "unknown_endpoint"
    return LocalResponse(
        status=404,
        body=_json(
            {
                "error": (
                    "unknown endpoint; this service speaks the Kestrel REST API v1 — "
                    "GET /v1 lists the endpoints"
                ),
                "code": "unknown_endpoint",
            }
        ),
        operation=map_path_to_operation(path),
        rejection=rejection,
        api=API_LEGACY if legacy else API_UNKNOWN,
    )
