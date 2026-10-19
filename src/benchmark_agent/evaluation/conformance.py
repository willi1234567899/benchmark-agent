from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from benchmark_agent.models import (
    ConformanceCheck,
    ConformanceResult,
    OperationSpec,
    ProxyWindowStats,
    RequestSample,
    WorkloadSpec,
)

CRITERIA_VERSION = 2
SPEC_ONLY_CHECKS: frozenset[tuple[str, str]] = frozenset(
    {("write", "ts_unit"), ("read", "labels"), ("delete", "labels")}
)
_PRECISION_RE = re.compile(r"[?&](?:precision|ts_unit|unit)=([a-z]+)")

_LIMIT_RE = re.compile(r"limit\(\s*n\s*:\s*(\d+)\s*\)")
_RANGE_RE = re.compile(r"range\(\s*start\s*:\s*([^,)\s]+)")
_MEASUREMENT_RE = re.compile(r"""_measurement\s*==?\s*["']([^"']+)["']""")
_STOP_SLACK_SECONDS = 300.0
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?$")


def evaluate_conformance(
    workload: WorkloadSpec, proxy_stats: ProxyWindowStats | None
) -> ConformanceResult | None:
    if proxy_stats is None:
        return None
    checks: list[ConformanceCheck] = []
    deviations: list[str] = []
    target = _target_measurement(workload)
    operations = workload.operations

    write = operations.get("write")
    if write is not None and write.enabled:
        _add_write_checks(write, proxy_stats, checks, deviations)

    read = operations.get("read")
    if read is not None and read.enabled:
        _add_read_checks(read, proxy_stats, target, checks, deviations)

    delete = operations.get("delete")
    if delete is not None and delete.enabled:
        _add_delete_checks(delete, proxy_stats, target, checks, deviations)

    spec_checks = [check for check in checks if (check.operation, check.name) in SPEC_ONLY_CHECKS]
    return ConformanceResult(
        ok=all(check.ok for check in checks),
        checks=checks,
        deviations=deviations,
        workload_name=workload.name,
        criteria_version=CRITERIA_VERSION,
        spec_coverage=(
            sum(1 for check in spec_checks if check.ok) / len(spec_checks) if spec_checks else None
        ),
        spec_checks=[
            f"{check.operation}/{check.name}:{'ok' if check.ok else 'fail'}"
            for check in spec_checks
        ],
    )


def _add_write_checks(
    write: OperationSpec,
    proxy_stats: ProxyWindowStats,
    checks: list[ConformanceCheck],
    deviations: list[str],
) -> None:
    settings = write.settings_dict()
    expected_points = int(settings.get("batch_size", 0)) * int(settings.get("batches", 1))
    stats = proxy_stats.by_operation.get("write")
    accepted = stats.n_points if stats else 0
    requests = (stats.n_requests - stats.error_count) if stats else 0
    checks.append(
        ConformanceCheck(
            operation="write",
            name="points",
            ok=expected_points > 0 and accepted == expected_points,
            detail=f"accepted points {accepted}, pinned {expected_points}",
        )
    )
    batches = int(settings.get("batches", 1))
    if requests and requests != batches:
        deviations.append(f"write: {requests} write requests, pinned {batches} batch(es)")
    expected_unit = settings.get("ts_unit")
    if expected_unit:
        units = [_time_unit_of(sample) for sample in _samples(proxy_stats, "write")]
        checks.append(
            ConformanceCheck(
                operation="write",
                name="ts_unit",
                ok=bool(units) and all(unit == str(expected_unit) for unit in units),
                detail=f"time units {units}, pinned {expected_unit}",
            )
        )


def _add_read_checks(
    read: OperationSpec,
    proxy_stats: ProxyWindowStats,
    target: str | None,
    checks: list[ConformanceCheck],
    deviations: list[str],
) -> None:
    settings = read.settings_dict()
    template = str(settings.get("query_template", ""))
    pinned_limit = _first(_LIMIT_RE, template)
    pinned_start = _first(_RANGE_RE, template)
    queries = [_flux_of(sample) for sample in _samples(proxy_stats, "read")]
    queries = [query for query in queries if query]
    repeats = int(settings.get("repeats", 1) or 1)
    if queries and len(queries) != repeats:
        deviations.append(f"read: {len(queries)} read requests, pinned {repeats}")
    checks.append(
        _read_check(
            "limit",
            queries,
            lambda query: pinned_limit is None or _first(_LIMIT_RE, query) == pinned_limit,
            f"pinned limit(n: {pinned_limit})",
        )
    )
    checks.append(
        _read_check(
            "range",
            queries,
            lambda query: (
                pinned_start is None or _same_start(_first(_RANGE_RE, query), pinned_start)
            ),
            f"pinned range(start: {pinned_start})",
        )
    )
    checks.append(
        _read_check(
            "measurement",
            queries,
            lambda query: target is None or _mentions_measurement(query, target),
            f"pinned measurement {target!r}",
        )
    )
    read_labels = settings.get("labels")
    if isinstance(read_labels, dict) and read_labels:
        checks.append(
            _read_check(
                "labels",
                queries,
                lambda query: all(
                    _flux_filters_label(query, str(key), str(value))
                    for key, value in read_labels.items()
                ),
                f"pinned label filter {read_labels}",
            )
        )


def _add_delete_checks(
    delete: OperationSpec,
    proxy_stats: ProxyWindowStats,
    target: str | None,
    checks: list[ConformanceCheck],
    deviations: list[str],
) -> None:
    settings = delete.settings_dict()
    pinned_delete_start = _parse_time(str(settings.get("start", "")))
    pinned_delete_stop = _parse_time(str(settings.get("stop", "")))
    bodies = [_json_of(sample) for sample in _samples(proxy_stats, "delete")]
    bodies = [body for body in bodies if body]
    predicates = [str(body.get("predicate", "")) for body in bodies]
    checks.append(
        ConformanceCheck(
            operation="delete",
            name="predicate",
            ok=any(
                target is None or _mentions_measurement(predicate, target)
                for predicate in predicates
            ),
            detail=(
                f"predicates {predicates}, pinned measurement {target!r}"
                if predicates
                else "no delete request body seen"
            ),
        )
    )
    ranges = [
        (_parse_time(str(body.get("start", ""))), _parse_time(str(body.get("stop", ""))))
        for body in bodies
    ]
    full_range = any(
        start is not None
        and stop is not None
        and (pinned_delete_start is None or start <= pinned_delete_start)
        and (pinned_delete_stop is None or stop >= pinned_delete_stop)
        for start, stop in ranges
    )
    data_written_after = datetime.now(UTC) - timedelta(
        seconds=proxy_stats.window_duration_s + _STOP_SLACK_SECONDS
    )
    covers_data = any(
        start is not None
        and stop is not None
        and (pinned_delete_start is None or start <= pinned_delete_start)
        and stop >= data_written_after
        for start, stop in ranges
    )
    checks.append(
        ConformanceCheck(
            operation="delete",
            name="time_range",
            ok=covers_data,
            detail=(
                f"ranges {[(str(body.get('start')), str(body.get('stop'))) for body in bodies]}, "
                f"pinned {settings.get('start')} .. {settings.get('stop')}"
            ),
        )
    )
    if covers_data and not full_range:
        deviations.append(
            "delete: time range covers the data but differs from the pinned literals "
            f"{settings.get('start')} .. {settings.get('stop')}"
        )
    delete_labels = settings.get("labels")
    if isinstance(delete_labels, dict) and delete_labels:
        checks.append(
            ConformanceCheck(
                operation="delete",
                name="labels",
                ok=any(
                    all(
                        _predicate_has_label(predicate, str(key), str(value))
                        for key, value in delete_labels.items()
                    )
                    for predicate in predicates
                ),
                detail=f"predicates {predicates}, pinned label filter {delete_labels}",
            )
        )


_FLUX_DURATION_RE = re.compile(r"^-?(\d+)(ns|us|ms|s|m|h|d|w)$")
_UNIT_SECONDS = {
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "w": 604800.0,
}


def _same_start(found: str | None, pinned: str) -> bool:
    if found is None:
        return False
    if found == pinned:
        return True
    found_duration = _FLUX_DURATION_RE.match(found)
    pinned_duration = _FLUX_DURATION_RE.match(pinned)
    if found_duration and pinned_duration:
        difference = _duration_seconds(found_duration) - _duration_seconds(pinned_duration)
        return abs(difference) < 1e-9
    return False


def _duration_seconds(duration: re.Match[str]) -> float:
    return float(duration.group(1)) * _UNIT_SECONDS[duration.group(2)]


def _time_unit_of(sample: RequestSample) -> str:
    for candidate in (sample.upstream_path, sample.path):
        match = _PRECISION_RE.search(candidate or "")
        if match:
            return match.group(1)
    return "ns"


def _flux_filters_label(flux: str, key: str, value: str) -> bool:
    pattern = re.compile(
        r"r(?:\[\s*\""
        + re.escape(key)
        + r"\"\s*\]|\."
        + re.escape(key)
        + r")\s*==\s*\""
        + re.escape(value)
        + r"\""
    )
    return bool(pattern.search(flux))


def _predicate_has_label(predicate: str, key: str, value: str) -> bool:
    pattern = re.compile(r"\b" + re.escape(key) + r"\s*=\s*\"" + re.escape(value) + r"\"")
    return bool(pattern.search(predicate))


def _read_check(
    name: str, queries: list[str], predicate: Callable[[str], bool], pinned: str
) -> ConformanceCheck:
    ok = any(predicate(query) for query in queries)
    detail = (
        f"{len(queries)} quer{'y' if len(queries) == 1 else 'ies'} seen, {pinned}"
        if queries
        else "no query body seen"
    )
    return ConformanceCheck(operation="read", name=name, ok=ok, detail=detail)


def _target_measurement(workload: WorkloadSpec) -> str | None:
    delete = workload.operations.get("delete")
    if delete is not None:
        predicate = str(delete.settings_dict().get("predicate", ""))
        found = _first(_MEASUREMENT_RE, predicate)
        if found:
            return found
    return None


def _samples(proxy_stats: ProxyWindowStats, operation: str) -> list[RequestSample]:
    return [
        sample
        for sample in proxy_stats.requests
        if sample.operation == operation and not sample.error
    ]


def _flux_of(sample: RequestSample) -> str:
    body = sample.body_excerpt or ""
    payload = _json_text(body)
    if isinstance(payload, dict) and isinstance(payload.get("query"), str):
        return str(payload["query"])
    return body


def _json_of(sample: RequestSample) -> dict[str, object]:
    payload = _json_text(sample.body_excerpt or "")
    return payload if isinstance(payload, dict) else {}


def _json_text(text: str) -> object:
    try:
        return json.loads(text)
    except ValueError:
        return None


def _first(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1) if match else None


def _mentions_measurement(text: str, measurement: str) -> bool:
    found = _MEASUREMENT_RE.findall(text)
    if found:
        return measurement in found
    return f'"{measurement}"' in text or f"'{measurement}'" in text


def _parse_time(text: str) -> datetime | None:
    text = text.strip()
    if not text or not _ISO_RE.match(text):
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
