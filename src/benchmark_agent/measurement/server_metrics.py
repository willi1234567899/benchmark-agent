from __future__ import annotations

import math
import re
from collections.abc import Callable

import httpx

from benchmark_agent.measurement.op_mapping import Operation, map_path_to_operation
from benchmark_agent.models import MetricsSnapshot, OperationWindowStats, ServerMetricsDelta

_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>(?:[^}\"]|\"(?:[^\"\\]|\\.)*\")*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')
_LABEL_ESCAPES = {"n": "\n", '"': '"', "\\": "\\"}

_COUNT_METRIC_CANDIDATES = (
    "http_api_requests_total",
    "http_requests_total",
)
_DURATION_COUNT_CANDIDATES = (
    "http_api_request_duration_seconds_count",
    "http_request_duration_seconds_count",
)
_DURATION_SUM_CANDIDATES = (
    "http_api_request_duration_seconds_sum",
    "http_request_duration_seconds_sum",
)
_DURATION_BUCKET_CANDIDATES = (
    "http_api_request_duration_seconds_bucket",
    "http_request_duration_seconds_bucket",
)

_PATH_LABEL_KEYS = ("path", "handler", "endpoint", "uri")
_ERROR_STATUS_CLASSES = {"4XX", "5XX"}

LabelPredicate = Callable[[dict[str, str]], bool]


class ServerMetricsCollector:
    def snapshot(self, metrics_url: str, *, timeout_seconds: float = 5.0) -> MetricsSnapshot:
        response = httpx.get(metrics_url, timeout=timeout_seconds)
        response.raise_for_status()
        return MetricsSnapshot(samples=_parse_prometheus_text(response.text))

    def delta(self, start: MetricsSnapshot, end: MetricsSnapshot) -> ServerMetricsDelta:
        by_operation: dict[str, OperationWindowStats] = {}
        for operation in ("write", "read", "delete"):
            n_requests = int(
                max(
                    0.0,
                    _operation_count(end.samples, operation)
                    - _operation_count(start.samples, operation),
                )
            )
            error_count = int(
                max(
                    0.0,
                    _operation_error_count(end.samples, operation)
                    - _operation_error_count(start.samples, operation),
                )
            )

            p50, p95, p99 = _estimate_percentiles(start.samples, end.samples, operation)
            if p50 is None:
                start_sum = _operation_duration_sum(start.samples, operation)
                end_sum = _operation_duration_sum(end.samples, operation)
                if start_sum is not None and end_sum is not None and n_requests:
                    p50 = (max(0.0, end_sum - start_sum) / n_requests) * 1000.0

            by_operation[operation] = OperationWindowStats(
                n_requests=n_requests,
                error_count=error_count,
                throughput_ops_s=float(n_requests),
                latency_p50_ms=p50,
                latency_p95_ms=p95,
                latency_p99_ms=p99,
            )
        return ServerMetricsDelta(by_operation=by_operation)


def _parse_prometheus_text(text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_LINE_RE.match(line)
        if match is None:
            continue
        name = match.group("name")
        labels = match.group("labels") or ""
        value = float(match.group("value"))
        key = f"{name}{{{labels}}}" if labels else name
        samples[key] = value
    return samples


def _operation_count(samples: dict[str, float], operation: Operation) -> float:
    for metric_name in (*_COUNT_METRIC_CANDIDATES, *_DURATION_COUNT_CANDIDATES):
        total = _sum_matching(samples, metric_name, operation)
        if total is not None:
            return total
    return 0.0


def _operation_error_count(samples: dict[str, float], operation: Operation) -> float:
    for metric_name in _COUNT_METRIC_CANDIDATES:
        total = _sum_matching(samples, metric_name, operation, predicate=_is_error_label_set)
        if total is not None:
            return total
    return 0.0


def _is_error_label_set(labels: dict[str, str]) -> bool:
    if labels.get("status", "").upper() in _ERROR_STATUS_CLASSES:
        return True
    code = labels.get("response_code") or labels.get("code") or labels.get("status_code")
    if code is None:
        return False
    try:
        return int(code) >= 400
    except ValueError:
        return False


def _operation_duration_sum(samples: dict[str, float], operation: Operation) -> float | None:
    for metric_name in _DURATION_SUM_CANDIDATES:
        total = _sum_matching(samples, metric_name, operation)
        if total is not None:
            return total
    return None


def _estimate_percentiles(
    start: dict[str, float],
    end: dict[str, float],
    operation: Operation,
) -> tuple[float | None, float | None, float | None]:
    for metric_name in _DURATION_BUCKET_CANDIDATES:
        end_buckets = _bucket_totals(end, metric_name, operation)
        if not end_buckets:
            continue
        start_buckets = _bucket_totals(start, metric_name, operation)
        deltas = {
            le: max(0.0, count - start_buckets.get(le, 0.0)) for le, count in end_buckets.items()
        }
        return (
            _histogram_quantile(deltas, 0.50),
            _histogram_quantile(deltas, 0.95),
            _histogram_quantile(deltas, 0.99),
        )
    return None, None, None


def _bucket_totals(
    samples: dict[str, float], metric_name: str, operation: Operation
) -> dict[float, float]:
    totals: dict[float, float] = {}
    for key, value in samples.items():
        name, labels = _split_sample_key(key)
        if name != metric_name or "le" not in labels:
            continue
        path = _label_value(labels, _PATH_LABEL_KEYS)
        if path is None or map_path_to_operation(path) != operation:
            continue
        try:
            le = float(labels["le"])
        except ValueError:
            continue
        totals[le] = totals.get(le, 0.0) + value
    return totals


def _histogram_quantile(buckets: dict[float, float], quantile: float) -> float | None:
    if not buckets:
        return None
    bounds = sorted(buckets)
    total = buckets[bounds[-1]]
    if total <= 0:
        return None
    rank = quantile * total
    lower_bound = 0.0
    lower_count = 0.0
    for upper_bound in bounds:
        count = buckets[upper_bound]
        if count >= rank:
            if math.isinf(upper_bound):
                return lower_bound * 1000.0
            width = count - lower_count
            if width <= 0:
                return upper_bound * 1000.0
            fraction = (rank - lower_count) / width
            return (lower_bound + (upper_bound - lower_bound) * fraction) * 1000.0
        lower_bound = upper_bound
        lower_count = count
    return (0.0 if math.isinf(bounds[-1]) else bounds[-1]) * 1000.0


def _sum_matching(
    samples: dict[str, float],
    metric_name: str,
    operation: Operation,
    *,
    predicate: LabelPredicate | None = None,
) -> float | None:
    matched = 0.0
    found = False
    for key, value in samples.items():
        name, labels = _split_sample_key(key)
        if name != metric_name:
            continue
        path = _label_value(labels, _PATH_LABEL_KEYS)
        if path is None:
            continue
        if map_path_to_operation(path) != operation:
            continue
        found = True
        if predicate is not None and not predicate(labels):
            continue
        matched += value
    return matched if found else None


def _split_sample_key(key: str) -> tuple[str, dict[str, str]]:
    if "{" not in key:
        return key, {}
    name, rest = key.split("{", 1)
    labels = {label: _unescape_label_value(value) for label, value in _LABEL_RE.findall(rest)}
    return name, labels


def _unescape_label_value(value: str) -> str:
    return re.sub(r"\\(.)", _unescape_match, value)


def _unescape_match(match: re.Match[str]) -> str:
    escaped = match.group(1)
    return _LABEL_ESCAPES.get(escaped, escaped)


def _label_value(labels: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in labels:
            return labels[key]
    return None
