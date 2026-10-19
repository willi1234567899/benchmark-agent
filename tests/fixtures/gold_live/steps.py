"""Harness gold fixture: live InfluxDB HTTP for write, query, and delete (workload v2).

This is not LLM output. It is the known-good artifact used to prove that
evaluate → proxy → CSV works before generated benchmarks are measured.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from behave import given, then, when

_WRITE_BATCH_SIZE = 100
_QUERY = (
    'from(bucket: "{bucket}") '
    "|> range(start: -2h) "
    '|> filter(fn: (r) => r._measurement == "benchmark") '
    '|> filter(fn: (r) => r.host == "h1") '
    "|> limit(n: 25)"
)
_DELETE_BODY = {
    "start": "1970-01-01T00:00:00Z",
    "stop": "2100-01-01T00:00:00Z",
    "predicate": '_measurement="benchmark" AND host="h1"',
}


@given("InfluxDB connection settings are available")
def step_connection_settings(context) -> None:
    context.influx_url = _required_env("INFLUX_URL", fallback="INFLUXDB_URL").rstrip("/")
    context.influx_token = _required_env("INFLUX_TOKEN")
    context.influx_org = _required_env("INFLUX_ORG")
    context.influx_bucket = _required_env("INFLUX_BUCKET")
    context.auth_headers = {
        "Authorization": f"Token {context.influx_token}",
    }


_WRITE_BATCHES = 10
_READ_REPEATS = 20


@when("the pinned write batch is sent")
def step_write_batch(context) -> None:
    # Workload v3: 10 sequential batches of 100 points, millisecond timestamps
    # (precision=ms) and labels host=h1..h5.
    now_ms = time.time_ns() // 1_000_000
    url = f"{context.influx_url}/api/v2/write?" + urllib.parse.urlencode(
        {"org": context.influx_org, "bucket": context.influx_bucket, "precision": "ms"}
    )
    total = _WRITE_BATCHES * _WRITE_BATCH_SIZE
    context.write_results = []
    for batch in range(_WRITE_BATCHES):
        offset = batch * _WRITE_BATCH_SIZE
        lines = "\n".join(
            f"benchmark,source=gold,host=h{index % 5 + 1} value={offset + index}i "
            f"{now_ms - (total - offset - index) * 1000}"
            for index in range(_WRITE_BATCH_SIZE)
        )
        context.write_results.append(
            _request(
                "POST",
                url,
                headers={**context.auth_headers, "Content-Type": "text/plain; charset=utf-8"},
                data=lines.encode("utf-8"),
            )
        )
    context.write_status, context.write_body = context.write_results[-1]


@then("the write is accepted")
def step_write_accepted(context) -> None:
    assert len(context.write_results) == _WRITE_BATCHES
    for status, body in context.write_results:
        assert status in {204, 200}, _http_failure("write", status, body)


@when("the pinned read query is sent")
def step_read_query(context) -> None:
    # Workload v3: the identical query 20 times in a row.
    query = urllib.parse.urlencode({"org": context.influx_org})
    url = f"{context.influx_url}/api/v2/query?{query}"
    flux = _QUERY.format(bucket=context.influx_bucket)
    context.query_results = [
        _request(
            "POST",
            url,
            headers={
                **context.auth_headers,
                "Content-Type": "application/vnd.flux",
                "Accept": "application/csv",
            },
            data=flux.encode("utf-8"),
        )
        for _ in range(_READ_REPEATS)
    ]
    context.query_status, context.query_body = context.query_results[-1]


@then("the query is accepted")
def step_query_accepted(context) -> None:
    assert len(context.query_results) == _READ_REPEATS
    for status, body in context.query_results:
        assert status == 200, _http_failure("query", status, body)
    assert context.query_body.strip(), "query returned an empty body"


@when("the pinned delete is sent")
def step_delete(context) -> None:
    url = f"{context.influx_url}/api/v2/delete?" + urllib.parse.urlencode(
        {"org": context.influx_org, "bucket": context.influx_bucket}
    )
    context.delete_status, context.delete_body = _request(
        "POST",
        url,
        headers={**context.auth_headers, "Content-Type": "application/json"},
        data=json.dumps(_DELETE_BODY).encode("utf-8"),
    )


@then("the delete is accepted")
def step_delete_accepted(context) -> None:
    assert context.delete_status in {204, 200}, _http_failure(
        "delete", context.delete_status, context.delete_body
    )


def _required_env(name: str, *, fallback: str | None = None) -> str:
    value = os.environ.get(name) or (os.environ.get(fallback) if fallback else None)
    if not value:
        names = name if fallback is None else f"{name} or {fallback}"
        raise AssertionError(f"Missing environment variable {names}")
    return value


def _request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    data: bytes | None = None,
    timeout: float = 30.0,
) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _http_failure(operation: str, status: int, body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")[:500]
    return f"{operation} failed with HTTP {status}: {text}"
