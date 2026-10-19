"""Harness gold fixture for the Kestrel facade: write, read and remove points (workload v2).

This is not LLM output. It is the known-good artifact that proves evaluate -> facade ->
store -> proxy -> CSV works for the second dialect. It uses only the Kestrel REST API v1
(inputs/openapi/kestrel-v1.yaml): millisecond times (unit=ms), labels host=h1..h5, label
filter host=h1 in read and delete.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from behave import given, then, when

_BATCH_SIZE = 100
_BATCHES = 10
_READS = 20
_HOSTS = ("h1", "h2", "h3", "h4", "h5")


@given("Kestrel connection settings are available")
def step_connection_settings(context) -> None:
    context.endpoint = _required_env("KESTREL_ENDPOINT").rstrip("/")
    context.token = _required_env("KESTREL_TOKEN")
    context.tenant = _required_env("KESTREL_TENANT")
    context.store = _required_env("KESTREL_STORE")
    context.headers = {"X-Api-Token": context.token, "Content-Type": "application/json"}
    context.points_url = (
        f"{context.endpoint}/v1/tenants/{urllib.parse.quote(context.tenant)}"
        f"/stores/{urllib.parse.quote(context.store)}/series/benchmark/points"
    )


@when("the pinned write batch is sent")
def step_write(context) -> None:
    # Workload v3: 10 sequential batches of 100 points (1000 points in total).
    now_ms = time.time_ns() // 1_000_000
    context.write_results = []
    for batch in range(_BATCHES):
        body = {
            "columns": ["t", "host", "source", "value"],
            "labels": ["host", "source"],
            "rows": [
                [
                    now_ms - (_BATCHES * _BATCH_SIZE - batch * _BATCH_SIZE - index) * 1_000,
                    _HOSTS[index % len(_HOSTS)],
                    "gold",
                    float(batch * _BATCH_SIZE + index),
                ]
                for index in range(_BATCH_SIZE)
            ],
        }
        context.write_results.append(
            _request(
                "PUT",
                f"{context.points_url}?unit=ms",
                headers=context.headers,
                data=json.dumps(body).encode(),
            )
        )
    context.write_status, context.write_body = context.write_results[-1]


@then("the write is accepted")
def step_write_accepted(context) -> None:
    assert len(context.write_results) == _BATCHES
    for status, body in context.write_results:
        assert status == 201, _http_failure("write", status, body)
        assert json.loads(body)["written"] == _BATCH_SIZE


@when("the pinned read is sent")
def step_read(context) -> None:
    # Workload v3: the identical read 20 times in a row.
    query = urllib.parse.urlencode({"since": "PT2H", "limit": 25, "label.host": "h1"})
    context.read_results = [
        _request("GET", f"{context.points_url}?{query}", headers={"X-Api-Token": context.token})
        for _ in range(_READS)
    ]
    context.read_status, context.read_body = context.read_results[-1]


@then("the read is accepted")
def step_read_accepted(context) -> None:
    assert len(context.read_results) == _READS
    for status, body in context.read_results:
        assert status == 200, _http_failure("read", status, body)
    payload = json.loads(context.read_body)
    assert 1 <= payload["count"] <= 25, f"unexpected count {payload['count']}"
    assert all(point.get("host", "h1") == "h1" for point in payload["points"])


@when("the pinned removal is sent")
def step_delete(context) -> None:
    query = urllib.parse.urlencode(
        {"from": "1970-01-01T00:00:00Z", "to": "2100-01-01T00:00:00Z", "label.host": "h1"}
    )
    context.delete_status, context.delete_body = _request(
        "DELETE", f"{context.points_url}?{query}", headers={"X-Api-Token": context.token}
    )


@then("the removal is accepted")
def step_delete_accepted(context) -> None:
    assert context.delete_status == 200, _http_failure(
        "delete", context.delete_status, context.delete_body
    )
    assert json.loads(context.delete_body)["removed"] is True


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise AssertionError(f"Missing environment variable {name}")
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
