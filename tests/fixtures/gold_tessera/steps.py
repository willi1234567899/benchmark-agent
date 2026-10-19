"""Harness gold fixture for the Tessera facade: ingest, fetch and purge over HTTP (workload v2).

This is not LLM output. It is the known-good artifact that proves evaluate -> facade ->
store -> proxy -> CSV works before generated benchmarks are measured behind the facade.
It uses only the Tessera API v1 (inputs/openapi/tessera-v1.yaml) and follows workload v2:
millisecond timestamps (ts_unit=ms), labels host=h1..h5, label filter host=h1 in fetch and purge.
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
_FETCHES = 20
_HOSTS = ("h1", "h2", "h3", "h4", "h5")
_FETCH_BODY = {
    "series": "benchmark",
    "window": {"from": "-2h"},
    "max_rows": 25,
    "labels": {"host": "h1"},
}
_PURGE_BODY = {
    "series": "benchmark",
    "from": "1970-01-01T00:00:00Z",
    "until": "2100-01-01T00:00:00Z",
    "labels": {"host": "h1"},
}


@given("Tessera connection settings are available")
def step_connection_settings(context) -> None:
    context.url = _required_env("TESSERA_URL").rstrip("/")
    context.key = _required_env("TESSERA_KEY")
    context.space = _required_env("TESSERA_SPACE")
    context.store = _required_env("TESSERA_STORE")
    context.headers = {"X-Tessera-Key": context.key, "Content-Type": "application/json"}


@when("the pinned ingest batch is sent")
def step_ingest(context) -> None:
    # Workload v3: 10 sequential batches of 100 records (1000 records in total).
    now_ms = time.time_ns() // 1_000_000
    url = f"{context.url}/tessera/v1/ingest?" + urllib.parse.urlencode(
        {"space": context.space, "store": context.store, "ts_unit": "ms"}
    )
    context.ingest_results = []
    for batch in range(_BATCHES):
        records = [
            {
                "series": "benchmark",
                "labels": {"host": _HOSTS[index % len(_HOSTS)], "source": "gold"},
                "values": {"value": float(batch * _BATCH_SIZE + index)},
                "ts": now_ms - (_BATCHES * _BATCH_SIZE - batch * _BATCH_SIZE - index) * 1_000,
            }
            for index in range(_BATCH_SIZE)
        ]
        context.ingest_results.append(
            _request(
                "POST",
                url,
                headers=context.headers,
                data=json.dumps({"records": records}).encode(),
            )
        )
    context.ingest_status, context.ingest_body = context.ingest_results[-1]


@then("the ingest is accepted")
def step_ingest_accepted(context) -> None:
    assert len(context.ingest_results) == _BATCHES
    for status, body in context.ingest_results:
        assert status == 200, _http_failure("ingest", status, body)
        assert json.loads(body)["accepted"] == _BATCH_SIZE


@when("the pinned fetch is sent")
def step_fetch(context) -> None:
    # Workload v3: the identical fetch 20 times in a row.
    url = f"{context.url}/tessera/v1/fetch?" + urllib.parse.urlencode({"space": context.space})
    body = {"store": context.store, **_FETCH_BODY}
    context.fetch_results = [
        _request("POST", url, headers=context.headers, data=json.dumps(body).encode())
        for _ in range(_FETCHES)
    ]
    context.fetch_status, context.fetch_body = context.fetch_results[-1]


@then("the fetch is accepted")
def step_fetch_accepted(context) -> None:
    assert len(context.fetch_results) == _FETCHES
    for status, body in context.fetch_results:
        assert status == 200, _http_failure("fetch", status, body)
    payload = json.loads(context.fetch_body)
    assert 1 <= payload["row_count"] <= 25, f"unexpected row_count {payload['row_count']}"
    assert all(row.get("host", "h1") == "h1" for row in payload["rows"])


@when("the pinned purge is sent")
def step_purge(context) -> None:
    url = f"{context.url}/tessera/v1/purge?" + urllib.parse.urlencode(
        {"space": context.space, "store": context.store}
    )
    context.purge_status, context.purge_body = _request(
        "POST", url, headers=context.headers, data=json.dumps(_PURGE_BODY).encode()
    )


@then("the purge is accepted")
def step_purge_accepted(context) -> None:
    assert context.purge_status == 200, _http_failure(
        "purge", context.purge_status, context.purge_body
    )
    assert json.loads(context.purge_body)["purged"] is True


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
