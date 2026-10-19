# >>> port-only: header, merged imports, PORT C02/C13 helpers
"""DPS25/bddbench (commit 42ddad5) write / query / delete, ported onto the PINNED workload v3
(InfluxDB passthrough mode, no facade) as one steps module for the Benchmark-Agent engine.

Not LLM output and not re-implemented: every code block below is a line range of the named
bddbench file (markers `# >>> bddbench <file>:<a>-<b> verbatim|CHANGED <ids>`), assembled by
tools/build_pinned_port.py. CHANGED blocks differ from the source only by the `PORT Cxx` edits
listed in PROVENANCE.md (full unified diff: changes.diff). Client library: influxdb-client
1.49.0 (the version in bddbench's uv.lock): synchronous WriteApi, QueryApi.query_csv,
DeleteApi - constructed and timed exactly as bddbench does.

Requests sent inside the measurement window (same operations, counts, order and targets as
tests/fixtures/gold_live):
  10 x POST /api/v2/write?org=..&bucket=..&precision=ms  (100 points each, sequential)
  20 x POST /api/v2/query?org=..                          (identical Flux, sequential)
   1 x POST /api/v2/delete?org=..&bucket=..               (_measurement="benchmark" AND host="h1")
No ping, no /health, no bucket lookups, no count queries, no KPI export, no ssh.
"""

import json
import logging
import math
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from behave import given, then, when
from behave.runner import Context
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# PORT C13: one second expressed in each WritePrecision unit (gold_live spaces its points 1 s).
_ONE_SECOND = {
    WritePrecision.NS: 1_000_000_000,
    WritePrecision.US: 1_000_000,
    WritePrecision.MS: 1_000,
    WritePrecision.S: 1,
}


def _map_engine_env_to_bddbench() -> None:
    """PORT C02: the engine injects INFLUX_URL / INFLUXDB_URL / INFLUX_TOKEN / INFLUX_ORG /
    INFLUX_BUCKET into the behave process (execution/sut.py); bddbench reads INFLUXDB_SUT_*
    (environment.py:147-150). The engine strips inherited *_TOKEN variables, so the mapping has
    to happen here, inside the behave process. Engine values always win."""
    mapping = (
        ("INFLUXDB_SUT_URL", ("INFLUX_URL", "INFLUXDB_URL")),
        ("INFLUXDB_SUT_TOKEN", ("INFLUX_TOKEN",)),
        ("INFLUXDB_SUT_ORG", ("INFLUX_ORG",)),
        ("INFLUXDB_SUT_BUCKET", ("INFLUX_BUCKET",)),
    )
    for target, sources in mapping:
        for source in sources:
            value = os.environ.get(source)
            if value:
                os.environ[target] = value
                break
# <<< port-only


# >>> bddbench src/utils.py:230-242 verbatim (influx_precision_from_str)
def influx_precision_from_str(p: str) -> WritePrecision:
    """
    used in influx_write and influx_multi_bucket
    maps a String on WritePrecision
    """
    p_lower = p.lower()
    if p_lower == "ns":
        return WritePrecision.NS
    if p_lower == "ms":
        return WritePrecision.MS
    if p_lower == "s":
        return WritePrecision.S
    raise ValueError(f"Unsupported precision: {p}")
# <<< bddbench src/utils.py:230-242


# >>> bddbench src/utils.py:245-254 verbatim (base_timestamp_for_precision)
def base_timestamp_for_precision(precision: WritePrecision) -> int:
    """
    used in influx_write and influx_multi_bucket
    gives a timestamp for the given precision
    """
    if precision == WritePrecision.NS:
        return time.time_ns()
    if precision == WritePrecision.MS:
        return int(time.time() * 1000)
    return int(time.time())
# <<< bddbench src/utils.py:245-254


# >>> bddbench src/utils.py:257-301 CHANGED C11,C12,C13 (build_benchmark_point)
def build_benchmark_point(
    measurement: str,
    base_ts: int,
    idx: int,
    point_complexity: str,
    tag_cardinality: int,
    time_ordering: str,
    precision: WritePrecision,
    run_id: Optional[str] = None,
) -> Point:
    """
    used in influx_write and influx_multi_bucket
    creates a Point for the @write benchmarks
    """

    device_id = idx % max(1, tag_cardinality)

    if time_ordering == "out_of_order":
        if precision == WritePrecision.NS:
            jitter = random.randint(-1_000_000_000, 1_000_000_000)
        elif precision == WritePrecision.MS:
            jitter = random.randint(-1000, 1000)
        else:
            jitter = random.randint(-1, 1)
        ts = base_ts + jitter
    else:
        ts = base_ts + idx * _ONE_SECOND[precision]  # PORT C13: 1 s apart, as gold_live

    # PORT C11: workload tag host=h1..h5 (cycled by idx) instead of device_id=dev-N; constant
    # marker tag source=gold as in gold_live, so the line protocol has the same bytes.
    p = Point(measurement).tag("source", "gold").tag("host", f"h{device_id + 1}")
    if run_id:
        p = p.tag("run_id", str(run_id))

    p = p.field("value", idx)  # PORT C12: one integer field value (gold_live: value=<n>i)

    # Feature files currently use low/medium/high. Keep backwards compatibility:
    # - medium: one extra field
    # - high:   several extra fields
    pc = (point_complexity or "").lower()
    if pc in {"medium", "high"}:
        p = p.field("aux1", float(idx % 100))
    if pc == "high":
        p = p.field("aux2", math.sin(idx)).field("aux3", math.cos(idx))

    p = p.time(ts, precision)
    return p
# <<< bddbench src/utils.py:257-301


# >>> bddbench src/utils.py:382-407 verbatim (write_json_report)
def write_json_report(
    outfile: str,
    data: Any,
    logger_: Optional[logging.Logger] = None,
    log_prefix: str = "",
) -> None:
    """
    used in:
    influx_query
    influx_write
    influx_delete
    influx_multi_bucket
    influx_user

    saves JSON reports
    """
    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    if logger_ is not None:
        if log_prefix:
            logger_.info("%s%s", log_prefix, outfile)
        else:
            logger_.info("Stored JSON report to %s", outfile)
# <<< bddbench src/utils.py:382-407


# >>> bddbench features/environment.py:14-25 verbatim; logger -> _env_logger (environment logger, _env_truthy, _env_strip)
_env_logger = logging.getLogger("bddbench.environment")

def _env_truthy(name: str, default: str = "0") -> bool:
    v = (os.getenv(name, default) or "").strip().lower()
    return v in ("1", "true", "yes", "y", "on")
    
def _env_strip(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    if v is None:
        return None
    v = str(v).strip()
    return v if v != "" else None
# <<< bddbench features/environment.py:14-25


# >>> bddbench features/environment.py:75-202 CHANGED C04,C05,C06; logger -> _env_logger (_load_env)
def _load_env(context: Context):
    """
    Load environment variables into the context.
    :param context:
    :return:
    """

    # dotenv files are loaded in before_all(). Keep this function focused on Influx init.

    context.influxdb = getattr(context, "influxdb", SimpleNamespace())
    context.influxdb.main = getattr(context.influxdb, "main", SimpleNamespace())
    context.influxdb.sut = getattr(context.influxdb, "sut", SimpleNamespace())

    context.influxdb.export_strict = _env_truthy("INFLUXDB_EXPORT_STRICT", "0")

    # PORT C04: MAIN (KPI) InfluxDB initialisation removed (bddbench environment.py:90-145):
    # no second InfluxDB, no KPI export; its default URL http://localhost:8086 is the
    # engine's measurement proxy.
    # ----- SUT influx DB -----
    context.influxdb.sut.url = _env_strip("INFLUXDB_SUT_URL", "http://localhost:8086")
    context.influxdb.sut.token = _env_strip("INFLUXDB_SUT_TOKEN", None)
    context.influxdb.sut.org = _env_strip("INFLUXDB_SUT_ORG", None)
    context.influxdb.sut.bucket = _env_strip("INFLUXDB_SUT_BUCKET", None)

    if context.influxdb.sut.url is None:
        text = "INFLUXDB_SUT_URL environment variable must be set"
        _env_logger.error(text)
        raise AssertionError(text)
    if not (context.influxdb.sut.token or "").strip():
        text = "INFLUXDB_SUT_TOKEN environment variable must be set"
        _env_logger.error(text)
        raise AssertionError(text)
    if not (context.influxdb.sut.org or "").strip():
        text = "INFLUXDB_SUT_ORG environment variable must be set"
        _env_logger.error(text)
        raise AssertionError(text)
    if not (context.influxdb.sut.bucket or "").strip():
        text = "INFLUXDB_SUT_BUCKET environment variable must be set"
        _env_logger.error(text)
        raise AssertionError(text)


    context.influxdb.sut.client = InfluxDBClient(
        url=context.influxdb.sut.url,
        token=context.influxdb.sut.token,
        org=context.influxdb.sut.org,
    )
    # PORT C05: removed (bddbench environment.py:175-183): client.ping() (GET /ping),
    # _validate_influx_auth (GET /api/v2/buckets), _validate_bucket_exists
    # (GET /api/v2/buckets?name=...) and the 'authenticated' log line - requests that
    # gold_live does not send inside the measurement window.

    context.influxdb.sut.write_api = context.influxdb.sut.client.write_api(
        write_options=SYNCHRONOUS
    )
    context.influxdb.sut.query_api = context.influxdb.sut.client.query_api()

    _env_logger.debug(f"SUT InfluxDB URL: {context.influxdb.sut.url}")
    _env_logger.debug(f"SUT InfluxDB ORG: {context.influxdb.sut.org}")
    _env_logger.debug(f"SUT InfluxDB BUCKET: {context.influxdb.sut.bucket}")



    # PORT C06: removed (bddbench environment.py:196-202): _run_on_sut(["hostname"]) (ssh
    # to the SUT host) and client.health() (GET /health); they only fed KPI-export tags.
# <<< bddbench features/environment.py:75-202


# >>> bddbench features/environment.py:231-237 verbatim (_ensure_influx_initialized)
def _ensure_influx_initialized(context: Context):
    if getattr(context, "influxdb", None) is None:
        _load_env(context)
        return
    if getattr(getattr(context.influxdb, "sut", None), "client", None) is None:
        _load_env(context)
        return
# <<< bddbench features/environment.py:231-237


# >>> bddbench features/steps/influx_global_endpoint_steps.py:8-24 CHANGED C07,C08 (Background steps)
@given("a SUT InfluxDB v2 endpoint is configured")
def step_bucket_from_env(context: Context):
    # PORT C07: bddbench text "... is configured and reachable"; its reachability check
    # client.ping() (GET /ping) is removed - gold_live sends no such request.
    # PORT C01: the engine stages no environment.py, so the SUT part of bddbench's
    # before_all (environment.py:290 -> _ensure_influx_initialized -> _load_env) runs
    # here, after PORT C02 maps the engine's variables onto bddbench's names.
    _map_engine_env_to_bddbench()
    _ensure_influx_initialized(context)


@given("the target bucket from the SUT config is set")  # PORT C08 (bddbench: "is available")
def step_target_bucket_available(context: Context) -> None:
    assert context.influxdb.sut.bucket is not None, (
        "SUT InfluxDB bucket is not configured"
    )
    # PORT C08: bucket lookup buckets_api().find_bucket_by_name (GET /api/v2/buckets?name=...)
    # removed - gold_live sends no such request.
# <<< bddbench features/steps/influx_global_endpoint_steps.py:8-24


# >>> bddbench features/steps/influx_write_steps.py:28-47 verbatim; logger -> _write_logger (write logger, BatchWriteMetrics, _maybe_cleanup_before_run)
_write_logger = logging.getLogger(f"bddbench.influx_write_steps")

# ----------- Datatype ------------


@dataclass
class BatchWriteMetrics:
    batch_index: int
    latency_s: float
    points: int
    status_code: int
    ok: bool


# ---------- Helpers ----------
def _maybe_cleanup_before_run(context, measurement: str):
    # cleanup logic not implemented
    logging.debug(
        f" NOTE: cleanup for measurement={measurement} is not implemented here."
    )
# <<< bddbench features/steps/influx_write_steps.py:28-47


# >>> bddbench features/steps/influx_write_steps.py:132-202 verbatim (_run_writer_worker)
def _run_writer_worker(
    writer_id: int,
    client: InfluxDBClient,
    bucket: str,
    org: str,
    measurement: str,
    batch_size: int,
    batches: int,
    precision_enum: WritePrecision,
    point_complexity: str,
    tag_cardinality: int,
    time_ordering: str,
    base_ts: int,
    run_id: str | None,
) -> List[BatchWriteMetrics]:
    """
    A writer writes `batches` batches per `batch_size` points
    and deliveres a list of BatchWriteMetrics.
    """
    write_api = client.write_api(write_options=SYNCHRONOUS)
    metrics: List[BatchWriteMetrics] = []

    for b in range(batches):
        batch_index = writer_id * batches + b

        points = []
        for i in range(batch_size):
            global_idx = batch_index * batch_size + i
            p = build_benchmark_point(
                measurement=measurement,
                base_ts=base_ts,
                idx=global_idx,
                point_complexity=point_complexity,
                tag_cardinality=tag_cardinality,
                time_ordering=time_ordering,
                precision=precision_enum,
                run_id=run_id
            )
            points.append(p)

        t0 = time.perf_counter()
        try:
            write_api.write(
                bucket=bucket,
                org=org,
                record=points,
            )
            t1 = time.perf_counter()
            latency_s = t1 - t0
            m = BatchWriteMetrics(
                batch_index=batch_index,
                latency_s=latency_s,
                points=len(points),
                status_code=204,
                ok=True,
            )
        except Exception as exc:
            t1 = time.perf_counter()
            latency_s = t1 - t0
            m = BatchWriteMetrics(
                batch_index=batch_index,
                latency_s=latency_s,
                points=len(points),
                status_code=500,
                ok=False,
            )
            logging.info(f" writer={writer_id} batch={batch_index} failed: {exc}")

        metrics.append(m)

    return metrics
# <<< bddbench features/steps/influx_write_steps.py:132-202


# >>> bddbench features/steps/influx_write_steps.py:205-341 CHANGED C13; logger -> _write_logger (write step)
# ---------- Scenario-Step ----------

@when(
    'I run a generic write benchmark on measurement "{measurement}" '
    "with batch size {batch_size:d}, {parallel_writers:d} parallel writers, "
    'compression "{compression}", timestamp precision "{precision}", '
    'point complexity "{point_complexity}", tag cardinality {tag_cardinality:d} '
    'and time ordering "{time_ordering}" for {batches:d} batches'
)
def step_run_write_benchmark(
    context,
    measurement,
    batch_size,
    parallel_writers,
    compression,
    precision,
    point_complexity,
    tag_cardinality,
    time_ordering,
    batches,
):
    """
    Does a Write-Benchmark woth the parameters from Scenario Outline.

    Inputs by specifikation:
      - batch_size
      - parallel_writers
      - compression (none/gzip)
      - precision (ns/ms/s)
      - point_complexity (low/high)
      - tag_cardinality
      - time_ordering (in_order/out_of_order)
      - batches (Anzahl Batches pro Writer)

    Outputs:
      - list Batch-Latenzen
      - Throughput (points/s)
      - Error-Rate
    """
    _write_logger.debug("starting generic write benchmark...")
    url = context.influxdb.sut.url
    token = context.influxdb.sut.token
    org = context.influxdb.sut.org
    bucket = context.influxdb.sut.bucket

    if not url or not token or not org or not bucket:
        raise RuntimeError(
            "INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET must be set"
        )

    _write_logger.debug("got SUT InfluxDB connection parameters")
    precision_enum = influx_precision_from_str(precision)
    # PORT C13: anchor the in-order series so that its last point lies 1 s before now and the
    # points are 1 s apart, exactly as gold_live (tests/fixtures/gold_live/steps.py:52,62).
    base_ts = base_timestamp_for_precision(precision_enum) - (
        parallel_writers * batches * batch_size * _ONE_SECOND[precision_enum]
    )

    _maybe_cleanup_before_run(context, measurement)

    enable_gzip = compression == "gzip"
    bench_client = InfluxDBClient(url=url, token=token, org=org, enable_gzip=enable_gzip)
    total_batches = parallel_writers * batches
    total_points = total_batches * batch_size

    _write_logger.debug(f"using base timestamp: {base_ts} ({precision_enum})")
    all_metrics: List[BatchWriteMetrics] = []

    started_at = time.perf_counter()
    try:    
        with ThreadPoolExecutor(max_workers=parallel_writers) as executor:
            futures = [
                executor.submit(
                    _run_writer_worker,
                    writer_id=w,
                    client=bench_client,
                    bucket=bucket,
                    org=org,
                    measurement=measurement,
                    batch_size=batch_size,
                    batches=batches,
                    precision_enum=precision_enum,
                    point_complexity=point_complexity,
                    tag_cardinality=tag_cardinality,
                    time_ordering=time_ordering,
                    base_ts=base_ts,
                    run_id=getattr(context, "run_id", None)
                )
                for w in range(parallel_writers)
            ]

            for fut in as_completed(futures):
                all_metrics.extend(fut.result())

        total_duration_s = time.perf_counter() - started_at
    finally:
        bench_client.close()

    _write_logger.debug(f"completed write benchmark in {total_duration_s:.3f} seconds")
    context.write_batches = all_metrics
    context.write_benchmark_meta = {
        "run_id": getattr(context, "run_id", None),
        "measurement": measurement,
        "batch_size": batch_size,
        "parallel_writers": parallel_writers,
        "compression": compression,
        "precision": precision,
        "point_complexity": point_complexity,
        "tag_cardinality": tag_cardinality,
        "time_ordering": time_ordering,
        "bucket": bucket,
        "org": org,
        "sut_url": url,
        "total_batches": total_batches,
        "total_points": total_points,
        "total_duration_s": total_duration_s,
    }

    latencies = [m.latency_s for m in all_metrics]
    errors = [m for m in all_metrics if not m.ok]

    throughput_points_per_s = (
        total_points / total_duration_s if total_duration_s > 0 else None
    )
    error_rate = len(errors) / len(all_metrics) if all_metrics else 0.0

    context.write_summary = {
        "latencies_s": latencies,
        "latency_stats": {
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "avg": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
        },
        "throughput": {
            "points_per_s": throughput_points_per_s,
        },
        "error_rate": error_rate,
        "errors_count": len(errors),
    }
    _write_logger.debug("generic write benchmark done")
# <<< bddbench features/steps/influx_write_steps.py:205-341


# >>> bddbench features/steps/influx_write_steps.py:344-367 CHANGED C10; logger -> _write_logger (store write result)
# ---------- write result ----------

@then('I store the generic write benchmark result as "{outfile}"')
def step_store_write_result(context, outfile):
    _write_logger.debug("storing benchmark result")
    batches: List[BatchWriteMetrics] = getattr(context, "write_batches", [])
    meta = getattr(context, "write_benchmark_meta", {})
    summary = getattr(context, "write_summary", {})

    result: Dict[str, Any] = {
        "meta": meta,
        "summary": summary,
        "batches": [asdict(b) for b in batches],
        "created_at_epoch_s": time.time(),
    }

    write_json_report(
        outfile,
        result,
        logger_=_write_logger,
        log_prefix="Stored generic write benchmark result to ",
    )

    # PORT C10: KPI export to the MAIN InfluxDB removed
    # (bddbench: _export_write_result_to_main_influx(result, outfile, context)).
# <<< bddbench features/steps/influx_write_steps.py:344-367


# >>> bddbench features/steps/influx_write_steps.py:369-392 verbatim; logger -> _write_logger (store write context)
@then('I store the write benchmark context as "{outfile}"')
def step_store_write_context(context: Context, outfile: str) -> None:
    """
    Stores the subset of data that follow-up scenarios (e.g., query/delete) may need,
    without relying on in-memory `context` across scenarios.

    Payload:
      - write_benchmark_meta (includes measurement, total_points, etc.)
      - created_at_epoch_s
    """
    meta: Dict[str, Any] = getattr(context, "write_benchmark_meta", None)
    if not isinstance(meta, dict):
        raise AssertionError("No write benchmark meta present on context.write_benchmark_meta")

    payload: Dict[str, Any] = {
        "write_benchmark_meta": meta,
        "created_at_epoch_s": time.time(),
    }

    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_logger.info(f"storing write benchmark context: {out_path}")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
# <<< bddbench features/steps/influx_write_steps.py:369-392


# >>> bddbench features/steps/influx_query_steps.py:24-38 verbatim; logger -> _query_logger (query logger, QueryRunMetrics)
_query_logger = logging.getLogger("bddbench.influx_query_steps")


# ---------- Datatypes -----------

@dataclass
class QueryRunMetrics:
    client_id: int
    status_code: int
    ok: bool
    time_to_first_result_s: float | None
    total_time_s: float | None
    bytes_returned: int
    rows_returned: int

# <<< bddbench features/steps/influx_query_steps.py:24-38


# >>> bddbench features/steps/influx_query_steps.py:40-108 CHANGED C15,C16 (_build_flux_query)
# ---------- Helpers ----------

def _build_flux_query(
    bucket: str, measurement: str, time_range: str, query_type: str, result_size: str, run_id: Optional[str] = None,
    host: Optional[str] = None,  # PORT C15
) -> str:
    """
    Builds a simple Flux-Query depending on query_type and result_size.

    assumptions:
      - measurement-field "_measurement" gets set via filter
      - result_size small/large controls limit()
    """
    limit_small = 25  # PORT C16: pinned read limit (workload v3: limit(n: 25)); bddbench: 500
    limit_large = 50_000

    limit_n = limit_small if result_size == "small" else limit_large

    base = f'''
from(bucket: "{bucket}")
  |> range(start: -{time_range})
  |> filter(fn: (r) => r["_measurement"] == "{measurement}")
'''.strip()

    if run_id:
        base = f"""{base}
  |> filter(fn: (r) => r["run_id"] == "{run_id}")
""".strip()

    # PORT C15: the pinned read filters on the tag host (workload v3) instead of
    # bddbench's `exists r["value"]` (a column that unpivoted rows do not have).
    if query_type == "filter":
        flux = f"""{base}
  |> filter(fn: (r) => r["host"] == "{host}")
  |> limit(n: {limit_n})
"""
    elif query_type == "aggregate":
        flux = f"""{base}
  |> aggregateWindow(every: 10s, fn: mean, createEmpty: false)
  |> limit(n: {limit_n})
"""
    elif query_type == "group_by":
        flux = f"""{base}
  |> group(columns: ["device_id"])
  |> limit(n: {limit_n})
"""
    elif query_type == "pivot":
        flux = f"""{base}
  |> limit(n: {limit_n})
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
"""
    elif query_type == "join":
        flux = f"""
left = {base}
  |> filter(fn: (r) => r["_field"] == "value")
  |> limit(n: {limit_n})

right = {base}
  |> filter(fn: (r) => r["_field"] == "value")
  |> limit(n: {limit_n})

join(
  tables: {{left: left, right: right}},
  on: ["_time", "device_id"]
)
"""
    else:
        flux = f"""{base}
  |> limit(n: {limit_n})
"""

    return flux
# <<< bddbench features/steps/influx_query_steps.py:40-108


# >>> bddbench features/steps/influx_query_steps.py:111-284 verbatim (_run_single_query, _summarize_query_runs)
def _run_single_query(
    client_id: int,
    base_url: str,
    token: str,
    org: str,
    flux: str,
    output_format: str,
    compression: str,
) -> QueryRunMetrics:
    """
    Execute a Flux query and collect timing/size metrics
    for a single logical client.
    """
    enable_gzip = compression == "gzip"
    t_start = time.perf_counter()

    try:
        with InfluxDBClient(
            url=base_url,
            token=token,
            org=org,
            enable_gzip=enable_gzip,
            timeout=300_000,  # ms
        ) as client:
            query_api = client.query_api()

            bytes_returned = 0
            rows_returned = 0
            time_to_first: float | None = None

            if output_format == "csv":
                csv_iter = query_api.query_csv(query=flux, org=org)

                for row in csv_iter:
                    now = time.perf_counter()
                    if time_to_first is None:
                        time_to_first = now - t_start

                    if isinstance(row, str):
                        line = row
                    else:
                        line = ",".join(
                            "" if cell is None else str(cell) for cell in row
                        )

                    rows_returned += 1
                    bytes_returned += len((line + "\n").encode("utf-8"))

            else:
                tables = query_api.query(query=flux, org=org)
                for table in tables:
                    for record in table.records:
                        now = time.perf_counter()
                        if time_to_first is None:
                            time_to_first = now - t_start

                        rows_returned += 1
                        bytes_returned += (
                            len(
                                json.dumps(
                                    record.values,
                                    separators=(",", ":"),
                                    ensure_ascii=False,
                                ).encode("utf-8")
                            )
                            + 1
                        )

    except Exception:
        return QueryRunMetrics(
            client_id=client_id,
            status_code=0,
            ok=False,
            time_to_first_result_s=None,
            total_time_s=None,
            bytes_returned=0,
            rows_returned=0,
        )

    t_end = time.perf_counter()

    return QueryRunMetrics(
        client_id=client_id,
        # The python client raises on non-2xx; reaching here implies success.
        status_code=200,
        ok=True,
        time_to_first_result_s=time_to_first,
        total_time_s=t_end - t_start,
        bytes_returned=bytes_returned,
        rows_returned=rows_returned,
    )


def _summarize_query_runs(runs: List[QueryRunMetrics]) -> Dict[str, Any]:
    """Return an aggregated summary in the same shape that we export to MAIN."""
    if not runs:
        return {
            "total_runs": 0,
            "errors_count": 0,
            "error_rate": 0.0,
            "latency_stats": {},
            "bytes_stats": {},
            "rows_stats": {},
            "throughput": {},
        }

    def safe_vals(fn: str):
        vals = [getattr(r, fn) for r in runs if getattr(r, fn) is not None]
        return vals

    ttf = safe_vals("time_to_first_result_s")
    ttotal = safe_vals("total_time_s")
    bytes_vals = safe_vals("bytes_returned")
    rows_vals = safe_vals("rows_returned")

    error_rate = len([r for r in runs if not r.ok]) / len(runs) if runs else 0.0

    def agg(vals: List[float]):
        if not vals:
            return {"min": None, "max": None, "avg": None, "median": None}
        return {
            "min": min(vals),
            "max": max(vals),
            "avg": statistics.mean(vals),
            "median": statistics.median(vals),
        }

    duration_avg = statistics.mean(ttotal) if ttotal else None

    throughput_bytes_per_s = None
    throughput_rows_per_s = None
    if duration_avg and bytes_vals:
        throughput_bytes_per_s = statistics.mean(bytes_vals) / duration_avg
    if duration_avg and rows_vals:
        throughput_rows_per_s = statistics.mean(rows_vals) / duration_avg

    total_runs = len(runs)
    errors_count = len([r for r in runs if not r.ok])

    # parallel run -> wall duration ~= max(total_time_s)
    total_duration_s = max(ttotal) if ttotal else 0.0
    queries_count = total_runs

    throughput_queries_per_s = (queries_count / total_duration_s) if total_duration_s > 0 else None
    throughput_bytes_per_s = (sum(bytes_vals) / total_duration_s) if total_duration_s > 0 and bytes_vals else None
    throughput_rows_per_s = (sum(rows_vals) / total_duration_s) if total_duration_s > 0 and rows_vals else None

    ttf_stats = agg(ttf)
    total_stats = agg(ttotal)

    return {
        "queries_count": queries_count,
        "total_duration_s": total_duration_s,
        "total_runs": total_runs,
        "errors_count": errors_count,
        "error_rate": (errors_count / total_runs) if total_runs else 0.0,
        "latency_stats": {
            "ttf_min": ttf_stats["min"],
            "ttf_max": ttf_stats["max"],
            "ttf_avg": ttf_stats["avg"],
            "ttf_median": ttf_stats["median"],
            "total_min": total_stats["min"],
            "total_max": total_stats["max"],
            "total_avg": total_stats["avg"],
            "total_median": total_stats["median"],
        },
        "bytes_stats": agg(bytes_vals),
        "rows_stats": agg(rows_vals),
        "throughput": {
            "queries_per_s": throughput_queries_per_s,
            "bytes_per_s": throughput_bytes_per_s,
            "rows_per_s": throughput_rows_per_s,
        },
    }
# <<< bddbench features/steps/influx_query_steps.py:111-284


# >>> bddbench features/steps/influx_query_steps.py:395-462 CHANGED C15,C17; logger -> _query_logger (query step)
# ----------- Scenario Steps -------------

@when(
    'I run a generic query benchmark on measurement "{measurement}" '
    'with time range "{time_range}" using query type "{query_type}" on tag host "{host}" '
    'and result size "{result_size}" with {concurrent_clients:d} concurrent clients, '
    '{repeats:d} sequential repeats, '  # PORT C15/C17: host filter, sequential repeats
    'output format "{output_format}" and compression "{compression}"'
)
def step_run_query_benchmark(
    context,
    measurement,
    time_range,
    query_type,
    host,  # PORT C15
    result_size,
    concurrent_clients,
    repeats,  # PORT C17
    output_format,
    compression,
):
    """
    runs the same Flux-Query n times parallel from (concurrent_clients) and measures:
      - time_to_first_result
      - total_time
      - bytes/rows
      - error_rate
    """
    flux = _build_flux_query(
        context.influxdb.sut.bucket,
        measurement,
        time_range,
        query_type,
        result_size,
        run_id=getattr(context, "run_id", None),
        host=host,  # PORT C15
    )

    runs: List[QueryRunMetrics] = []
    with ThreadPoolExecutor(max_workers=concurrent_clients) as executor:
        futures = [
            executor.submit(
                _run_single_query,
                client_id=i,
                base_url=context.influxdb.sut.url,
                token=context.influxdb.sut.token,
                org=context.influxdb.sut.org,
                flux=flux,
                output_format=output_format,
                compression=compression,
            )
            # PORT C17: concurrent_clients x repeats queries; with 1 worker they run strictly
            # one after another (gold_live: 20 sequential reads). bddbench: range(concurrent_clients)
            for i in range(concurrent_clients * repeats)
        ]

        for fut in as_completed(futures):
            runs.append(fut.result())

    context.query_runs = runs
    context.query_benchmark_meta = {
        "run_id": getattr(context, "run_id", None),
        "measurement": measurement,
        "time_range": time_range,
        "query_type": query_type,
        "result_size": result_size,
        "concurrent_clients": concurrent_clients,
        "output_format": output_format,
        "compression": compression,
        "bucket": context.influxdb.sut.bucket,
        "org": context.influxdb.sut.org,
        "sut_url": context.influxdb.sut.url,
    }
    context.query_summary = _summarize_query_runs(runs)
# <<< bddbench features/steps/influx_query_steps.py:395-462


# >>> bddbench features/steps/influx_query_steps.py:465-485 CHANGED C10; logger -> _query_logger (store query result)
@then('I store the generic query benchmark result as "{outfile}"')
def step_store_query_result(context, outfile):
    runs: List[QueryRunMetrics] = getattr(context, "query_runs", [])
    meta = getattr(context, "query_benchmark_meta", {})
    summary = getattr(context, "query_summary", {})

    result = {
        "meta": meta,
        "summary": summary,
        "runs": [asdict(r) for r in runs],
        "created_at_epoch_s": time.time(),
    }

    write_json_report(
        outfile,
        result,
        logger_=_query_logger,
        log_prefix="Stored generic query benchmark result to ",
    )

    # PORT C10: KPI export to the MAIN InfluxDB removed
    # (bddbench: _export_query_result_to_main_influx(meta, summary, runs, outfile, context)).
# <<< bddbench features/steps/influx_query_steps.py:465-485


# >>> bddbench features/steps/influx_delete_steps.py:21-21 verbatim; logger -> _delete_logger (delete logger)
_delete_logger = logging.getLogger("bddbench.influx_delete_steps")
# <<< bddbench features/steps/influx_delete_steps.py:21-21


# >>> bddbench features/steps/influx_delete_steps.py:84-90 verbatim (DeleteRunMetrics)
@dataclass
class DeleteRunMetrics:
    latency_s: float
    status_code: int
    ok: bool
    points_before: int
    points_after: int
# <<< bddbench features/steps/influx_delete_steps.py:84-90


# >>> bddbench features/steps/influx_delete_steps.py:203-268 CHANGED C09,C18; logger -> _delete_logger (_delete_measurement_run)
def _delete_measurement_run(context: Context, measurement: str, host: str) -> DeleteRunMetrics:  # PORT C18
    """
    Delete points by (measurement + run_id). Requires context.write_benchmark_meta.
    Also stores context.delete_* so the reporting step can reuse it.
    """
    sut = context.influxdb.sut
    if not getattr(sut, "client", None):
        raise RuntimeError("SUT InfluxDB client is not configured on context.influxdb.sut")

    # PORT C18: no run scope - the pinned delete targets measurement + tag host; bddbench's
    # write-context guard _require_run_scoped_context would refuse exactly that.
    run_id, wb_meta = None, {}

    # PORT C09: count query before the delete removed (an extra POST /api/v2/query that
    # gold_live does not send); -1 = not counted, as in bddbench's bucket-wipe mode.
    points_before = -1

    delete_api = sut.client.delete_api()
    start = "1970-01-01T00:00:00Z"
    stop = "2100-01-01T00:00:00Z"
    predicate = f'_measurement="{measurement}" AND host="{host}"'  # PORT C18 (bddbench: run_id)

    t0 = time.perf_counter()
    _delete_logger.debug(f"t0 = {t0:.6f}s – starting delete for measurement={measurement} run_id={run_id}")
    try:
        delete_api.delete(start=start, stop=stop, predicate=predicate, bucket=sut.bucket, org=sut.org)
        ok = True
        status_code = 204
    except Exception as exc:
        _delete_logger.info(f" delete for measurement={measurement} run_id={run_id} failed: {exc}")
        ok = False
        status_code = 500

    t1 = time.perf_counter()
    latency_s = t1 - t0
    _delete_logger.debug(
        f"t1 = {t1:.6f}s – delete completed for measurement={measurement} run_id={run_id} "
        f"(latency_s={latency_s:.6f})"
    )

    points_after = -1  # PORT C09: count query after the delete removed

    expected_points = wb_meta.get("total_points") if wb_meta.get("measurement") == measurement else None

    context.delete_metrics = DeleteRunMetrics(
        latency_s=latency_s,
        status_code=status_code,
        ok=ok,
        points_before=points_before,
        points_after=points_after,
    )
    context.delete_benchmark_meta = {
        "measurement": measurement,
        "run_id": run_id,
        "bucket": sut.bucket,
        "org": sut.org,
        "sut_url": sut.url,
        "expected_points": expected_points,
    }
    context.delete_summary = {
        "latency_s": latency_s,
        "points_before": points_before,
        "points_after": points_after,
        "deleted_points": points_before - points_after,
        "expected_points": expected_points,
        "ok": ok,
        "status_code": status_code,
    }

    return context.delete_metrics
# <<< bddbench features/steps/influx_delete_steps.py:203-268


# >>> bddbench features/steps/influx_delete_steps.py:284-341 CHANGED C19; logger -> _delete_logger (delete step)
@when('I run the delete benchmark on measurement "{measurement}" for tag host "{host}"')
def step_run_delete_benchmark(context: Context, measurement: str, host: str) -> None:
    """
    PORT C19: replaces bddbench's single entry point (`When I run the delete benchmark`:
    smoke + load run-scoped deletes read from write-context files, each followed by a
    count check and a report). The pinned workload sends exactly one delete:
    measurement + tag host over 1970-01-01..2100-01-01 (bddbench's own range).
    """
    _delete_measurement_run(context, measurement, host)
# <<< bddbench features/steps/influx_delete_steps.py:284-341


# >>> bddbench features/steps/influx_delete_steps.py:361-389 CHANGED C10; logger -> _delete_logger (store delete result)
@then('I store the generic delete benchmark result as "{outfile}"')
def step_store_delete_result(context: Context, outfile: str) -> None:
    metrics: DeleteRunMetrics = getattr(context, "delete_metrics", None)
    meta: Dict[str, Any] = getattr(context, "delete_benchmark_meta", {})
    summary: Dict[str, Any] = getattr(context, "delete_summary", {})

    if metrics is None:
        raise AssertionError("No delete metrics recorded on context.delete_metrics")

    result: Dict[str, Any] = {
        "meta": meta,
        "summary": summary,
        "run": asdict(metrics),
        "created_at_epoch_s": time.time(),
    }

    write_json_report(
        outfile=outfile,
        data=result,
        logger_=_delete_logger,
        log_prefix="Stored generic delete benchmark result to ",
    )

    # PORT C10: KPI export to the MAIN InfluxDB removed
    # (bddbench: _export_delete_result_to_main_influx(context=..., meta=..., summary=..., outfile=...)).
# <<< bddbench features/steps/influx_delete_steps.py:361-389
