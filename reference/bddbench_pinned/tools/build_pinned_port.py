"""Build fixture/steps.py of the bddbench port onto the PINNED workload (influx passthrough).

The bddbench checkout is read purely as TEXT (never imported, never executed). steps.py is the
concatenation of

- PORT-ONLY blocks (header, merged imports, two tiny helpers) - marked `# >>> port-only`;
- bddbench line ranges, either VERBATIM or CHANGED by the PORT Cxx edits in CHANGES below
  (each edit replaces an exact original line range whose first line is asserted, so an
  off-by-one fails the build). Markers: `# >>> bddbench <file>:<a>-<b> verbatim|CHANGED ...`.

Outputs (next to the fixture): steps.py, changes.diff (unified diff of every CHANGED block
against its original range), build_manifest.json (sha256 of every source and output file,
sections, changes with original file:line).

Usage (from anywhere, isolated mode; the bddbench dir is passed as data):
    python -I build_pinned_port.py <bddbench_src_dir> <bddbench_pinned_dir>
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

COMMIT = "42ddad598eba29481f66f37c2ce3abff75db3e43"
REPO = "https://github.com/DPS25/bddbench"

U = "src/utils.py"
E = "features/environment.py"
G = "features/steps/influx_global_endpoint_steps.py"
W = "features/steps/influx_write_steps.py"
Q = "features/steps/influx_query_steps.py"
D = "features/steps/influx_delete_steps.py"
SOURCE_FILES = [U, E, G, W, Q, D,
                "features/influx_write_benchmark.feature",
                "features/influx_query_benchmark.feature",
                "features/influx_delete_benchmark.feature"]

HEADER = '''"""DPS25/bddbench (commit 42ddad5) write / query / delete, ported onto the PINNED workload v3
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
'''

# Output order. ("src", file, first, last, logger_rename | None, title) or ("port", text, title).
SECTIONS: list[tuple] = [
    ("port", HEADER, "header, merged imports, PORT C02/C13 helpers"),
    ("src", U, 230, 242, None, "influx_precision_from_str"),
    ("src", U, 245, 254, None, "base_timestamp_for_precision"),
    ("src", U, 257, 301, None, "build_benchmark_point"),
    ("src", U, 382, 407, None, "write_json_report"),
    ("src", E, 14, 25, "_env_logger", "environment logger, _env_truthy, _env_strip"),
    ("src", E, 75, 202, "_env_logger", "_load_env"),
    ("src", E, 231, 237, None, "_ensure_influx_initialized"),
    ("src", G, 8, 24, None, "Background steps"),
    ("src", W, 28, 47, "_write_logger", "write logger, BatchWriteMetrics, _maybe_cleanup_before_run"),
    ("src", W, 132, 202, None, "_run_writer_worker"),
    ("src", W, 205, 341, "_write_logger", "write step"),
    ("src", W, 344, 367, "_write_logger", "store write result"),
    ("src", W, 369, 392, "_write_logger", "store write context"),
    ("src", Q, 24, 38, "_query_logger", "query logger, QueryRunMetrics"),
    ("src", Q, 40, 108, None, "_build_flux_query"),
    ("src", Q, 111, 284, None, "_run_single_query, _summarize_query_runs"),
    ("src", Q, 395, 462, "_query_logger", "query step"),
    ("src", Q, 465, 485, "_query_logger", "store query result"),
    ("src", D, 21, 21, "_delete_logger", "delete logger"),
    ("src", D, 84, 90, None, "DeleteRunMetrics"),
    ("src", D, 203, 268, "_delete_logger", "_delete_measurement_run"),
    ("src", D, 284, 341, "_delete_logger", "delete step"),
    ("src", D, 361, 389, "_delete_logger", "store delete result"),
]

# (id, file, first, last, expected stripped text of the first original line, new text, what)
# New text is written with the original name `logger`; the section's rename applies after.
CHANGES: list[tuple[str, str, int, int, str, str, str]] = [
    ("C13", U, 283, 283, "ts = base_ts + idx",
     "        ts = base_ts + idx * _ONE_SECOND[precision]  # PORT C13: 1 s apart, as gold_live\n",
     "in-order spacing 1 time unit -> 1 s"),
    ("C11", U, 285, 285, 'p = Point(measurement).tag("device_id", f"dev-{device_id}")',
     "    # PORT C11: workload tag host=h1..h5 (cycled by idx) instead of device_id=dev-N; constant\n"
     "    # marker tag source=gold as in gold_live, so the line protocol has the same bytes.\n"
     '    p = Point(measurement).tag("source", "gold").tag("host", f"h{device_id + 1}")\n',
     "tags device_id=dev-N -> host=h(N+1) + constant source=gold"),
    ("C12", U, 289, 289, 'p = p.field("value", float(idx)).field("seq", idx)',
     '    p = p.field("value", idx)  # PORT C12: one integer field value (gold_live: value=<n>i)\n',
     "fields value(float)+seq(int) -> value(int)"),
    ("C04", E, 90, 145, "# ----- main influx DB -----",
     "    # PORT C04: MAIN (KPI) InfluxDB initialisation removed (bddbench environment.py:90-145):\n"
     "    # no second InfluxDB, no KPI export; its default URL http://localhost:8086 is the\n"
     "    # engine's measurement proxy.\n",
     "MAIN KPI database init removed"),
    ("C05", E, 175, 183, "if not context.influxdb.sut.client.ping():",
     "    # PORT C05: removed (bddbench environment.py:175-183): client.ping() (GET /ping),\n"
     "    # _validate_influx_auth (GET /api/v2/buckets), _validate_bucket_exists\n"
     "    # (GET /api/v2/buckets?name=...) and the 'authenticated' log line - requests that\n"
     "    # gold_live does not send inside the measurement window.\n",
     "ping/auth/bucket requests removed"),
    ("C06", E, 196, 202, 'context.influxdb.sut.host = _run_on_sut(["hostname"]).stdout.strip()',
     "    # PORT C06: removed (bddbench environment.py:196-202): _run_on_sut([\"hostname\"]) (ssh\n"
     "    # to the SUT host) and client.health() (GET /health); they only fed KPI-export tags.\n",
     "ssh hostname + GET /health removed"),
    ("C07", G, 8, 12, '@given("a SUT InfluxDB v2 endpoint is configured and reachable")',
     '@given("a SUT InfluxDB v2 endpoint is configured")\n'
     "def step_bucket_from_env(context: Context):\n"
     "    # PORT C07: bddbench text \"... is configured and reachable\"; its reachability check\n"
     "    # client.ping() (GET /ping) is removed - gold_live sends no such request.\n"
     "    # PORT C01: the engine stages no environment.py, so the SUT part of bddbench's\n"
     "    # before_all (environment.py:290 -> _ensure_influx_initialized -> _load_env) runs\n"
     "    # here, after PORT C02 maps the engine's variables onto bddbench's names.\n"
     "    _map_engine_env_to_bddbench()\n"
     "    _ensure_influx_initialized(context)\n",
     "Background step 1: text + ping removed, client init moved here"),
    ("C08", G, 15, 15, '@given("the target bucket from the SUT config is available")',
     '@given("the target bucket from the SUT config is set")  # PORT C08 (bddbench: "is available")\n',
     "Background step 2: text"),
    ("C08", G, 20, 24, "bucket_api = context.influxdb.sut.client.buckets_api()",
     "    # PORT C08: bucket lookup buckets_api().find_bucket_by_name (GET /api/v2/buckets?name=...)\n"
     "    # removed - gold_live sends no such request.\n",
     "Background step 2: bucket lookup removed"),
    ("C13", W, 257, 257, "base_ts = base_timestamp_for_precision(precision_enum)",
     "    # PORT C13: anchor the in-order series so that its last point lies 1 s before now and the\n"
     "    # points are 1 s apart, exactly as gold_live (tests/fixtures/gold_live/steps.py:52,62).\n"
     "    base_ts = base_timestamp_for_precision(precision_enum) - (\n"
     "        parallel_writers * batches * batch_size * _ONE_SECOND[precision_enum]\n"
     "    )\n",
     "base timestamp anchored total_points x 1 s before now"),
    ("C10", W, 367, 367, "_export_write_result_to_main_influx(result, outfile, context)",
     "    # PORT C10: KPI export to the MAIN InfluxDB removed\n"
     "    # (bddbench: _export_write_result_to_main_influx(result, outfile, context)).\n",
     "write KPI export call removed"),
    ("C15", Q, 43, 43,
     "bucket: str, measurement: str, time_range: str, query_type: str, result_size: str, run_id: Optional[str] = None",
     "    bucket: str, measurement: str, time_range: str, query_type: str, result_size: str, run_id: Optional[str] = None,\n"
     "    host: Optional[str] = None,  # PORT C15\n",
     "_build_flux_query: host parameter"),
    ("C16", Q, 52, 52, "limit_small = 500",
     "    limit_small = 25  # PORT C16: pinned read limit (workload v3: limit(n: 25)); bddbench: 500\n",
     "result size small: limit 500 -> 25"),
    ("C15", Q, 68, 70, 'if query_type == "filter":',
     "    # PORT C15: the pinned read filters on the tag host (workload v3) instead of\n"
     "    # bddbench's `exists r[\"value\"]` (a column that unpivoted rows do not have).\n"
     '    if query_type == "filter":\n'
     '        flux = f"""{base}\n'
     '  |> filter(fn: (r) => r["host"] == "{host}")\n',
     "filter query: exists r[\"value\"] -> r[\"host\"] == host"),
    ("C17", Q, 399, 400, """'with time range "{time_range}" using query type "{query_type}" '""",
     """    'with time range "{time_range}" using query type "{query_type}" on tag host "{host}" '\n"""
     """    'and result size "{result_size}" with {concurrent_clients:d} concurrent clients, '\n"""
     """    '{repeats:d} sequential repeats, '  # PORT C15/C17: host filter, sequential repeats\n""",
     "query step text: + host, + repeats"),
    ("C17", Q, 407, 409, "query_type,",
     "    query_type,\n"
     "    host,  # PORT C15\n"
     "    result_size,\n"
     "    concurrent_clients,\n"
     "    repeats,  # PORT C17\n",
     "query step parameters: + host, + repeats"),
    ("C15", Q, 426, 426, 'run_id=getattr(context, "run_id", None),',
     '        run_id=getattr(context, "run_id", None),\n'
     "        host=host,  # PORT C15\n",
     "host passed to _build_flux_query"),
    ("C17", Q, 442, 442, "for i in range(concurrent_clients)",
     "            # PORT C17: concurrent_clients x repeats queries; with 1 worker they run strictly\n"
     "            # one after another (gold_live: 20 sequential reads). bddbench: range(concurrent_clients)\n"
     "            for i in range(concurrent_clients * repeats)\n",
     "number of queries: concurrent_clients -> concurrent_clients x repeats"),
    ("C10", Q, 485, 485, "_export_query_result_to_main_influx(meta, summary, runs, outfile, context)",
     "    # PORT C10: KPI export to the MAIN InfluxDB removed\n"
     "    # (bddbench: _export_query_result_to_main_influx(meta, summary, runs, outfile, context)).\n",
     "query KPI export call removed"),
    ("C18", D, 203, 203, "def _delete_measurement_run(context: Context, measurement: str) -> DeleteRunMetrics:",
     "def _delete_measurement_run(context: Context, measurement: str, host: str) -> DeleteRunMetrics:  # PORT C18\n",
     "_delete_measurement_run: host parameter"),
    ("C18", D, 212, 212, "run_id, wb_meta = _require_run_scoped_context(context, measurement)",
     "    # PORT C18: no run scope - the pinned delete targets measurement + tag host; bddbench's\n"
     "    # write-context guard _require_run_scoped_context would refuse exactly that.\n"
     "    run_id, wb_meta = None, {}\n",
     "run-scope guard removed"),
    ("C09", D, 214, 214, "points_before = _count_points_for_measurement(context, measurement, run_id=run_id)",
     "    # PORT C09: count query before the delete removed (an extra POST /api/v2/query that\n"
     "    # gold_live does not send); -1 = not counted, as in bddbench's bucket-wipe mode.\n"
     "    points_before = -1\n",
     "count query before delete removed"),
    ("C18", D, 219, 219, """predicate = f'_measurement="{measurement}" AND run_id="{run_id}"'""",
     """    predicate = f'_measurement="{measurement}" AND host="{host}"'  # PORT C18 (bddbench: run_id)\n""",
     "delete predicate run_id -> host"),
    ("C09", D, 239, 239, "points_after = _count_points_for_measurement(context, measurement, run_id=run_id)",
     "    points_after = -1  # PORT C09: count query after the delete removed\n",
     "count query after delete removed"),
    ("C19", D, 284, 341, '@when("I run the delete benchmark")',
     '@when(\'I run the delete benchmark on measurement "{measurement}" for tag host "{host}"\')\n'
     "def step_run_delete_benchmark(context: Context, measurement: str, host: str) -> None:\n"
     '    """\n'
     "    PORT C19: replaces bddbench's single entry point (`When I run the delete benchmark`:\n"
     "    smoke + load run-scoped deletes read from write-context files, each followed by a\n"
     "    count check and a report). The pinned workload sends exactly one delete:\n"
     "    measurement + tag host over 1970-01-01..2100-01-01 (bddbench's own range).\n"
     '    """\n'
     "    _delete_measurement_run(context, measurement, host)\n",
     "delete step: one pinned delete instead of smoke+load"),
    ("C10", D, 384, 389, "_export_delete_result_to_main_influx(",
     "    # PORT C10: KPI export to the MAIN InfluxDB removed\n"
     "    # (bddbench: _export_delete_result_to_main_influx(context=..., meta=..., summary=..., outfile=...)).\n",
     "delete KPI export call removed"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rename(lines: list[str], target: str | None) -> list[str]:
    if not target:
        return lines
    return [re.sub(r"\blogger\b", target, line) for line in lines]


def main() -> None:
    src_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    sources = {name: (src_dir / name).read_text(encoding="utf-8").splitlines() for name in SOURCE_FILES}

    used: set[int] = set()
    out: list[str] = []
    diffs: list[str] = []
    manifest_sections = []
    for section in SECTIONS:
        if section[0] == "port":
            _, text, title = section
            out.append(f"# >>> port-only: {title}")
            out.extend(text.rstrip("\n").splitlines())
            out.append("# <<< port-only")
            out.append("")
            out.append("")
            manifest_sections.append({"kind": "port-only", "title": title})
            continue
        _, name, first, last, logger_rename, title = section
        original = sources[name][first - 1:last]
        # Changes inside this section, applied bottom-up on the original line numbers.
        hits = [(i, c) for i, c in enumerate(CHANGES) if c[1] == name and first <= c[2] and c[3] <= last]
        ported = list(original)
        for index, (cid, _f, a, b, expect, new_text, _what) in sorted(hits, key=lambda h: -h[1][2]):
            actual = sources[name][a - 1].strip()
            if actual != expect.strip():
                raise SystemExit(f"{cid} {name}:{a}: expected {expect!r}, found {actual!r}")
            ported[a - first:b - first + 1] = new_text.rstrip("\n").splitlines()
            used.add(index)
        ids = sorted({c[0] for _, c in hits})
        kind = "CHANGED " + ",".join(ids) if ids else "verbatim"
        suffix = f"; logger -> {logger_rename}" if logger_rename else ""
        tag = f"bddbench {name}:{first}-{last}"
        out.append(f"# >>> {tag} {kind}{suffix} ({title})")
        out.extend(rename(ported, logger_rename))
        out.append(f"# <<< {tag}")
        out.append("")
        out.append("")
        if ids:
            diffs.extend(difflib.unified_diff(
                rename(original, logger_rename), rename(ported, logger_rename),
                fromfile=f"bddbench/{name}:{first}-{last} (logger renamed)" if logger_rename else f"bddbench/{name}:{first}-{last}",
                tofile=f"fixture/steps.py [{tag}]", lineterm="", n=2))
            diffs.append("")
        manifest_sections.append({"kind": "changed" if ids else "verbatim", "file": name,
                                  "lines": [first, last], "logger_rename": logger_rename,
                                  "title": title, "changes": ids})
    unused = [CHANGES[i][0] + f" {CHANGES[i][1]}:{CHANGES[i][2]}" for i in range(len(CHANGES)) if i not in used]
    if unused:
        raise SystemExit(f"changes outside every section: {unused}")

    steps_path = out_dir / "fixture" / "steps.py"
    steps_path.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8", newline="\n")
    diff_path = out_dir / "changes.diff"
    diff_path.write_text("\n".join(diffs) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "repo": REPO,
        "commit": COMMIT,
        "source_sha256": {name: sha256(src_dir / name) for name in SOURCE_FILES},
        "sections": manifest_sections,
        "changes": [{"id": c[0], "file": c[1], "original_lines": [c[2], c[3]], "what": c[6]} for c in CHANGES],
        "outputs_sha256": {"fixture/steps.py": sha256(steps_path), "changes.diff": sha256(diff_path)},
    }
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {steps_path} ({len(out)} lines), {diff_path}, build_manifest.json")


if __name__ == "__main__":
    main()
