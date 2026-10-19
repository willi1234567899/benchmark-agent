"""Static checks of the bddbench_pinned fixture (phase 1: no SUT, no network, no install).

Nothing from the bddbench checkout is imported or executed (it is read as text). The only code
of the fixture that is executed is the pure function `_build_flux_query` (extracted by AST from
fixture/steps.py) to obtain the exact Flux text the port sends.

Checks
  1  build manifest: steps.py unchanged since the build, bddbench sources unchanged
  2  every `verbatim` block equals its bddbench line range (after the logger rename)
  3  compile() of steps.py
  4  behave Parser on both features (same call as the engine's validator)
  5  step matching: every (expanded) step matches exactly one definition of the right type
  6  the engine's static gate validate_artifacts(dry_run=False): parse + gherlint + compile
  7  the engine's behave --dry-run (_dry_run) with a stub influxdb_client on PYTHONPATH
     (binds every step; executes none) - for bddbench_pinned and gold_live
  8  request plan of both fixtures + offline workload conformance (the engine's
     evaluate_conformance on synthetic proxy samples built from the plans)

Usage (from the repo root):
  uv run --no-sync python -I <pinned>/tools/check_pinned_static.py <bddbench_src> <pinned> <repo_root>
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

import parse
from behave.parser import Parser, parse_file

MARK = re.compile(
    r"^# >>> bddbench (?P<file>\S+):(?P<a>\d+)-(?P<b>\d+) (?P<kind>verbatim|CHANGED \S+)"
    r"(?:; logger -> (?P<ren>\w+))? \(.*\)$"
)
FAILURES: list[str] = []


def check(ok: bool, label: str) -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {label}")
    if not ok:
        FAILURES.append(label)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def step_definitions(steps_path: Path) -> list[tuple[str, str, int]]:
    tree = ast.parse(steps_path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and getattr(dec.func, "id", "") in {"given", "when", "then", "step"}:
                    found.append((dec.func.id, ast.literal_eval(dec.args[0]), node.lineno))
    return found


def step_matching(feature_path: Path, steps_path: Path) -> None:
    defs = step_definitions(steps_path)
    patterns = [p for _, p, _ in defs]
    check(len(patterns) == len(set(patterns)), f"{steps_path.parent.name}: no duplicate step patterns ({len(defs)} definitions)")
    feature = parse_file(str(feature_path))
    used: set[str] = set()
    n_steps = 0
    for scenario in feature.walk_scenarios():
        steps = list(feature.background.steps if feature.background else []) + list(scenario.steps)
        for step in steps:
            n_steps += 1
            hits = [(kind, pat) for kind, pat, _ in defs if parse.compile(pat).parse(step.name) is not None]
            if len(hits) != 1:
                check(False, f"step {step.step_type} {step.name!r}: {len(hits)} matching definitions")
                continue
            kind, pat = hits[0]
            if kind not in (step.step_type, "step"):
                check(False, f"step {step.name!r}: keyword {step.step_type} but definition @{kind}")
            used.add(pat)
    unused = [p for p in patterns if p not in used]
    check(not any(f.startswith("step ") for f in FAILURES), f"{steps_path.parent.name}: {n_steps} expanded steps, each matches exactly one definition of its type")
    print(f"       definitions not used by this feature: {unused}")


def verbatim_blocks(src: Path, steps_path: Path) -> None:
    lines = steps_path.read_text(encoding="utf-8").splitlines()
    n_verbatim = n_changed = 0
    for i, line in enumerate(lines):
        m = MARK.match(line)
        if not m:
            continue
        a, b = int(m["a"]), int(m["b"])
        end = lines.index(f"# <<< bddbench {m['file']}:{a}-{b}", i)
        body = lines[i + 1:end]
        if m["kind"] != "verbatim":
            n_changed += 1
            continue
        n_verbatim += 1
        source = (src / m["file"]).read_text(encoding="utf-8").splitlines()[a - 1:b]
        if m["ren"]:
            source = [re.sub(r"\blogger\b", m["ren"], s) for s in source]
        check(body == source, f"verbatim {m['file']}:{a}-{b}")
    print(f"       {n_verbatim} verbatim blocks, {n_changed} CHANGED blocks (diff: changes.diff)")


def extract_function(steps_path: Path, name: str):
    tree = ast.parse(steps_path.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace: dict = {"Optional": Optional}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(steps_path), "exec"), namespace)  # noqa: S102
    return namespace[name]


def module_literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def examples_row(feature_path: Path, tag: str) -> dict[str, str]:
    feature = parse_file(str(feature_path))
    for outline in feature.scenarios:
        if tag in outline.tags:
            row = outline.examples[0].table.rows[0]
            return dict(zip(row.headings, row.cells, strict=True))
    raise KeyError(tag)


def gold_lines(now_ms: int) -> list[list[str]]:
    # tests/fixtures/gold_live/steps.py:52-64 (same formula)
    total, size = 1000, 100
    return [[f"benchmark,source=gold,host=h{i % 5 + 1} value={b * size + i}i {now_ms - (total - b * size - i) * 1000}"
             for i in range(size)] for b in range(10)]


def port_lines(now_ms: int, row: dict[str, str]) -> list[list[str]]:
    # build_benchmark_point (PORT C11-C13) + influxdb-client Point.to_line_protocol: tags sorted
    # by key (client/write/point.py:_append_tags), int field -> "<n>i", int time verbatim.
    size, batches, writers = int(row["batch_size"]), int(row["batches"]), int(row["parallel_writers"])
    card = int(row["tag_cardinality"])
    base = now_ms - writers * batches * size * 1000
    out = []
    for b in range(writers * batches):
        out.append([f"{row['measurement']},host=h{(b * size + i) % card + 1},source=gold value={b * size + i}i {base + (b * size + i) * 1000}"
                    for i in range(size)])
    return out


def semantic(line: str) -> tuple:
    head, field, ts = line.split(" ")
    measurement, *tags = head.split(",")
    return measurement, frozenset(tags), field, ts


def main() -> None:
    src, pinned, repo = (Path(p).resolve() for p in sys.argv[1:4])
    sys.path.insert(0, str(repo / "src"))
    from benchmark_agent.config import _load_yaml_model
    from benchmark_agent.evaluation.conformance import evaluate_conformance
    from benchmark_agent.execution import validator
    from benchmark_agent.models import (
        OperationWindowStats,
        ProxyWindowStats,
        RequestSample,
        WorkloadSpec,
    )

    fx = pinned / "fixture"
    gold = repo / "tests" / "fixtures" / "gold_live"
    steps, feature = fx / "steps.py", fx / "benchmark.feature"

    print("1 build manifest")
    manifest = json.loads((pinned / "build_manifest.json").read_text(encoding="utf-8"))
    check(manifest["outputs_sha256"]["fixture/steps.py"] == sha256(steps), "steps.py == built output (no hand edits)")
    for name, digest in manifest["source_sha256"].items():
        check(sha256(src / name) == digest, f"source unchanged: {name}")

    print("2 verbatim blocks")
    verbatim_blocks(src, steps)

    print("3 compile")
    try:
        compile(steps.read_text(encoding="utf-8"), str(steps), "exec")
        check(True, "compile(steps.py)")
    except SyntaxError as error:
        check(False, f"compile(steps.py): {error}")

    print("4 behave Parser")
    for fp in (feature, gold / "benchmark.feature"):
        try:
            Parser().parse(fp.read_text(encoding="utf-8"), filename=str(fp))
            check(True, f"behave Parser: {fp.parent.name}/{fp.name}")
        except Exception as error:  # noqa: BLE001
            check(False, f"behave Parser {fp}: {error}")

    print("5 step matching")
    step_matching(feature, steps)
    step_matching(gold / "benchmark.feature", gold / "steps.py")

    print("6 engine static gate validate_artifacts(dry_run=False)")
    for fdir in (fx, gold):
        result = validator.validate_artifacts(fdir / "benchmark.feature", fdir / "steps.py", dry_run=False)
        check(result.syntax_ok, f"{fdir.name}: syntax_ok={result.syntax_ok} issues={[i.code for i in result.issues]}")
        print(f"       gherlint warnings: {[f'{w.code}: {w.message}' for w in result.warnings]}")

    print("7 engine behave --dry-run with stub influxdb_client (binds, executes nothing)")
    stub = pinned / "tools" / "dryrun_stub"
    for fdir in (fx, gold):
        issues = validator._dry_run(fdir / "benchmark.feature", fdir / "steps.py", env={"PYTHONPATH": str(stub)})  # noqa: SLF001
        check(not issues, f"{fdir.name}: dry-run issues={[f'{i.code}: {i.message}' for i in issues]}")

    print("8 request plan + offline conformance")
    workload = _load_yaml_model(repo / "config" / "workload.yaml", WorkloadSpec)
    w_row, q_row, d_row = (examples_row(feature, t) for t in ("write", "query", "delete"))
    now_ms = 1_760_000_000_000
    g_batches, p_batches = gold_lines(now_ms), port_lines(now_ms, w_row)
    check(len(g_batches) == len(p_batches) == 10 and all(len(b) == 100 for b in p_batches), "write: 10 batches x 100 points in both")
    check([[semantic(x) for x in b] for b in g_batches] == [[semantic(x) for x in b] for b in p_batches],
          "write: same points (measurement, tag set, field, ms timestamp) in the same order, batch by batch")
    check([len("\n".join(b)) for b in g_batches] == [len("\n".join(b)) for b in p_batches],
          f"write: same body length per batch ({len(chr(10).join(g_batches[0]))} B in batch 1)")
    print(f"       gold line 1: {g_batches[0][0]}\n       port line 1: {p_batches[0][0]}")

    flux_port = extract_function(steps, "_build_flux_query")(
        "benchmark", q_row["measurement"], q_row["time_range"], q_row["query_type"], q_row["result_size"],
        run_id=None, host=q_row["host"])
    flux_gold = module_literal(gold / "steps.py", "_QUERY").format(bucket="benchmark")
    query_body_port = json.dumps({
        "extern": {"imports": [], "body": []},
        "query": flux_port,
        "dialect": {"header": True, "delimiter": ",", "annotations": ["datatype", "group", "default"],
                    "commentPrefix": "#", "dateTimeFormat": "RFC3339"},
    })
    print("       gold flux: " + flux_gold)
    print("       port flux: " + " ".join(flux_port.split()))
    check(int(q_row["concurrent_clients"]) * int(q_row["repeats"]) == 20 and q_row["concurrent_clients"] == "1",
          "read: 1 worker x 20 queries -> 20 sequential requests (gold: 20 sequential)")

    delete_body_gold = json.dumps(module_literal(gold / "steps.py", "_DELETE_BODY"))
    delete_body_port = json.dumps({"start": "1970-01-01T00:00:00Z", "stop": "2100-01-01T00:00:00Z",
                                   "predicate": f'_measurement="{d_row["measurement"]}" AND host="{d_row["host"]}"'})
    check(delete_body_gold == delete_body_port, f"delete: identical JSON body {delete_body_port}")

    def stats(read_bodies: list[str], delete_body: str) -> ProxyWindowStats:
        samples = [RequestSample(operation="write", method="POST", status_code=204, latency_ms=1.0, error=False,
                                 path="/api/v2/write?org=benchmark&bucket=benchmark&precision=ms", n_points=100)
                   for _ in range(10)]
        samples += [RequestSample(operation="read", method="POST", status_code=200, latency_ms=1.0, error=False,
                                  path="/api/v2/query?org=benchmark", body_excerpt=b) for b in read_bodies]
        samples.append(RequestSample(operation="delete", method="POST", status_code=204, latency_ms=1.0, error=False,
                                     path="/api/v2/delete?org=benchmark&bucket=benchmark", body_excerpt=delete_body))
        by_op = {"write": OperationWindowStats(n_requests=10, n_points=1000),
                 "read": OperationWindowStats(n_requests=len(read_bodies)),
                 "delete": OperationWindowStats(n_requests=1)}
        return ProxyWindowStats(window_duration_s=1.0, by_operation=by_op, requests=samples)

    for label, read_body, delete_body in (("gold_live", flux_gold, delete_body_gold),
                                          ("bddbench_pinned", query_body_port, delete_body_port)):
        result = evaluate_conformance(workload, stats([read_body] * 20, delete_body))
        check(result.ok and not result.deviations,
              f"{label}: conformance ok={result.ok} spec_coverage={result.spec_coverage} deviations={result.deviations}")
        print("       " + "; ".join(f"{c.operation}/{c.name}:{'ok' if c.ok else 'FAIL'}" for c in result.checks))
    print(f"       read body bytes: gold {len(flux_gold.encode())} (application/vnd.flux), "
          f"port {len(query_body_port.encode())} (application/json, < 4000 excerpt limit)")

    print()
    print("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
