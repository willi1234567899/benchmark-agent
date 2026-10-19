"""Run reference fixtures N times, interleaved rep-major, through the real measurement engine.

Adapted from bddbench_fixture_draft/tools/run_reference_series.py (itself a generalisation of
scripts/run_gold_ceiling.py: same WorkspaceManager run-NNN directories, same
EvaluationRunner.run_artifacts(start_sut=True) call, i.e. a fresh InfluxDB per run, seeded
outside the measurement window). PHASE 1: prepared, NEVER executed.

What it adds over the draft
- item syntax api:epoch:fixture_dir:condition_name - the condition name is set by the runner
  (gold_live is run as `_ref_live`, never with a `_gold_` prefix: thesis_results.py,
  thesis_stage_b_client.py and thesis_dataset_export.py treat `_gold_*` as the E12-E17 ceiling);
- epoch guards: every item has its own epoch, epoch >= 18, never 8, 9, 12-17, and an epoch that
  already occurs in the configured artifacts dir is refused (unless --resume for the same
  condition);
- exclusivity pre-flight: `docker ps` must work and show no container publishing 8086/8087 or
  belonging to this checkout's compose project; 127.0.0.1:8086 and :8087 must be free; the
  configured metrics.csv must not have been written in the last 120 s (the engine rewrites the
  whole CSV per run, a concurrent writer would lose rows);
- the engine's own static gate (validate_artifacts, incl. behave --dry-run, i.e. the import of
  influxdb_client) on every item BEFORE the SUT is started, and an influxdb_client import
  re-check before every bddbench run (a concurrent exact `uv sync` would remove the group);
- --results-root guard: the resolved results_artifacts_dir and results_raw_dir must lie under
  it (pilot into the scratchpad cannot write into the repo by accident);
- the SUT is always stopped at the end (finally: SutManager.down(), also after a failure).

Usage (PowerShell, from the repo root; exact phase-2 commands: PROVENANCE.md section 9):
    uv run --group bddbench python <pinned>/tools/run_reference_series.py --repetitions 10 `
        --results-root results `
        --item influx:18:<pinned>/fixture:_bddbench_pinned `
        --item influx:19:tests/fixtures/gold_live:_ref_live
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

_APIS = ("influx", "kestrel", "tessera")
_MIN_EPOCH = 18
_FORBIDDEN_EPOCHS = {8, 9, 12, 13, 14, 15, 16, 17}
_PORTS = (8086, 8087)
_CSV_IDLE_SECONDS = 120


def main() -> None:
    args = _parse_args()
    project_root = Path.cwd().resolve()
    if not (project_root / "pyproject.toml").is_file() or not (project_root / "src" / "benchmark_agent").is_dir():
        raise SystemExit(f"run from the Benchmark-Agent repo root (cwd={project_root})")
    items = [_parse_item(raw, project_root) for raw in args.item]
    _check_items(items)

    from benchmark_agent.config import load_app_config
    from benchmark_agent.execution import SutManager, validate_artifacts
    from benchmark_agent.execution.sut import compose_project_name, sut_environment
    from benchmark_agent.orchestration.evaluation_runner import EvaluationRunner
    from benchmark_agent.orchestration.workspace import WorkspaceManager

    # ---- pre-flight (nothing is started before all of this passed) ----
    results_root = Path(args.results_root)
    results_root = (results_root if results_root.is_absolute() else project_root / results_root).resolve()
    configs = {}
    for item in items:
        _apply_overrides(item)
        config = load_app_config(project_root)
        if config.sut.api != item["api"] or config.defaults.epoch != item["epoch"]:
            raise SystemExit(f"env override not applied for {item['condition']}: "
                             f"api={config.sut.api} epoch={config.defaults.epoch}")
        for label, path in (("artifacts", config.paths.results_artifacts_dir), ("raw", config.paths.results_raw_dir)):
            if not _is_under(path.resolve(), results_root):
                raise SystemExit(f"{label} dir {path} is not under --results-root {results_root}")
        configs[item["condition"]] = config
        print(f"item {item['condition']}: api={config.sut.api} epoch={config.defaults.epoch} "
              f"fixture={item['fixture']}\n      artifacts={config.paths.results_artifacts_dir}\n"
              f"      csv={config.paths.results_raw_dir / 'metrics.csv'}")
    first_config = configs[items[0]["condition"]]
    _check_epoch_unused(items, first_config.paths.results_artifacts_dir, resume=args.resume)
    _check_exclusive(compose_project_name(project_root))
    _check_csv_idle(first_config.paths.results_raw_dir / "metrics.csv", ignore=args.ignore_recent_csv)
    for item in items:
        config = configs[item["condition"]]
        env = sut_environment(config.sut, influx_url=config.sut.influxdb_url)
        gate = validate_artifacts(item["fixture"] / "benchmark.feature", item["fixture"] / "steps.py", env=env)
        if not gate.syntax_ok:
            raise SystemExit(f"static gate failed for {item['condition']}: "
                             f"{[f'{i.code}: {i.message}' for i in gate.issues]}")
        print(f"static gate ok: {item['condition']} (warnings: {len(gate.warnings)})")
    if args.preflight_only:
        print("pre-flight passed; --preflight-only: nothing started")
        return

    # ---- series ----
    summary: dict[str, list[str]] = {item["condition"]: [] for item in items}
    try:
        for rep in range(1, args.repetitions + 1):
            for position, item in enumerate(items):
                if item["needs_influxdb_client"] and importlib.util.find_spec("influxdb_client") is None:
                    raise SystemExit("influxdb_client vanished from the venv (concurrent `uv sync`?) - series stopped")
                _apply_overrides(item)
                evaluator = EvaluationRunner(project_root)  # loads the config with the overrides
                config = evaluator._app_config  # noqa: SLF001 - the object run_artifacts uses
                if config.sut.api != item["api"] or config.defaults.epoch != item["epoch"]:
                    raise SystemExit(f"override lost for {item['condition']}")
                run_directory = WorkspaceManager(config.paths.results_artifacts_dir).create_run_directory(item["condition"])
                for name in ("benchmark.feature", "steps.py"):
                    shutil.copyfile(item["fixture"] / name, run_directory / name)
                metadata = dict(item["metadata"])
                metadata["condition"] = {**metadata["condition"], "name": item["condition"], "guidance_level": None}
                metadata["epoch"] = item["epoch"]
                metadata["sut_api"] = item["api"]
                metadata["repetition"] = rep
                metadata["series"] = {
                    "runner": "bddbench_pinned/tools/run_reference_series.py",
                    "repetition": rep, "of": args.repetitions, "interleave_position": position,
                    "items": [i["condition"] for i in items],
                    "source_fixture": str(item["fixture"]),
                    "source_condition_name": item["source_condition"],
                    "sha256": {name: _sha256(run_directory / name) for name in ("benchmark.feature", "steps.py")},
                }
                (run_directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                started = time.monotonic()
                result = evaluator.run_artifacts(run_directory, start_sut=True, stop_sut=False, enable_measurement=True)
                line = _line(rep, args.repetitions, run_directory, result, time.monotonic() - started)
                summary[item["condition"]].append(f"{result.status}/{_semantics(result)}")
                print(line, flush=True)
    finally:
        try:
            SutManager(project_root, first_config.sut).down()
            print("SUT stopped (docker compose down --volumes)")
        except Exception as error:  # noqa: BLE001 - report, do not mask the original error
            print(f"WARNING: SUT teardown failed: {error}", file=sys.stderr)
        for condition, outcomes in summary.items():
            print(f"summary {condition}: {len(outcomes)} runs, "
                  f"ok+semantics: {sum(1 for o in outcomes if o == 'ok/True')} -> {outcomes}")


def _apply_overrides(item: dict) -> None:
    os.environ["BENCHMARK_AGENT_SUT__API"] = item["api"]
    os.environ["BENCHMARK_AGENT_EPOCH"] = str(item["epoch"])


def _semantics(result) -> str:  # noqa: ANN001
    return "-" if result.success is None else str(result.success.semantics_ok)


def _line(rep: int, total: int, run_directory: Path, result, seconds: float) -> str:  # noqa: ANN001
    parts = [f"rep {rep:>2}/{total}", f"{run_directory.parent.name}/{run_directory.name}",
             f"status={result.status}", f"semantics_ok={_semantics(result)}", f"{seconds:.0f}s"]
    stats = result.proxy_stats
    if stats is not None:
        for op in ("write", "read", "delete"):
            s = stats.by_operation.get(op)
            if s is not None and s.n_requests:
                parts.append(f"{op}: n={s.n_requests} err={s.error_count} pts={s.n_points} "
                             f"tp={s.throughput_ops_s:.1f}/s p50={s.latency_p50_ms or 0:.1f}ms")
        parts.append(f"unmapped={stats.unmapped_requests}")
    if result.conformance is not None:
        parts.append(f"conformance={result.conformance.ok} deviations={result.conformance.deviations}")
    return "  ".join(parts)


def _parse_item(raw: str, project_root: Path) -> dict:
    try:
        api, epoch, rest = raw.split(":", 2)
        fixture, condition = rest.rsplit(":", 1)
    except ValueError:
        raise SystemExit(f"item {raw!r}: expected api:epoch:fixture_dir:condition_name") from None
    if api not in _APIS:
        raise SystemExit(f"unknown api {api!r} in {raw!r}")
    path = Path(fixture)
    path = (path if path.is_absolute() else project_root / path).resolve()
    for name in ("benchmark.feature", "steps.py", "metadata.json"):
        if not (path / name).is_file():
            raise SystemExit(f"missing {path / name}")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8-sig"))
    steps_text = (path / "steps.py").read_text(encoding="utf-8")
    return {"api": api, "epoch": int(epoch), "fixture": path, "metadata": metadata,
            "condition": condition, "source_condition": metadata.get("condition", {}).get("name"),
            "needs_influxdb_client": "influxdb_client" in steps_text}


def _check_items(items: list[dict]) -> None:
    conditions = [item["condition"] for item in items]
    epochs = [item["epoch"] for item in items]
    if len(set(conditions)) != len(conditions):
        raise SystemExit(f"duplicate condition names {conditions}")
    if len(set(epochs)) != len(epochs):
        raise SystemExit(f"every series needs its own epoch, got {epochs}")
    for item in items:
        name, epoch = item["condition"], item["epoch"]
        if not name.startswith("_"):
            raise SystemExit(f"condition {name!r} must start with '_' (fixture convention, evaluation/runs.py)")
        if name.startswith("_gold_"):
            raise SystemExit(f"condition {name!r}: the _gold_ prefix is reserved for the E12-E17 ceiling series")
        if epoch < _MIN_EPOCH or epoch in _FORBIDDEN_EPOCHS:
            raise SystemExit(f"epoch {epoch} not allowed (new epochs >= {_MIN_EPOCH}; never 8, 9, 12-17)")


def _check_epoch_unused(items: list[dict], artifacts_dir: Path, *, resume: bool) -> None:
    if not artifacts_dir.is_dir():
        return
    for metadata_path in artifacts_dir.rglob("metadata.json"):
        try:
            epoch = json.loads(metadata_path.read_text(encoding="utf-8-sig")).get("epoch")
        except (OSError, ValueError):
            continue
        for item in items:
            if epoch != item["epoch"]:
                continue
            same_condition = metadata_path.parent.parent.name == item["condition"]
            if not (resume and same_condition):
                raise SystemExit(f"epoch {epoch} already used by {metadata_path} "
                                 f"(choose a new epoch, or --resume for the same condition)")


def _check_exclusive(compose_project: str) -> None:
    try:
        completed = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Ports}}\t{{.Label \"com.docker.compose.project\"}}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SystemExit(f"docker ps failed ({error}); start Docker Desktop first") from None
    if completed.returncode != 0:
        raise SystemExit(f"docker ps failed: {completed.stderr.strip()}; start Docker Desktop first")
    running = [line for line in completed.stdout.splitlines() if line.strip()]
    print(f"docker ps: {len(running)} running container(s)")
    for line in running:
        name, ports, project = (line.split("\t") + ["", "", ""])[:3]
        print(f"   {name}  {ports}  project={project or '-'}")
        if any(f":{port}->" in ports for port in _PORTS) or project.startswith("benchmark-agent-"):
            raise SystemExit(f"container {name} (project {project or '-'}) uses the SUT ports or is a "
                             f"Benchmark-Agent SUT (this checkout: {compose_project}); another run is active")
    for port in _PORTS:
        if _port_busy(port):
            raise SystemExit(f"127.0.0.1:{port} is in use (proxy/SUT of another run?)")
    print(f"ports {_PORTS} free")


def _port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            return True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        try:
            server.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def _check_csv_idle(csv_path: Path, *, ignore: bool) -> None:
    if not csv_path.is_file():
        return
    age = time.time() - csv_path.stat().st_mtime
    print(f"{csv_path}: last written {age:.0f}s ago")
    if age < _CSV_IDLE_SECONDS and not ignore:
        raise SystemExit(f"{csv_path} was written {age:.0f}s ago - another run may be active "
                         "(--ignore-recent-csv to override)")


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--item", action="append", required=True,
                        help="api:epoch:fixture_dir:condition_name (repeatable; run order within a repetition)")
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--results-root", required=True,
                        help="the resolved results artifacts/raw dirs must lie under this directory")
    parser.add_argument("--preflight-only", action="store_true", help="run every check, start nothing")
    parser.add_argument("--resume", action="store_true",
                        help="allow an epoch that already occurs, but only under the same condition")
    parser.add_argument("--ignore-recent-csv", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
