from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from benchmark_agent.config import AppConfig
from benchmark_agent.evaluation.adherence import evaluate_adherence
from benchmark_agent.evaluation.conformance import evaluate_conformance
from benchmark_agent.evaluation.contamination import scan_text
from benchmark_agent.evaluation.success import behave_executed_ok, evaluate_semantics
from benchmark_agent.execution.behave_runner import BehaveRunner, sanitized_environment
from benchmark_agent.execution.sut import sut_environment
from benchmark_agent.measurement.facade import wire_code
from benchmark_agent.measurement.proxy import MeasurementProxy
from benchmark_agent.models import (
    BehaveRunResult,
    ConformanceResult,
    ProxyWindowStats,
    WorkloadSpec,
)

_MAX_OUTPUT_LINES = 12
_MAX_LINE_CHARS = 200
_OUTPUT_LINE_RE = re.compile(
    r"Assertion|AssertionError|Traceback|Error|Exception|assert |FAILED|failed|HOOK-ERROR",
)
_LIMIT_RE = re.compile(r"limit\(\s*n\s*:\s*(\d+)\s*\)")
_RANGE_RE = re.compile(r"range\(\s*start\s*:\s*-?(\d+)([smhd])")
_UNIT_WORDS = {"s": "second", "m": "minute", "h": "hour", "d": "day"}


@dataclass
class ExecutionFeedback:
    ok: bool
    executed_ok: bool
    semantics_ok: bool
    text: str
    rejections: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    withheld_lines: int = 0
    behave: BehaveRunResult | None = None
    proxy_stats: ProxyWindowStats | None = None

    def log_entry(self) -> dict[str, object]:
        behave = self.behave
        behave_summary: dict[str, object] | None = None
        if behave is not None:
            behave_summary = {
                "exit_code": behave.exit_code,
                "timed_out": behave.timed_out,
                "scenarios_passed": behave.scenarios_passed,
                "scenarios_failed": behave.scenarios_failed,
                "scenarios_error": behave.scenarios_error,
            }
        return {
            "kind": "execution",
            "ok": self.ok,
            "executed_ok": self.executed_ok,
            "semantics_ok": self.semantics_ok,
            "rejections": list(self.rejections),
            "failed_checks": list(self.failed_checks),
            "withheld_lines": self.withheld_lines,
            "behave": behave_summary,
        }


def execute_for_feedback(
    app_config: AppConfig,
    feature_path: Path,
    steps_path: Path,
    *,
    proxy: MeasurementProxy | None = None,
) -> ExecutionFeedback:
    sut = app_config.sut
    timeout = app_config.execution.behave_timeout_seconds
    own_proxy = proxy is None
    if proxy is None:
        proxy = MeasurementProxy(
            listen_host=app_config.measurement.proxy_listen_host,
            listen_port=app_config.measurement.proxy_listen_port,
            upstream_url=sut.influxdb_direct_url,
            upstream_timeout_s=timeout + 5,
            api=sut.api,
            org=sut.org,
            bucket=sut.bucket,
        )
        proxy.start()
    try:
        env = {
            **sanitized_environment(os.environ),
            **sut_environment(sut, influx_url=sut.influxdb_url),
        }
        proxy.open_window()
        behave = BehaveRunner(timeout).run(feature_path, steps_path, env=env)
        stats = proxy.close_window()
    finally:
        if own_proxy:
            proxy.stop()
    return judge_execution(app_config.workload, sut.api, behave, stats)


def judge_execution(
    workload: WorkloadSpec, api: str, behave: BehaveRunResult, stats: ProxyWindowStats
) -> ExecutionFeedback:
    conformance = evaluate_conformance(workload, stats)
    adherence = evaluate_adherence(stats, api=api)
    executed = behave_executed_ok(behave)
    semantics = evaluate_semantics(stats, behave, conformance=conformance)
    ok = bool(executed and semantics)
    rejections = _rejection_lines(stats)
    failed_checks: list[str] = []
    if conformance is not None:
        failed_checks = [
            f"{check.operation}/{check.name}" for check in conformance.checks if not check.ok
        ]
    feedback = ExecutionFeedback(
        ok=ok,
        executed_ok=executed,
        semantics_ok=bool(semantics),
        text="",
        rejections=rejections,
        failed_checks=failed_checks,
        behave=behave,
        proxy_stats=stats,
    )
    if ok:
        return feedback
    text, withheld = format_execution_feedback(
        workload,
        behave,
        stats,
        conformance,
        rejections,
        adherence_ok=adherence.ok if adherence is not None else True,
    )
    feedback.text = text
    feedback.withheld_lines = withheld
    return feedback


def format_execution_feedback(
    workload: WorkloadSpec,
    behave: BehaveRunResult,
    stats: ProxyWindowStats,
    conformance: ConformanceResult | None,
    rejections: list[str],
    *,
    adherence_ok: bool,
) -> tuple[str, int]:
    lines = ["Execution feedback (the artifacts were run once against the live service):"]
    if behave.timed_out:
        lines.append(f"- behave did not finish within the time limit ({behave.exit_code}).")
    elif behave.exit_code != 0 or (behave.scenarios_failed or 0) or (behave.scenarios_error or 0):
        lines.append(
            f"- behave exit code {behave.exit_code}: scenarios passed "
            f"{behave.scenarios_passed or 0}, failed {behave.scenarios_failed or 0}, "
            f"errors {behave.scenarios_error or 0}."
        )
    if not adherence_ok and rejections:
        lines.append("- The service rejected requests:")
        lines.extend(f"  - {item}" for item in rejections)
    if conformance is not None and not conformance.ok:
        lines.append("- The pinned workload was not met:")
        for check in conformance.checks:
            if not check.ok:
                lines.append(f"  - {check.operation}: {_neutral_check_text(workload, check.name)}")
    if stats.by_operation and not any(
        operation.n_requests for operation in stats.by_operation.values()
    ):
        lines.append(
            "- No request reached the service at all (check the base URL and the "
            "environment variables)."
        )
    output = _relevant_output(behave)
    withheld = 0
    if output:
        lines.append("- Relevant lines from the behave output:")
        for line in output:
            if scan_text(line).total:
                withheld += 1
                continue
            lines.append(f"    {line}")
    if withheld:
        lines.append(f"  ({withheld} output line(s) withheld)")
    lines.append(
        "Fix the artifacts so that every scenario passes against the service and the pinned "
        "workload is met exactly."
    )
    return "\n".join(lines), withheld


def _rejection_lines(stats: ProxyWindowStats) -> list[str]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for sample in stats.requests:
        if sample.rejection:
            path = sample.path.split("?", 1)[0]
            counter[(wire_code(sample.rejection), sample.method, path)] += 1
    return [
        f"{count} request(s) rejected with `{code}` ({method} {path})"
        for (code, method, path), count in sorted(counter.items(), key=lambda item: -item[1])
    ]


def _neutral_check_text(workload: WorkloadSpec, check: str) -> str:
    operations = workload.operations
    write_spec = operations.get("write")
    read_spec = operations.get("read")
    delete_spec = operations.get("delete")
    write = write_spec.settings_dict() if write_spec is not None else {}
    read = read_spec.settings_dict() if read_spec is not None else {}
    delete = delete_spec.settings_dict() if delete_spec is not None else {}
    expected_points = int(write.get("batch_size", 0)) * int(write.get("batches", 1))
    template = str(read.get("query_template", ""))
    limit = _LIMIT_RE.search(template)
    window = _RANGE_RE.search(template)
    read_labels = ", ".join(f"{key} = {value}" for key, value in (read.get("labels") or {}).items())
    delete_labels = ", ".join(
        f"{key} = {value}" for key, value in (delete.get("labels") or {}).items()
    )
    label_filter = read_labels or delete_labels
    texts = {
        "points": (
            f"the service did not accept exactly {expected_points} written points "
            f"({write.get('batches', 1)} request(s) x {write.get('batch_size', 0)} points)"
        ),
        "ts_unit": f"the write did not declare the pinned time unit ({write.get('ts_unit')})",
        "limit": (
            f"the read did not request at most {limit.group(1)} points"
            if limit
            else "the read did not apply the pinned result limit"
        ),
        "range": (
            f"the read did not cover the last {window.group(1)} "
            f"{_UNIT_WORDS.get(window.group(2), window.group(2))}(s)"
            if window
            else "the read did not cover the pinned time window"
        ),
        "measurement": "the read did not target the pinned series `benchmark`",
        "labels": "the request lacks the pinned label filter"
        + (f" ({label_filter})" if label_filter else ""),
        "predicate": "the delete did not target the pinned series `benchmark`",
        "time_range": (
            "the delete time range does not cover all written points (use the full pinned range)"
        ),
    }
    return texts.get(check, f"the pinned {check} was not met")


def _relevant_output(behave: BehaveRunResult) -> list[str]:
    combined = (behave.stdout or "") + "\n" + (behave.stderr or "")
    picked: list[str] = []
    for raw_line in combined.splitlines():
        line = raw_line.strip()
        if not line or not _OUTPUT_LINE_RE.search(line):
            continue
        picked.append(line[:_MAX_LINE_CHARS])
    return picked[-_MAX_OUTPUT_LINES:]
