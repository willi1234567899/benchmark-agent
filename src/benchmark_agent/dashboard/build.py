from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from pydantic import BaseModel, Field

from benchmark_agent.config import AppConfig, load_app_config
from benchmark_agent.dashboard.render import render_html
from benchmark_agent.evaluation.coverage import REQUIRED_OPERATIONS
from benchmark_agent.evaluation.runs import (
    SOURCE_PROXY,
    SOURCE_SERVER,
    STATUS_ORDER,
    RunCollection,
    RunRecord,
    collect_runs,
)
from benchmark_agent.evaluation.statistics import (
    LEVEL_ORDER,
    PROMPTING_ORDER,
    StatisticsReport,
    compute_statistics,
)

DASHBOARD_DIR_NAME = "dashboard"
HTML_NAME = "dashboard.html"
MEASURED_SOURCES: tuple[str, ...] = (SOURCE_PROXY, SOURCE_SERVER)

DEFINITIONS: dict[str, str] = {
    "syntax_ok": "gherlint-Syntax + py_compile der Steps ohne Fehler (Stil-Regeln sind Warnungen).",
    "failure_class": (
        "Warum ein Run nicht erfolgreich war (evaluation/failure_class.py), aus der Evidenz "
        "abgeleitet: keine Artefakte, Syntaxfehler, Harness, Timeout, API-Ablehnung der Fassade, "
        "zitiertes Literal im URL-Pfad, Step-Fehler, fehlgeschlagenes Szenario, Workload nicht "
        "konform. Erste zutreffende Klasse gewinnt."
    ),
    "p_adjusted": (
        "Holm-korrigiertes p innerhalb der Testfamilie (Stufe-B-Batterie bzw. paarweise "
        "Fisher-Tests); Tests ohne Familie behalten ihr p."
    ),
    "execution_feedback": (
        "L4/L5 seit Epoche 10: ein gültiger Entwurf wird einmal gegen den Dienst ausgeführt; "
        "Ablehnungen, fehlgeschlagene Szenarien und nicht erfüllte Workload-Teile gehen als "
        "Feedback zurück (agent/execution_feedback.py); Zeilen mit Backend-Vokabular werden "
        "zurückgehalten."
    ),
    "executed_ok": (
        "behave lief ohne Timeout, Exit 0, kein failed/error-Szenario, >= 1 passed-Szenario."
    ),
    "semantics_ok": (
        "Messung aktiv, executed_ok und jede Operation (write/read/delete) hat >= 1 "
        "fehlerfreien Proxy-Request."
    ),
    "success": "syntax_ok ∧ executed_ok ∧ semantics_ok (Stufe A).",
    "adherence (Epoche 3)": (
        "Hinter der Tessera-Fassade zählt der Proxy je Run akzeptierte Requests, per Kanarienvogel "
        "abgelehnte InfluxDB-Konventionen (precision, org/bucket, Authorization, Line Protocol, "
        "Flux, predicate/start/stop) und Aufrufe alter InfluxDB-Endpunkte; adherence_ok = keine "
        "Ablehnung und kein Legacy-Aufruf. Quantifiziert Trainingsgedächtnis vs. Spezifikation."
    ),
    "spec_coverage": (
        "Anteil der bestandenen spezifikations-exklusiven Prüfungen (Workload v2: Zeit-Einheit "
        "ms beim Schreiben, Label-Filter in Fetch und Purge). Nur aus Spec oder Probing erfüllbar; "
        "leer für Workload v1."
    ),
    "vocab_leak": (
        "Statischer Scan des Artefakts auf InfluxDB-Vokabular (influx, Flux, /api/v2, Line "
        "Protocol, precision, bucket, org, _measurement, predicate, Authorization: Token). "
        "Hinter der Fassade nennt kein Prompt InfluxDB; Treffer stammen aus dem Pre-Training."
    ),
    "conformance (Kriterien v2)": (
        "semantics_ok verlangt zusätzlich Workload-Konformität aus der Proxy-Evidenz. write: "
        "akzeptierte Punkte = batch_size × batches; read: Flux mit gepinntem limit/range und "
        "Ziel-Measurement; delete: Predicate auf das Ziel-Measurement über den vollen Zeitraum. "
        "Abweichungen von den gepinnten Literalen (z. B. stop 2030 statt 2100) werden nur "
        "vermerkt. criteria_version/epoch stehen je Run in runs.csv; Epochen nicht poolen."
    ),
    "coverage": (
        "Anteil der Operationen mit executed_ok und >= 1 fehlerfreiem Request (0, 1/3, 2/3, 1)."
    ),
    "throughput_ops_s": (
        "n_requests / aktive Spanne der Operation (erster Request-Start bis letztes "
        "Response-Ende); Fehlläufe sind NA, nicht 0 (Stufe B)."
    ),
    "latency": "Client-seitige Latenz am Proxy, p50/p95/p99 über die Requests der Operation.",
    "proxy_vs_server": (
        "Proxy = primäre Messung; /metrics-Delta des SUT nur als Cross-Check der Zählung."
    ),
    "fixture": (
        "Konditionen mit führendem '_' (z. B. _gold_live) sind manuelle Referenzen/Fixtures, "
        "kein Modell-Output; per Filter zuschaltbar, nie in den Tests."
    ),
}


class DashboardBuild(BaseModel):
    html_path: Path
    table_paths: list[Path] = Field(default_factory=list)
    n_runs: int = 0
    n_measured: int = 0
    synthetic: bool = False
    issues: list[str] = Field(default_factory=list)


def default_output_dir(app_config: AppConfig, *, seed_demo: bool = False) -> Path:
    if seed_demo:
        return app_config.paths.results_reports_dir / "seed" / DASHBOARD_DIR_NAME
    return app_config.paths.results_reports_dir / DASHBOARD_DIR_NAME


def collect_for_config(app_config: AppConfig) -> RunCollection:
    return collect_runs(
        csv_path=app_config.paths.results_raw_dir / "metrics.csv",
        artifacts_dir=app_config.paths.results_artifacts_dir,
        project_root=app_config.paths.project_root,
    )


def build_dashboard(
    app_config: AppConfig | None = None,
    *,
    output_dir: Path | None = None,
    seed_demo: bool = False,
    write_tables: bool = True,
    refresh_seconds: int | None = None,
    project_root: Path | None = None,
) -> DashboardBuild:
    config = app_config or load_app_config(project_root)
    target = (output_dir or default_output_dir(config, seed_demo=seed_demo)).resolve()
    target.mkdir(parents=True, exist_ok=True)

    collection = _collection(config, seed_demo=seed_demo)
    payload = build_payload(collection, config, synthetic=seed_demo)
    full_stats = payload.pop("full_stats")
    html_path = target / HTML_NAME
    html_path.write_text(render_html(payload, refresh_seconds=refresh_seconds), encoding="utf-8")

    tables: list[Path] = []
    if write_tables:
        tables.append(write_runs_table(target / "runs.csv", collection.runs))
        primary = full_stats[f"write|{SOURCE_PROXY}"]
        tables.append(write_cells_table(target / "cells.csv", primary, synthetic=seed_demo))
        stats_path = target / "statistics.json"
        stats_path.write_text(json.dumps(full_stats, indent=2), encoding="utf-8")
        tables.append(stats_path)

    return DashboardBuild(
        html_path=html_path,
        table_paths=tables,
        n_runs=len(collection.runs),
        n_measured=len(collection.measured),
        synthetic=seed_demo,
        issues=list(collection.issues),
    )


def render_current_html(
    app_config: AppConfig, *, seed_demo: bool = False, refresh_seconds: int | None = None
) -> str:
    collection = _collection(app_config, seed_demo=seed_demo)
    payload = build_payload(collection, app_config, synthetic=seed_demo)
    payload.pop("full_stats", None)
    return render_html(payload, refresh_seconds=refresh_seconds)


def _collection(config: AppConfig, *, seed_demo: bool) -> RunCollection:
    if seed_demo:
        from benchmark_agent.evaluation.seed_runs import seed_run_records

        return RunCollection(
            runs=seed_run_records(),
            csv_path="(synthetische Seed-Matrix, keine Messung)",
            artifacts_dir=None,
        )
    return collect_for_config(config)


def build_payload(
    collection: RunCollection,
    app_config: AppConfig | None = None,
    *,
    synthetic: bool = False,
    generated_at: str | None = None,
) -> dict[str, Any]:
    runs = collection.runs
    epochs = sorted({run.epoch or 1 for run in runs if not run.is_fixture})
    if not epochs:
        epochs = [1]
    current_epoch = app_config.defaults.epoch if app_config is not None else epochs[-1]
    if current_epoch not in epochs:
        current_epoch = epochs[-1]
    epoch_runs = [run for run in runs if (run.epoch or 1) == current_epoch]
    stats: dict[str, dict[str, Any]] = {}
    for operation in REQUIRED_OPERATIONS:
        for source in MEASURED_SOURCES:
            report = compute_statistics(epoch_runs, operation=operation, source=source)
            report.notes.append(
                f"Nur Epoche {current_epoch}: {len(epoch_runs)} Runs; andere Epochen sind "
                "ausgeblendet (nie poolen)."
            )
            stats[f"{operation}|{source}"] = report.model_dump(mode="json")
    embedded_stats = {
        key: {
            name: report[name]
            for name in ("operation", "source", "n_runs", "n_measured", "funnel", "tests", "notes")
        }
        for key, report in stats.items()
    }

    models = sorted({run.model or "?" for run in runs})
    levels = [level for level in LEVEL_ORDER if any(run.guidance_level == level for run in runs)]
    levels += sorted(
        {run.guidance_level or "?" for run in runs if run.guidance_level not in LEVEL_ORDER}
    )
    promptings = [
        prompting
        for prompting in PROMPTING_ORDER
        if any(run.prompting == prompting for run in runs)
    ]
    promptings += sorted(
        {run.prompting or "?" for run in runs if run.prompting not in PROMPTING_ORDER}
    )
    statuses = [status for status in STATUS_ORDER if any(run.status == status for run in runs)]

    matrices = load_matrix_reports(app_config.paths.results_raw_dir) if app_config else []
    return {
        "title": "Benchmark-Agent · Run-Dashboard",
        "generated_at": generated_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "synthetic": synthetic,
        "sources": {
            "csv_path": collection.csv_path,
            "artifacts_dir": collection.artifacts_dir,
            "csv_row_count": collection.csv_row_count,
            "issues": list(collection.issues),
            "quarantined": list(collection.quarantined),
        },
        "levels": levels or list(LEVEL_ORDER),
        "models": models,
        "promptings": promptings or list(PROMPTING_ORDER),
        "statuses": statuses,
        "epochs": [str(epoch) for epoch in epochs],
        "current_epoch": str(current_epoch),
        "operations": list(REQUIRED_OPERATIONS),
        "measured_sources": list(MEASURED_SOURCES),
        "runs": [run.model_dump(mode="json") for run in runs],
        "stats": embedded_stats,
        "full_stats": stats,
        "matrices": matrices,
        "definitions": DEFINITIONS,
        "ceiling": ceiling_summary(epoch_runs),
    }


def ceiling_summary(epoch_runs: list[RunRecord]) -> dict[str, dict[str, dict[str, Any]]]:
    gold_runs = [
        run
        for run in epoch_runs
        if run.is_fixture
        and run.condition_name.startswith("_gold_")
        and run.status != "not_evaluated"
    ]
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for source in MEASURED_SOURCES:
        by_operation: dict[str, dict[str, Any]] = {}
        for operation in REQUIRED_OPERATIONS:
            stats = [run.op(source, operation) for run in gold_runs]
            exercised = [stat for stat in stats if stat is not None and stat.exercised]
            if not exercised:
                continue
            throughput = [
                stat.throughput_ops_s for stat in exercised if stat.throughput_ops_s is not None
            ]
            latency_p50 = [
                stat.latency_p50_ms for stat in exercised if stat.latency_p50_ms is not None
            ]
            latency_p95 = [
                stat.latency_p95_ms for stat in exercised if stat.latency_p95_ms is not None
            ]
            latency_p99 = [
                stat.latency_p99_ms for stat in exercised if stat.latency_p99_ms is not None
            ]
            by_operation[operation] = {
                "n": len(exercised),
                "median_throughput_ops_s": median(throughput) if throughput else None,
                "median_latency_p50_ms": median(latency_p50) if latency_p50 else None,
                "median_latency_p95_ms": median(latency_p95) if latency_p95 else None,
                "median_latency_p99_ms": median(latency_p99) if latency_p99 else None,
            }
        if by_operation:
            summary[source] = by_operation
    return summary


def load_matrix_reports(raw_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    if not raw_dir.is_dir():
        return reports
    for path in sorted(raw_dir.glob("matrix_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except OSError, ValueError:
            continue
        outcomes = payload.get("outcomes") if isinstance(payload, dict) else None
        if not isinstance(outcomes, list):
            continue
        valid_outcomes = [outcome for outcome in outcomes if isinstance(outcome, dict)]
        failed = [outcome for outcome in valid_outcomes if outcome.get("success") is not True]
        conditions = sorted({str(outcome.get("condition_name")) for outcome in valid_outcomes})
        errors = [str(outcome.get("error")) for outcome in valid_outcomes if outcome.get("error")]
        reports.append(
            {
                "file": path.name,
                "timestamp": payload.get("timestamp"),
                "n_runs": len(outcomes),
                "n_failed": len(failed),
                "conditions": conditions,
                "errors": errors[:10],
            }
        )
    reports.reverse()
    return reports


_RUN_COLUMNS: tuple[str, ...] = (
    "run_id",
    "condition_id",
    "condition_name",
    "guidance_level",
    "model",
    "prompting",
    "repetition",
    "status",
    "syntax_ok",
    "executed_ok",
    "semantics_ok",
    "success",
    "coverage_ratio",
    "feedback_iterations",
    "attempts",
    "probe_calls",
    "scenarios_passed",
    "scenarios_failed",
    "scenarios_error",
    "generated_at",
    "evaluated_at",
    "model_version",
    "seed",
    "conformance_ok",
    "spec_coverage",
    "sut_api",
    "adherence_ok",
    "api_legacy_requests",
    "vocab_leak",
    "criteria_version",
    "epoch",
    "temperature",
    "llm_backend",
    "prompt_sha256",
    "failure_class",
    "failure_detail",
    "quoted_literal_requests",
    "llm_calls",
    "execution_feedback_rounds",
    "error",
    "synthetic",
)


def write_runs_table(path: Path, runs: list[RunRecord]) -> Path:
    columns = list(_RUN_COLUMNS)
    for source in MEASURED_SOURCES:
        for operation in REQUIRED_OPERATIONS:
            prefix = f"{source}_{operation}"
            columns += [
                f"{prefix}_n_requests",
                f"{prefix}_error_count",
                f"{prefix}_throughput_ops_s",
                f"{prefix}_latency_p50_ms",
                f"{prefix}_latency_p95_ms",
                f"{prefix}_latency_p99_ms",
            ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            row: dict[str, Any] = {name: getattr(run, name) for name in _RUN_COLUMNS}
            for source in MEASURED_SOURCES:
                for operation in REQUIRED_OPERATIONS:
                    stats = run.op(source, operation)
                    prefix = f"{source}_{operation}"
                    row[f"{prefix}_n_requests"] = stats.n_requests if stats else None
                    row[f"{prefix}_error_count"] = stats.error_count if stats else None
                    row[f"{prefix}_throughput_ops_s"] = stats.throughput_ops_s if stats else None
                    row[f"{prefix}_latency_p50_ms"] = stats.latency_p50_ms if stats else None
                    row[f"{prefix}_latency_p95_ms"] = stats.latency_p95_ms if stats else None
                    row[f"{prefix}_latency_p99_ms"] = stats.latency_p99_ms if stats else None
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    return path


_CELL_COLUMNS: tuple[str, ...] = (
    "guidance_level",
    "model",
    "prompting",
    "n",
    "n_syntax_ok",
    "n_executed_ok",
    "n_semantics_ok",
    "n_success",
    "success_rate",
    "mean_coverage",
    "mean_feedback_iterations",
    "n_stage_b",
    "median_throughput",
    "iqr_throughput",
    "naive_mean_throughput",
    "median_latency_p50",
    "median_latency_p95",
    "median_latency_p99",
    "request_error_rate",
)


def write_cells_table(path: Path, report: dict[str, Any], *, synthetic: bool) -> Path:
    parsed = StatisticsReport.model_validate(report)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*_CELL_COLUMNS, "operation", "source", "synthetic"])
        for cell in parsed.cells:
            writer.writerow(
                [
                    *(_csv_value(getattr(cell, column)) for column in _CELL_COLUMNS),
                    parsed.operation,
                    parsed.source,
                    synthetic,
                ]
            )
    return path


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return value
