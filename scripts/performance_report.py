from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from benchmark_agent.config import load_app_config
from benchmark_agent.dashboard.build import ceiling_summary
from benchmark_agent.evaluation.coverage import REQUIRED_OPERATIONS
from benchmark_agent.evaluation.runs import SOURCE_PROXY, RunRecord, collect_runs
from benchmark_agent.evaluation.statistics import LEVEL_ORDER, compute_statistics

EPOCHS: tuple[int, ...] = (6, 7, 8, 9, 10, 11, 12, 13)
EPOCH_META: dict[int, dict[str, str]] = {
    6: {"model": "flash-lite", "dialect": "Kestrel", "regime": "zero-shot, Syntax-Feedback"},
    7: {"model": "flash-lite", "dialect": "Tessera", "regime": "zero-shot, Syntax-Feedback"},
    8: {"model": "flash", "dialect": "Kestrel", "regime": "zero-shot, Syntax-Feedback"},
    9: {"model": "flash", "dialect": "Tessera", "regime": "zero-shot, Syntax-Feedback"},
    10: {"model": "flash-lite", "dialect": "Kestrel", "regime": "zero-shot, Ausf.-Feedback"},
    11: {"model": "flash-lite", "dialect": "Tessera", "regime": "zero-shot, Ausf.-Feedback"},
    12: {
        "model": "flash + flash-lite",
        "dialect": "Tessera",
        "regime": "multi-shot, Ausf.-Feedback",
    },
    13: {
        "model": "flash + flash-lite",
        "dialect": "Kestrel",
        "regime": "multi-shot, Ausf.-Feedback",
    },
}
ZS_MS_PAIRS: tuple[tuple[str, str, int, int], ...] = (
    ("flash", "Kestrel", 8, 13),
    ("flash", "Tessera", 9, 12),
    ("flash-lite", "Kestrel", 10, 13),
    ("flash-lite", "Tessera", 11, 12),
)
MODEL_KEYS = {"flash": "gemini", "flash-lite": "gemini_lite"}
PERCENTILES: tuple[tuple[str, str], ...] = (
    ("latency_p50_ms", "p50"),
    ("latency_p95_ms", "p95"),
    ("latency_p99_ms", "p99"),
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
RAMP_3 = ("#6da7ec", "#2a78d6", "#184f95")
RAMP_2 = ("#86b6ef", "#1c5cab")
DPI = 150
LABEL_BOX: dict[str, Any] = {"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none"}


def main() -> None:
    config = load_app_config()
    out_dir = config.paths.results_reports_dir / "performance_gemini"
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    _style()

    collection = collect_runs(
        csv_path=config.paths.results_raw_dir / "metrics.csv",
        artifacts_dir=config.paths.results_artifacts_dir,
        project_root=config.paths.project_root,
    )
    by_epoch: dict[int, list[RunRecord]] = defaultdict(list)
    for run in collection.runs:
        if run.epoch in EPOCHS:
            by_epoch[run.epoch].append(run)
    for epoch in EPOCHS:
        if not by_epoch.get(epoch):
            raise SystemExit(f"Epoch {epoch} has no runs; nothing to plot.")

    ceilings = {epoch: ceiling_summary(by_epoch[epoch]).get(SOURCE_PROXY, {}) for epoch in EPOCHS}
    stats = {
        epoch: {
            op: compute_statistics(by_epoch[epoch], operation=op, source=SOURCE_PROXY)
            for op in REQUIRED_OPERATIONS
        }
        for epoch in EPOCHS
    }

    written: list[tuple[str, str]] = []
    for epoch in EPOCHS:
        runs = _model_runs(by_epoch[epoch])
        written.append(
            (
                _fig_throughput(epoch, runs, ceilings[epoch], out_dir),
                f"Epoche {epoch}: Throughput je Run und Level (write/read/delete)",
            )
        )
        written.append(
            (
                _fig_latency(epoch, runs, ceilings[epoch], out_dir),
                f"Epoche {epoch}: Latenz p50/p95/p99 je Run und Level",
            )
        )
    written.append(
        (
            _fig_final_level_vs_performance(by_epoch, ceilings, out_dir),
            "Gesamtbild Stufe B: write-Throughput je Run gegen das Guidance-Level, alle "
            "Epochen 6–13 in einem Koordinatensystem (Farbe = Dialekt, Form = Modell)",
        )
    )
    written.append(
        (
            _fig_stage_a_vs_b(by_epoch, out_dir),
            "Stufe A (Erfolgsquote) neben Stufe B (Median write-Throughput) je Level, alle Epochen",
        )
    )
    written.append(
        (
            _fig_spearman(stats, out_dir),
            "Spearman ρ Level↔Throughput je Epoche und Operation (95-%-KI, gefüllt = p < 0,05)",
        )
    )
    written.append(
        (
            _fig_ceiling(by_epoch, ceilings, out_dir),
            "Epochen 12–13: Level-Mediane gegen die bddbench-Ceiling je Operation",
        )
    )
    written.append(
        (
            _fig_zero_vs_multi(by_epoch, out_dir),
            "Zero-Shot- gegen Multi-Shot-Epoche je Modell × Dialekt (Median write-Throughput)",
        )
    )
    written.append(
        (
            _fig_latency_summary(by_epoch, out_dir),
            "Median-Latenz p50 und p95 (write) je Level, alle Epochen",
        )
    )

    _write_run_table(tables_dir / "messwerte_je_run.csv", by_epoch)
    _write_level_table(tables_dir / "auswertung_je_level.csv", by_epoch)
    _write_spearman_table(tables_dir / "spearman.csv", stats)
    _write_ceiling_table(tables_dir / "ceiling.csv", ceilings)
    _write_final_table(tables_dir / "final_level_vs_performance.csv", by_epoch, ceilings)
    _write_readme(out_dir, written, by_epoch, stats, ceilings)
    (out_dir / "generated.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "epochs": list(EPOCHS),
                "runs_per_epoch": {str(epoch): len(by_epoch[epoch]) for epoch in EPOCHS},
                "figures": [name for name, _ in written],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Written {len(written)} figures + 5 tables + README.md -> {out_dir}")


def _model_runs(runs: Sequence[RunRecord]) -> list[RunRecord]:
    return [run for run in runs if not run.is_fixture]


def _values(runs: Sequence[RunRecord], level: str, op: str, metric: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        if run.guidance_level != level:
            continue
        value = run.stage_b_value(SOURCE_PROXY, op, metric)
        if value is not None:
            values.append(value)
    return values


def _quartiles(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    q1, q2, q3 = np.percentile(array, [25, 50, 75])
    return float(q1), float(q2), float(q3)


def _success_rate(runs: Sequence[RunRecord], level: str) -> tuple[int, int]:
    level_runs = [run for run in runs if run.guidance_level == level]
    return sum(1 for run in level_runs if run.success), len(level_runs)


def _fisher_ci(rho: float, n: int) -> tuple[float, float]:
    if n <= 3 or abs(rho) >= 1:
        return rho, rho
    z = math.atanh(rho)
    half = 1.96 / math.sqrt(n - 3)
    return math.tanh(z - half), math.tanh(z + half)


def _epoch_title(epoch: int) -> str:
    meta = EPOCH_META[epoch]
    return f"Epoche {epoch} · {meta['model']} · {meta['dialect']} · {meta['regime']}"


def _fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "–"
    return f"{value:.{digits}f}".replace(".", ",")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.labelcolor": INK_2,
        }
    )


def _titles(fig: Any, title: str, *, bottom: float, top: float) -> None:
    fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, bottom, 1, top))


def _finish(fig: Any, path: Path, *, subtitle: str, footer: str | None = None) -> str:
    fig.text(0.01, 0.958, subtitle, ha="left", va="top", fontsize=8.5, color=INK_2)
    if footer:
        fig.text(0.01, 0.008, footer, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path.name


def _strip(
    ax: Axes,
    runs: Sequence[RunRecord],
    op: str,
    metric: str,
    *,
    unit: str,
    ceiling: float | None,
    label_medians: bool = True,
) -> None:
    rng = np.random.default_rng(20260715)
    top = 0.0
    for idx, level in enumerate(LEVEL_ORDER):
        values = _values(runs, level, op, metric)
        if not values:
            continue
        x = idx + rng.uniform(-0.22, 0.22, size=len(values))
        ax.scatter(
            x,
            values,
            s=34,
            color=SERIES_1,
            edgecolors=SURFACE,
            linewidths=1.2,
            zorder=3,
        )
        med = median(values)
        ax.hlines(med, idx - 0.32, idx + 0.32, color=INK, linewidth=2, zorder=4)
        if label_medians:
            ax.annotate(
                _fmt(med),
                (idx + 0.36, med),
                ha="left",
                va="center",
                fontsize=7.5,
                color=INK_2,
                zorder=5,
                bbox=LABEL_BOX,
            )
        top = max(top, max(values))
    if ceiling is not None:
        ax.axhline(ceiling, color=SERIES_2, linewidth=1.6, linestyle=(0, (5, 3)), zorder=2)
        top = max(top, ceiling)
    ax.set_xticks(range(len(LEVEL_ORDER)))
    ax.set_xticklabels(LEVEL_ORDER)
    ax.set_xlim(-0.6, len(LEVEL_ORDER) - 0.4 + 0.5)
    ax.set_ylim(0, top * 1.12 if top > 0 else 1)
    ax.set_ylabel(unit)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)


def _legend_dots_ceiling(fig: Any, with_ceiling: bool, *, ceiling_label: str) -> None:
    handles = [
        Line2D(
            [], [], marker="o", color=SERIES_1, linestyle="", markersize=6, label="Run (Stufe B)"
        ),
        Line2D([], [], color=INK, linewidth=2, label="Median je Level"),
    ]
    if with_ceiling:
        handles.append(
            Line2D(
                [], [], color=SERIES_2, linewidth=1.6, linestyle=(0, (5, 3)), label=ceiling_label
            )
        )
    fig.legend(handles=handles, loc="lower right", ncol=len(handles), bbox_to_anchor=(0.99, 0.0))


def _fig_throughput(
    epoch: int, runs: Sequence[RunRecord], ceiling: dict[str, dict[str, Any]], out_dir: Path
) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)
    has_ceiling = False
    for ax, op in zip(axes, REQUIRED_OPERATIONS, strict=True):
        ceil = ceiling.get(op, {}).get("median_throughput_ops_s") if ceiling else None
        has_ceiling = has_ceiling or ceil is not None
        _strip(ax, runs, op, "throughput_ops_s", unit="ops/s", ceiling=ceil)
        n = sum(len(_values(runs, level, op, "throughput_ops_s")) for level in LEVEL_ORDER)
        ax.set_title(f"{op} · n = {n} Stufe-B-Runs", loc="left")
    _legend_dots_ceiling(
        fig, has_ceiling, ceiling_label="bddbench-Referenz (Median, N = 10, gleicher Messkern)"
    )
    _titles(fig, f"{_epoch_title(epoch)}: Throughput je Run", bottom=0.07, top=0.85)
    return _finish(
        fig,
        out_dir / f"E{epoch:02d}_throughput.png",
        subtitle=(
            "Proxy-Messung, ein Punkt je Run mit >= 1 fehlerfreiem Request der Operation; "
            "Fehlläufe sind NA (nicht 0). L3 in E6–E11: zero- und multi-shot gemeinsam."
        ),
        footer="Quelle: results/raw/metrics.csv + results/artifacts (evaluation/runs.py) · "
        "Tabelle: tables/messwerte_je_run.csv",
    )


def _fig_latency(
    epoch: int, runs: Sequence[RunRecord], ceiling: dict[str, dict[str, Any]], out_dir: Path
) -> str:
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
    has_ceiling = False
    for row, (metric, percentile_label) in enumerate(PERCENTILES):
        for col, op in enumerate(REQUIRED_OPERATIONS):
            ax = axes[row][col]
            ceil = ceiling.get(op, {}).get(f"median_{metric}") if ceiling else None
            has_ceiling = has_ceiling or ceil is not None
            _strip(ax, runs, op, metric, unit="ms", ceiling=ceil, label_medians=(row == 0))
            ax.set_title(f"{op} · {percentile_label}", loc="left")
    _legend_dots_ceiling(
        fig, has_ceiling, ceiling_label="bddbench-Referenz (Median, N = 10, gleicher Messkern)"
    )
    _titles(fig, f"{_epoch_title(epoch)}: Latenz je Run (p50 / p95 / p99)", bottom=0.04, top=0.93)
    return _finish(
        fig,
        out_dir / f"E{epoch:02d}_latenz.png",
        subtitle=(
            "Client-seitige Latenz am Proxy je Run; delete hat einen Request je Run "
            "(p50 = p95 = p99). Median-Werte nur in der p50-Zeile beschriftet."
        ),
        footer="Quelle wie Throughput · Tabelle: tables/messwerte_je_run.csv",
    )


def _fig_stage_a_vs_b(by_epoch: dict[int, list[RunRecord]], out_dir: Path) -> str:
    fig, axes = plt.subplots(2, len(EPOCHS), figsize=(16, 6.2), sharex=True)
    x = np.arange(len(LEVEL_ORDER))
    max_tp = 0.0
    for col, epoch in enumerate(EPOCHS):
        runs = _model_runs(by_epoch[epoch])
        meta = EPOCH_META[epoch]
        ax_a = axes[0][col]
        rates = []
        for level in LEVEL_ORDER:
            ok, n = _success_rate(runs, level)
            rates.append(ok / n if n else 0.0)
        ax_a.bar(x, rates, width=0.55, color=SERIES_1, zorder=3)
        for xi, level in zip(x, LEVEL_ORDER, strict=True):
            ok, n = _success_rate(runs, level)
            if n:
                ax_a.annotate(
                    f"{ok}/{n}", (xi, rates[xi] + 0.03), ha="center", fontsize=7, color=INK_2
                )
        ax_a.set_ylim(0, 1.18)
        ax_a.set_yticks([0, 0.5, 1.0])
        ax_a.set_yticklabels(["0 %", "50 %", "100 %"] if col == 0 else ["", "", ""])
        regime = meta["regime"].replace("Feedback", "FB").replace(", ", " · ")
        ax_a.set_title(
            f"E{epoch} · {meta['model']}\n{meta['dialect']}\n{regime}",
            loc="left",
            fontsize=8,
        )
        ax_a.tick_params(axis="both", length=0)

        ax_b = axes[1][col]
        for xi, level in zip(x, LEVEL_ORDER, strict=True):
            values = _values(runs, level, "write", "throughput_ops_s")
            if not values:
                continue
            q1, q2, q3 = _quartiles(values)
            max_tp = max(max_tp, q3)
            ax_b.vlines(xi, q1, q3, color=SERIES_1, linewidth=2, zorder=3)
            ax_b.scatter(
                [xi], [q2], s=42, color=SERIES_1, edgecolors=SURFACE, linewidths=1.2, zorder=4
            )
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(LEVEL_ORDER)
        ax_b.tick_params(axis="both", length=0)
        if col == 0:
            ax_a.set_ylabel("Erfolgsquote (Stufe A)")
            ax_b.set_ylabel("write ops/s, Median + IQR")
    for col in range(len(EPOCHS)):
        axes[1][col].set_ylim(0, max_tp * 1.12)
        if col:
            axes[1][col].set_yticklabels([])
    _titles(
        fig,
        "Stufe A (oben) und Stufe B (unten) je Guidance-Level, eine Spalte je Epoche, nie gepoolt",
        bottom=0.04,
        top=0.89,
    )
    return _finish(
        fig,
        out_dir / "auswertung_stufeA_vs_stufeB.png",
        subtitle=(
            "Oben: Anteil erfolgreicher Runs (syntax ∧ ausgeführt ∧ Workload konform) je Level. "
            "Unten: Median und Interquartilsabstand des write-Throughputs der Stufe-B-Runs. "
            "Gleiche x-Achse, bewusst zwei Diagramme statt Doppelachse."
        ),
        footer="Tabelle: tables/auswertung_je_level.csv · Tests: tables/spearman.csv",
    )


def _fig_spearman(stats: dict[int, dict[str, Any]], out_dir: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    x = np.arange(len(EPOCHS))
    for ax, op in zip(axes, REQUIRED_OPERATIONS, strict=True):
        ax.axhline(0, color=BASELINE, linewidth=0.8, zorder=1)
        for xi, epoch in zip(x, EPOCHS, strict=True):
            test = _spearman_test(stats[epoch][op])
            if test is None or test.statistic is None:
                continue
            rho = float(test.statistic)
            lo, hi = _fisher_ci(rho, test.n)
            significant = test.p_value is not None and test.p_value < 0.05
            ax.vlines(xi, lo, hi, color=SERIES_1, linewidth=1.6, zorder=3)
            ax.scatter(
                [xi],
                [rho],
                s=52,
                facecolors=SERIES_1 if significant else SURFACE,
                edgecolors=SERIES_1,
                linewidths=1.6,
                zorder=4,
            )
            ax.annotate(
                f"p = {_fmt(test.p_value, 2)}",
                (xi, hi + 0.04),
                ha="center",
                fontsize=7,
                color=INK_2,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"E{e}" for e in EPOCHS])
        ax.set_ylim(-1, 1)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_title(op, loc="left")
        ax.tick_params(axis="both", length=0)
    axes[0].set_ylabel("Spearman ρ (Level 1…5 ↔ Throughput je Run)")
    handles = [
        Line2D([], [], marker="o", color=SERIES_1, linestyle="", markersize=7, label="ρ, p < 0,05"),
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor=SURFACE,
            markeredgecolor=SERIES_1,
            color=SERIES_1,
            linestyle="",
            markersize=7,
            label="ρ, nicht signifikant",
        ),
        Line2D([], [], color=SERIES_1, linewidth=1.6, label="95-%-KI (Fisher-z)"),
    ]
    fig.legend(handles=handles, loc="lower right", ncol=3, bbox_to_anchor=(0.99, 0.0))
    _titles(
        fig,
        "Zeigt sich die Stufe-A-Leiter in Stufe B? Rangkorrelation Guidance-Level ↔ Throughput",
        bottom=0.08,
        top=0.85,
    )
    return _finish(
        fig,
        out_dir / "auswertung_spearman.png",
        subtitle=(
            "Derselbe Test wie im Dashboard (evaluation/statistics.py); nur Stufe-B-Runs, "
            "unkorrigierte p-Werte. E12/E13 poolen flash + flash-lite innerhalb der Epoche."
        ),
        footer="Tabelle: tables/spearman.csv",
    )


def _spearman_test(report: Any) -> Any | None:
    for test in report.tests:
        if test.name == "Spearman":
            return test
    return None


def _fig_ceiling(
    by_epoch: dict[int, list[RunRecord]],
    ceilings: dict[int, dict[str, dict[str, Any]]],
    out_dir: Path,
) -> str:
    ceiling_epochs = [epoch for epoch in EPOCHS if ceilings[epoch]]
    fig, axes = plt.subplots(len(ceiling_epochs), 3, figsize=(12, 3.6 * len(ceiling_epochs)))
    axes = np.atleast_2d(axes)
    x = np.arange(len(LEVEL_ORDER))
    for row, epoch in enumerate(ceiling_epochs):
        runs = _model_runs(by_epoch[epoch])
        for col, op in enumerate(REQUIRED_OPERATIONS):
            ax = axes[row][col]
            top = 0.0
            for xi, level in zip(x, LEVEL_ORDER, strict=True):
                values = _values(runs, level, op, "throughput_ops_s")
                if not values:
                    continue
                q1, q2, q3 = _quartiles(values)
                top = max(top, q3)
                ax.vlines(xi, q1, q3, color=SERIES_1, linewidth=2, zorder=3)
                ax.scatter(
                    [xi], [q2], s=42, color=SERIES_1, edgecolors=SURFACE, linewidths=1.2, zorder=4
                )
                ax.annotate(
                    _fmt(q2),
                    (xi + 0.14, q2),
                    ha="left",
                    va="center",
                    fontsize=7.5,
                    color=INK_2,
                    zorder=5,
                    bbox=LABEL_BOX,
                )
            ceil = ceilings[epoch].get(op, {}).get("median_throughput_ops_s")
            if ceil is not None:
                ax.axhline(ceil, color=SERIES_2, linewidth=1.6, linestyle=(0, (5, 3)), zorder=2)
                ax.annotate(
                    f"bddbench {_fmt(ceil)}",
                    (-0.45, ceil),
                    ha="left",
                    va="bottom",
                    fontsize=7.5,
                    color=INK_2,
                    zorder=5,
                    bbox=LABEL_BOX,
                )
                top = max(top, ceil)
            ax.set_ylim(0, top * 1.18)
            ax.set_xlim(-0.6, len(LEVEL_ORDER) - 0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(LEVEL_ORDER)
            ax.set_title(f"E{epoch} · {EPOCH_META[epoch]['dialect']} · {op}", loc="left")
            ax.tick_params(axis="both", length=0)
            if col == 0:
                ax.set_ylabel("ops/s, Median + IQR")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=SERIES_1,
            linestyle="",
            markersize=6,
            label="Level-Median (IQR als Strich)",
        ),
        Line2D(
            [],
            [],
            color=SERIES_2,
            linewidth=1.6,
            linestyle=(0, (5, 3)),
            label="bddbench-Referenz, Median über N = 10",
        ),
    ]
    fig.legend(handles=handles, loc="lower right", ncol=2, bbox_to_anchor=(0.99, 0.0))
    _titles(
        fig,
        "Generierte Clients gegen die handgeschriebene bddbench-Referenz (gleicher Messkern)",
        bottom=0.05,
        top=0.91,
    )
    return _finish(
        fig,
        out_dir / "auswertung_ceiling.png",
        subtitle=(
            "Referenz = tests/fixtures/gold_<dialekt> N = 10-mal via scripts/run_gold_ceiling.py; "
            "L1 hat keine Stufe-B-Runs (0/20 erfolgreich)."
        ),
        footer="Tabellen: tables/ceiling.csv, tables/auswertung_je_level.csv",
    )


FINAL_MARKER: dict[str, str] = {"gemini_lite": "o", "gemini": "^"}
FINAL_MODEL_LABEL: dict[str, str] = {"gemini_lite": "flash-lite", "gemini": "flash"}
FINAL_HUE: dict[str, str] = {"Kestrel": SERIES_1, "Tessera": SERIES_2}


def _fig_final_level_vs_performance(
    by_epoch: dict[int, list[RunRecord]],
    ceilings: dict[int, dict[str, dict[str, Any]]],
    out_dir: Path,
) -> str:
    fig, ax = plt.subplots(figsize=(13, 6.4))
    rng = np.random.default_rng(20260715)
    offsets = np.linspace(-0.36, 0.36, len(EPOCHS))
    top = 0.0
    minor_positions: list[float] = []
    minor_labels: list[str] = []
    for idx, level in enumerate(LEVEL_ORDER):
        if idx > 0:
            ax.axvline(idx - 0.5, color=GRID, linewidth=0.8, zorder=1)
        for slot, epoch in zip(offsets, EPOCHS, strict=True):
            dialect = EPOCH_META[epoch]["dialect"]
            cell = [run for run in _model_runs(by_epoch[epoch]) if run.guidance_level == level]
            values: list[float] = []
            for run in cell:
                value = run.stage_b_value(SOURCE_PROXY, "write", "throughput_ops_s")
                if value is None:
                    continue
                values.append(value)
                ax.scatter(
                    [idx + slot + rng.uniform(-0.028, 0.028)],
                    [value],
                    s=22,
                    color=FINAL_HUE[dialect],
                    marker=FINAL_MARKER.get(run.model or "", "o"),
                    edgecolors=SURFACE,
                    linewidths=0.8,
                    alpha=0.9,
                    zorder=3,
                )
            if level != LEVEL_ORDER[0]:
                minor_positions.append(idx + slot)
                minor_labels.append(str(epoch))
            if not values:
                continue
            med = median(values)
            ax.hlines(
                med, idx + slot - 0.042, idx + slot + 0.042, color=INK, linewidth=1.8, zorder=4
            )
            top = max(top, max(values))
    for epoch in EPOCHS:
        ceil = ceilings[epoch].get("write", {}).get("median_throughput_ops_s")
        if ceil is None:
            continue
        dialect = EPOCH_META[epoch]["dialect"]
        ax.axhline(ceil, color=FINAL_HUE[dialect], linewidth=1.4, linestyle=(0, (5, 3)), zorder=2)
        ax.annotate(
            f"bddbench {dialect} {_fmt(ceil)} (E{epoch})",
            (-0.45, ceil),
            ha="left",
            va="bottom",
            fontsize=7.5,
            color=INK_2,
            zorder=5,
            bbox=LABEL_BOX,
        )
        top = max(top, ceil)
    n_l1 = sum(
        1 for epoch in EPOCHS for run in _model_runs(by_epoch[epoch]) if run.guidance_level == "L1"
    )
    ax.text(
        0,
        top * 0.72,
        f"L1: 0 von {n_l1} Runs\nausführbar,\nkeine Stufe-B-Werte",
        ha="center",
        va="center",
        fontsize=8,
        color=MUTED,
        zorder=5,
    )
    ax.set_xlim(-0.5, len(LEVEL_ORDER) - 0.5)
    ax.set_ylim(0, top * 1.1)
    ax.set_xticks(range(len(LEVEL_ORDER)))
    ax.set_xticklabels(LEVEL_ORDER, fontsize=10)
    ax.set_xticks(minor_positions, minor=True)
    ax.set_xticklabels(minor_labels, minor=True, fontsize=6.5, color=MUTED)
    ax.tick_params(axis="x", which="major", length=0, pad=14)
    ax.tick_params(axis="x", which="minor", length=0, pad=2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Guidance-Level (innerhalb jedes Levels: Epoche 6 → 13, kleine Ziffern)")
    ax.set_ylabel("write-Throughput, ops/s (ein Punkt je Run)")
    handles = [
        Line2D(
            [],
            [],
            marker=FINAL_MARKER[model],
            color=FINAL_HUE[dialect],
            linestyle="",
            markersize=6,
            label=f"{dialect} · {FINAL_MODEL_LABEL[model]}",
        )
        for dialect in ("Kestrel", "Tessera")
        for model in ("gemini_lite", "gemini")
    ]
    handles.append(Line2D([], [], color=INK, linewidth=1.8, label="Median je Epoche × Level"))
    handles.append(
        Line2D(
            [],
            [],
            color=INK_2,
            linewidth=1.4,
            linestyle=(0, (5, 3)),
            label="bddbench-Referenz (Farbe = Dialekt)",
        )
    )
    fig.legend(handles=handles, loc="lower right", ncol=3, bbox_to_anchor=(0.99, 0.0))
    _titles(
        fig,
        "Stufe B, Gesamtbild: write-Throughput je Guidance-Level, Epochen 6–13 in einem "
        "Koordinatensystem",
        bottom=0.1,
        top=0.91,
    )
    return _finish(
        fig,
        out_dir / "auswertung_final_level_vs_performance.png",
        subtitle=(
            "x = Guidance-Level, y = write-Throughput je Stufe-B-Run; Farbe = Dialekt, Form = "
            "Modell; Strich = Median je Epoche × Level. Epochen nie gepoolt (innerhalb eines "
            "Levels nebeneinander)."
        ),
        footer=(
            "Nur executed_ok-Runs mit ≥ 1 fehlerfreiem write-Request. Referenz gemessen in E12 "
            "(Tessera) und E13 (Kestrel).\nDrift zwischen Epochen bis ±50 % bei identischen "
            "Artefakten. Tabelle: tables/final_level_vs_performance.csv"
        ),
    )


def _fig_zero_vs_multi(by_epoch: dict[int, list[RunRecord]], out_dir: Path) -> str:
    fig, axes = plt.subplots(1, len(ZS_MS_PAIRS), figsize=(15, 4.4), sharey=True)
    x = np.arange(len(LEVEL_ORDER))
    top = 0.0
    for ax, (model, dialect, zs_epoch, ms_epoch) in zip(axes, ZS_MS_PAIRS, strict=True):
        key = MODEL_KEYS[model]
        zs_runs = [
            run
            for run in _model_runs(by_epoch[zs_epoch])
            if run.model == key and run.prompting == "zero_shot"
        ]
        ms_runs = [
            run
            for run in _model_runs(by_epoch[ms_epoch])
            if run.model == key and run.prompting == "multi_shot"
        ]
        for xi, level in zip(x, LEVEL_ORDER, strict=True):
            zs_values = _values(zs_runs, level, "write", "throughput_ops_s")
            ms_values = _values(ms_runs, level, "write", "throughput_ops_s")
            if not zs_values and not ms_values:
                continue
            zs_med = median(zs_values) if zs_values else None
            ms_med = median(ms_values) if ms_values else None
            if zs_med is not None and ms_med is not None:
                ax.plot([xi, xi], [zs_med, ms_med], color=BASELINE, linewidth=1.4, zorder=2)
            if zs_med is not None:
                ax.scatter(
                    [xi],
                    [zs_med],
                    s=90,
                    color=RAMP_2[0],
                    edgecolors=SURFACE,
                    linewidths=1.2,
                    zorder=3,
                )
                top = max(top, zs_med)
            if ms_med is not None:
                ax.scatter(
                    [xi],
                    [ms_med],
                    s=40,
                    color=RAMP_2[1],
                    edgecolors=SURFACE,
                    linewidths=1.2,
                    zorder=4,
                )
                top = max(top, ms_med)
        ax.set_xticks(x)
        ax.set_xticklabels(LEVEL_ORDER)
        ax.set_title(
            f"{model} · {dialect}\nzero-shot E{zs_epoch} → multi-shot E{ms_epoch}",
            loc="left",
            fontsize=8.5,
        )
        ax.tick_params(axis="both", length=0)
    for ax in axes:
        ax.set_ylim(0, top * 1.15)
    axes[0].set_ylabel("write ops/s, Median je Level")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=RAMP_2[0],
            linestyle="",
            markersize=7,
            label="zero-shot-Epoche",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color=RAMP_2[1],
            linestyle="",
            markersize=7,
            label="multi-shot-Epoche",
        ),
    ]
    fig.legend(handles=handles, loc="lower right", ncol=2, bbox_to_anchor=(0.99, 0.0))
    _titles(
        fig,
        "Verschiebt Multi-Shot den Throughput? Median je Level, zero-shot- gegen multi-shot-Epoche",
        bottom=0.08,
        top=0.85,
    )
    return _finish(
        fig,
        out_dir / "auswertung_zeroshot_vs_multishot.png",
        subtitle=(
            "Je Panel dasselbe Modell und derselbe Dialekt; Punkte sind Mediane zweier Epochen "
            "nebeneinander, nicht gepoolt. Vorsicht bei flash: E8/E9 hatten nur Syntax-Feedback, "
            "E12/E13 Ausführungs-Feedback (L4/L5-Regime nicht identisch)."
        ),
        footer="Tabelle: tables/auswertung_je_level.csv (Spalten model, prompting)",
    )


def _fig_latency_summary(by_epoch: dict[int, list[RunRecord]], out_dir: Path) -> str:
    rows = (("latency_p50_ms", "p50"), ("latency_p95_ms", "p95"))
    fig, axes = plt.subplots(len(rows), len(EPOCHS), figsize=(16, 5.6), sharex=True)
    x = np.arange(len(LEVEL_ORDER))
    row_top = [0.0 for _ in rows]
    for col, epoch in enumerate(EPOCHS):
        runs = _model_runs(by_epoch[epoch])
        for row, (metric, percentile_label) in enumerate(rows):
            ax = axes[row][col]
            for xi, level in zip(x, LEVEL_ORDER, strict=True):
                values = _values(runs, level, "write", metric)
                if not values:
                    continue
                q1, q2, q3 = _quartiles(values)
                row_top[row] = max(row_top[row], q3)
                ax.vlines(xi, q1, q3, color=SERIES_1, linewidth=2, zorder=3)
                ax.scatter(
                    [xi], [q2], s=38, color=SERIES_1, edgecolors=SURFACE, linewidths=1.2, zorder=4
                )
            ax.set_xticks(x)
            ax.set_xticklabels(LEVEL_ORDER)
            ax.tick_params(axis="both", length=0)
            if row == 0:
                meta = EPOCH_META[epoch]
                ax.set_title(
                    f"E{epoch} · {meta['model']}\n{meta['dialect']}", loc="left", fontsize=8.5
                )
            if col == 0:
                ax.set_ylabel(f"write {percentile_label} ms, Median + IQR")
    for row in range(len(rows)):
        for col in range(len(EPOCHS)):
            axes[row][col].set_ylim(0, row_top[row] * 1.12)
            if col:
                axes[row][col].set_yticklabels([])
    _titles(
        fig,
        "Latenz der write-Requests je Level: p50 (oben) und p95 (unten), eine Spalte je Epoche",
        bottom=0.04,
        top=0.89,
    )
    return _finish(
        fig,
        out_dir / "auswertung_latenz.png",
        subtitle="Median und IQR über die Stufe-B-Runs je Level; einzelne Runs in E<n>_latenz.png.",
        footer="Tabelle: tables/auswertung_je_level.csv",
    )


def _write_run_table(path: Path, by_epoch: dict[int, list[RunRecord]]) -> None:
    columns = [
        "epoch",
        "run_id",
        "condition_name",
        "guidance_level",
        "model",
        "prompting",
        "repetition",
        "status",
        "success",
        "executed_ok",
    ]
    for op in REQUIRED_OPERATIONS:
        columns += [
            f"{op}_throughput_ops_s",
            f"{op}_p50_ms",
            f"{op}_p95_ms",
            f"{op}_p99_ms",
            f"{op}_n_requests",
            f"{op}_error_count",
        ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for epoch in EPOCHS:
            for run in sorted(by_epoch[epoch], key=lambda run: run.run_id):
                row: list[Any] = [
                    epoch,
                    run.run_id,
                    run.condition_name,
                    run.guidance_level,
                    run.model,
                    run.prompting,
                    run.repetition,
                    run.status,
                    run.success,
                    run.executed_ok,
                ]
                for op in REQUIRED_OPERATIONS:
                    op_stats = run.op(SOURCE_PROXY, op)
                    row += [
                        run.stage_b_value(SOURCE_PROXY, op, "throughput_ops_s"),
                        run.stage_b_value(SOURCE_PROXY, op, "latency_p50_ms"),
                        run.stage_b_value(SOURCE_PROXY, op, "latency_p95_ms"),
                        run.stage_b_value(SOURCE_PROXY, op, "latency_p99_ms"),
                        op_stats.n_requests if op_stats else None,
                        op_stats.error_count if op_stats else None,
                    ]
                writer.writerow(["" if value is None else value for value in row])


def _write_level_table(path: Path, by_epoch: dict[int, list[RunRecord]]) -> None:
    columns = ["epoch", "model", "prompting", "guidance_level", "n_runs", "n_success", "n_stage_b"]
    for op in REQUIRED_OPERATIONS:
        columns += [
            f"{op}_median_throughput",
            f"{op}_q1_throughput",
            f"{op}_q3_throughput",
            f"{op}_median_p50_ms",
            f"{op}_median_p95_ms",
            f"{op}_median_p99_ms",
        ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for epoch in EPOCHS:
            runs = _model_runs(by_epoch[epoch])
            groups: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
            for run in runs:
                groups[(run.model or "?", run.prompting or "?")].append(run)
            for (model, prompting), members in sorted(groups.items()):
                for level in LEVEL_ORDER:
                    cell = [run for run in members if run.guidance_level == level]
                    if not cell:
                        continue
                    row: list[Any] = [
                        epoch,
                        model,
                        prompting,
                        level,
                        len(cell),
                        sum(1 for run in cell if run.success),
                        len(_values(cell, level, "write", "throughput_ops_s")),
                    ]
                    for op in REQUIRED_OPERATIONS:
                        throughputs = _values(cell, level, op, "throughput_ops_s")
                        q1, q2, q3 = _quartiles(throughputs) if throughputs else (None, None, None)
                        row += [q2, q1, q3]
                        for metric, _ in PERCENTILES:
                            values = _values(cell, level, op, metric)
                            row.append(median(values) if values else None)
                    writer.writerow(
                        [
                            ""
                            if value is None
                            else (f"{value:.4g}" if isinstance(value, float) else value)
                            for value in row
                        ]
                    )


def _write_spearman_table(path: Path, stats: dict[int, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "operation",
                "n_stage_b_runs",
                "rho",
                "ci95_low",
                "ci95_high",
                "p_value",
                "kruskal_wallis_H",
                "kruskal_wallis_p",
            ]
        )
        for epoch in EPOCHS:
            for op in REQUIRED_OPERATIONS:
                report = stats[epoch][op]
                spearman = _spearman_test(report)
                kruskal = next(
                    (test for test in report.tests if test.name == "Kruskal-Wallis"), None
                )
                rho = spearman.statistic if spearman else None
                lo: float | None = None
                hi: float | None = None
                if spearman and rho is not None:
                    lo, hi = _fisher_ci(float(rho), spearman.n)
                writer.writerow(
                    [
                        epoch,
                        op,
                        spearman.n if spearman else "",
                        _cell(rho),
                        _cell(lo),
                        _cell(hi),
                        _cell(spearman.p_value if spearman else None),
                        _cell(kruskal.statistic if kruskal else None),
                        _cell(kruskal.p_value if kruskal else None),
                    ]
                )


def _write_ceiling_table(path: Path, ceilings: dict[int, dict[str, dict[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "dialect",
                "operation",
                "n",
                "median_throughput_ops_s",
                "median_p50_ms",
                "median_p95_ms",
                "median_p99_ms",
            ]
        )
        for epoch in EPOCHS:
            for op, entry in ceilings[epoch].items():
                writer.writerow(
                    [
                        epoch,
                        EPOCH_META[epoch]["dialect"],
                        op,
                        entry["n"],
                        _cell(entry["median_throughput_ops_s"]),
                        _cell(entry["median_latency_p50_ms"]),
                        _cell(entry["median_latency_p95_ms"]),
                        _cell(entry["median_latency_p99_ms"]),
                    ]
                )


def _write_final_table(
    path: Path,
    by_epoch: dict[int, list[RunRecord]],
    ceilings: dict[int, dict[str, dict[str, Any]]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "dialect",
                "models",
                "regime",
                "level",
                "n_stage_b",
                "write_median_ops_s",
                "write_q1_ops_s",
                "write_q3_ops_s",
                "ceiling_write_median_ops_s",
            ]
        )
        for epoch in EPOCHS:
            runs = _model_runs(by_epoch[epoch])
            meta = EPOCH_META[epoch]
            ceil = ceilings[epoch].get("write", {}).get("median_throughput_ops_s")
            for level in LEVEL_ORDER:
                values = _values(runs, level, "write", "throughput_ops_s")
                q1, q2, q3 = _quartiles(values) if values else (None, None, None)
                writer.writerow(
                    [
                        epoch,
                        meta["dialect"],
                        meta["model"],
                        meta["regime"],
                        level,
                        len(values),
                        _cell(q2),
                        _cell(q1),
                        _cell(q3),
                        _cell(ceil),
                    ]
                )


def _cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return value


def _write_readme(
    out_dir: Path,
    written: Sequence[tuple[str, str]],
    by_epoch: dict[int, list[RunRecord]],
    stats: dict[int, dict[str, Any]],
    ceilings: dict[int, dict[str, dict[str, Any]]],
) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Performance-Auswertung (Stufe B): Gemini-Modelle",
        "",
        f"Erzeugt {stamp} mit `uv run python scripts/performance_report.py`. Quelle: "
        "`results/raw/metrics.csv` + `results/artifacts/` über `evaluation/runs.py`, Tests aus "
        "`evaluation/statistics.py`, Ceiling aus `dashboard/build.py:ceiling_summary`, also dieselben "
        "Rechenwege wie das Dashboard, keine eigene Statistik.",
        "",
        "Umfang: Epochen 6–13 (Workload v3, N = 10 je Zelle), die einzigen Epochen, in denen "
        "Stufe B Verteilungen misst (10 Write-Batches, 20 Reads je Run). Epochen werden nie "
        "gepoolt: jede Abbildung zeigt eine Epoche je Bild oder eine Epoche je Spalte. "
        "`flash` = gemini-3.5-flash, `flash-lite` = gemini-3.5-flash-lite.",
        "",
        "| Epoche | Modell | Dialekt | Regime | Runs | Stufe-B-Runs (write) |",
        "|---|---|---|---|---|---|",
    ]
    for epoch in EPOCHS:
        runs = _model_runs(by_epoch[epoch])
        n_b = sum(len(_values(runs, level, "write", "throughput_ops_s")) for level in LEVEL_ORDER)
        meta = EPOCH_META[epoch]
        lines.append(
            f"| {epoch} | {meta['model']} | {meta['dialect']} | {meta['regime']} | "
            f"{len(runs)} | {n_b} |"
        )
    lines += [
        "",
        "## Bilder",
        "",
        "### Einzelne Messwerte (ein Punkt je Run)",
        "",
    ]
    for name, description in written:
        if name.startswith("E"):
            lines.append(f"- `{name}`: {description}")
    lines += ["", "### Auswertung", ""]
    for name, description in written:
        if name.startswith("auswertung"):
            lines.append(f"- `{name}`: {description}")
    lines += [
        "",
        "## Kernzahlen",
        "",
        "Spearman ρ (Guidance-Level 1…5 ↔ write-Throughput je Stufe-B-Run), unkorrigiert:",
        "",
        "| Epoche | n | ρ | 95-%-KI | p |",
        "|---|---|---|---|---|",
    ]
    for epoch in EPOCHS:
        spearman = _spearman_test(stats[epoch]["write"])
        if spearman is None or spearman.statistic is None:
            lines.append(f"| {epoch} | – | – | – | – |")
            continue
        lo, hi = _fisher_ci(float(spearman.statistic), spearman.n)
        lines.append(
            f"| {epoch} | {spearman.n} | {_fmt(float(spearman.statistic), 2)} | "
            f"{_fmt(lo, 2)} … {_fmt(hi, 2)} | {_fmt(spearman.p_value, 3)} |"
        )
    lines += ["", "bddbench-Referenz (Median über N = 10 Läufe durch denselben Messkern):", ""]
    lines += [
        "| Epoche | Dialekt | write ops/s | read ops/s | delete ops/s | write p50 / p95 ms |",
        "|---|---|---|---|---|---|",
    ]
    for epoch in EPOCHS:
        ceiling = ceilings[epoch]
        if not ceiling:
            continue
        write = ceiling.get("write", {})
        read = ceiling.get("read", {})
        delete = ceiling.get("delete", {})
        tp = "median_throughput_ops_s"
        lines.append(
            f"| {epoch} | {EPOCH_META[epoch]['dialect']} | {_fmt(write.get(tp))} | "
            f"{_fmt(read.get(tp))} | {_fmt(delete.get(tp))} | "
            f"{_fmt(write.get('median_latency_p50_ms'))} / {_fmt(write.get('median_latency_p95_ms'))} |"
        )
    lines += [
        "",
        "## Tabellen (Zahlen hinter jedem Bild)",
        "",
        "- `tables/messwerte_je_run.csv`: jeder Run der Epochen 6–13 mit Throughput/p50/p95/p99 "
        "je Operation (Stufe-B-Regel: nur executed_ok und >= 1 fehlerfreier Request, sonst leer).",
        "- `tables/auswertung_je_level.csv`: je Epoche × Modell × Prompting × Level: n, Erfolge, "
        "Median/Q1/Q3 Throughput und Median-Latenzen je Operation.",
        "- `tables/spearman.csv`: Spearman ρ mit 95-%-KI und Kruskal-Wallis je Epoche × Operation.",
        "- `tables/ceiling.csv`: bddbench-Referenz je Epoche × Operation.",
        "- `tables/final_level_vs_performance.csv`: je Epoche × Level: n, Median/Q1/Q3 des "
        "write-Throughputs und die Referenz der Epoche (Zahlen hinter dem Gesamtbild).",
        "",
        "## Lesehinweise",
        "",
        "- Stufe-B-Werte gibt es nur für Runs, die ausgeführt wurden und >= 1 fehlerfreien Request "
        "der Operation hatten; Fehlläufe sind NA, nie 0. L1 hat in jeder Epoche 0 Stufe-B-Runs.",
        "- L3 enthält in E6–E11 zero- und multi-shot gemeinsam (je 10 + 10 Runs); E12/E13 "
        "enthalten nur multi-shot, dafür beide Modelle je Epoche.",
        "- Messrauschen: identische Artefakte streuen bei Wiederholung um bis zu ±50 % "
        "(Auswertung_und_Pipeline.md, Abschnitt 5); Unterschiede innerhalb dieses Bands sind keine "
        "Leistungsaussage.",
        "- Einordnung der Befunde: Auswertung_und_Pipeline.md, Punkte 13–16.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
