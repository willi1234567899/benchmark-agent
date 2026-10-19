from __future__ import annotations

import csv
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LightSource, Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy import stats

from benchmark_agent.evaluation.seed_matrix import (
    LEVELS,
    MODELS,
    PROMPTS,
    REF_OPS,
    ROW_LABELS,
    MetricName,
    cell_value,
    generate_reps,
    latency_p95_ratio,
    row_to_model_prompt,
    server_ops,
    summarize,
)

NAVY = "#1B3355"
TEAL = "#118286"
ORANGE = "#E0841B"
BLUE = "#4C6FBF"
PURPLE = "#7B5EA7"
ROSE = "#C45C6A"
GOLD = "#C9A227"
SLATE = "#5C6B7A"
BG = "#F7F8FA"
DISCLAIMER = "Synthetische Seed-Daten · write · Proxy · Platzhalter für metrics.csv"
SYNTHETIC_COLUMN = "synthetic"
SERIES_COLORS = (NAVY, TEAL, ORANGE, BLUE, PURPLE, ROSE)
CMAP = plt.colormaps["turbo"]
NORM = Normalize(vmin=0.0, vmax=1.0)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ("Segoe UI", "DejaVu Sans", "Arial"),
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlecolor": NAVY,
            "axes.labelcolor": SLATE,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#D5DBE3",
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "savefig.pad_inches": 0.18,
        }
    )


def _footnote(fig: Figure) -> None:
    fig.text(0.01, 0.01, DISCLAIMER, fontsize=7.5, color=SLATE, ha="left", va="bottom")


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _gemini_zero_shot_ratios(guidance: int) -> list[float]:
    return [rep.ratio for rep in generate_reps(guidance, 0, 0) if rep.ratio is not None]


def _proxy_server_pairs() -> tuple[list[float], list[float]]:
    proxy: list[float] = []
    server: list[float] = []
    for guidance in range(5):
        for model in range(3):
            for prompting in range(2):
                summary = summarize(guidance, model, prompting, 8)
                if summary.median_ratio is None:
                    continue
                ops = REF_OPS * summary.median_ratio
                proxy.append(ops)
                server.append(server_ops(ops, guidance * 100 + model * 20 + prompting))
    return proxy, server


def _style_3d(ax: Axes3D, zlabel: str) -> None:
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.45, 1.7, 0.95))
    ax.set_xlim(-0.4, 4.9)
    ax.set_ylim(-0.4, 5.9)
    ax.set_zlim(0.0, 1.02)
    ax.set_xticks(range(5), LEVELS)
    ax.set_yticks(range(6), ROW_LABELS, fontsize=6.5)
    ax.set_xlabel("Guidance", labelpad=6)
    ax.set_ylabel("Modell × Prompting", labelpad=10)
    ax.set_zlabel(zlabel, labelpad=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("#E2E6EC")
        axis.pane.set_alpha(1.0)
    ax.grid(True, color="#E8ECF1", linewidth=0.4)
    ax.tick_params(pad=1, colors=SLATE)
    ax.xaxis.set_pane_color((0.97, 0.97, 0.98, 1.0))
    ax.yaxis.set_pane_color((0.97, 0.97, 0.98, 1.0))
    ax.zaxis.set_pane_color((0.97, 0.97, 0.98, 1.0))


def _draw_bars3d(ax: Axes3D, n: int, metric: MetricName) -> None:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    dz: list[float] = []
    colors: list[tuple[float, ...]] = []
    for row in range(6):
        model, prompting = row_to_model_prompt(row)
        for guidance in range(5):
            value = cell_value(guidance, model, prompting, n, metric)
            if value is None:
                continue
            xs.append(guidance - 0.32)
            ys.append(row - 0.32)
            zs.append(0.0)
            dz.append(float(value))
            colors.append(tuple(CMAP(NORM(value))))
    if not xs:
        return
    ax.bar3d(
        xs,
        ys,
        zs,
        0.64,
        0.64,
        dz,
        color=colors,
        shade=True,
        lightsource=LightSource(azdeg=315, altdeg=45),
        edgecolor="white",
        linewidth=0.25,
        alpha=0.95,
    )
    _style_3d(ax, "Median-Ratio" if metric == "ratio" else "k / N")


def _heatmap(ax: Axes, n: int, metric: MetricName, *, annotate: bool = True) -> AxesImage:
    grid = np.full((6, 5), np.nan)
    for row in range(6):
        model, prompting = row_to_model_prompt(row)
        for guidance in range(5):
            value = cell_value(guidance, model, prompting, n, metric)
            if value is not None:
                grid[row, guidance] = value
    cmap = CMAP.with_extremes(bad="#E6E9EE")
    image = ax.imshow(grid, cmap=cmap, norm=NORM, aspect="equal", origin="upper")
    ax.set_xticks(range(5), LEVELS)
    ax.set_yticks(range(6), ROW_LABELS)
    ax.set_xlabel("Guidance")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#D5DBE3")
    if annotate:
        for row in range(6):
            for guidance in range(5):
                value = grid[row, guidance]
                if np.isnan(value):
                    ax.text(
                        guidance,
                        row,
                        "NA",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#8A94A1",
                    )
                    continue
                tone = "white" if value > 0.55 else NAVY
                ax.text(
                    guidance,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=tone,
                )
    ax.set_facecolor(BG)
    return image


def plot_triptych(output: Path, metric: MetricName, n_center: int = 8) -> None:
    z_name = "Median throughput_ratio" if metric == "ratio" else "Erfolgsquote k/N"
    fig = plt.figure(figsize=(14.8, 4.9), facecolor="white")
    grid = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.05, 1.0, 0.045], wspace=0.28)
    left = cast(Axes3D, fig.add_subplot(grid[0, 0], projection="3d"))
    mid = cast(Axes3D, fig.add_subplot(grid[0, 1], projection="3d"))
    right = fig.add_subplot(grid[0, 2])
    cax = fig.add_subplot(grid[0, 3])
    _draw_bars3d(left, 1, metric)
    _draw_bars3d(mid, n_center, metric)
    image = _heatmap(right, n_center, metric)
    left.set_title("N = 1  ·  schmaler Peak")
    mid.set_title(f"N = {n_center}  ·  nach Wiederholungen")
    right.set_title("Draufsicht derselben Matrix")
    fig.colorbar(image, cax=cax)
    cax.set_ylabel(z_name, rotation=270, labelpad=14, color=SLATE)
    fig.suptitle(
        f"3D-Histogramm + Heatmap  ·  {z_name}",
        fontsize=13,
        color=NAVY,
        x=0.48,
        y=1.02,
    )
    _footnote(fig)
    _save(fig, output)


def _funnel_counts(guidance: int, model: int = 0, prompting: int = 0, n: int = 8) -> dict[str, int]:
    counts = {"syntax": 0, "execute": 0, "coverage": 0, "success": 0}
    for rep in generate_reps(guidance, model, prompting)[:n]:
        if not rep.syntax_ok:
            counts["syntax"] += 1
        elif not rep.executed_ok:
            counts["execute"] += 1
        elif rep.n_requests < 100:
            counts["coverage"] += 1
        else:
            counts["success"] += 1
    return counts


def plot_funnel(output: Path) -> None:
    labels = list(LEVELS)
    buckets = ("syntax", "execute", "coverage", "success")
    names = ("Syntax-Fail", "Execute-Fail", "Coverage-Lücke", "Stufe B voll")
    colors = (ROSE, ORANGE, GOLD, TEAL)
    stacked = {key: [_funnel_counts(guidance)[key] for guidance in range(5)] for key in buckets}
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    bottoms = [0] * 5
    x = np.arange(5)
    for key, name, color in zip(buckets, names, colors, strict=True):
        values = stacked[key]
        ax.bar(x, values, bottom=bottoms, color=color, width=0.62, label=name, edgecolor="white")
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    ax.set_xticks(x, labels)
    ax.set_ylabel("Wiederholungen (N = 8)")
    ax.set_ylim(0, 8.4)
    ax.set_title("Stufe A: Trichter  ·  Gemini zero-shot")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.yaxis.grid(True, color="#EEF1F5", linewidth=0.8)
    ax.set_axisbelow(True)
    _footnote(fig)
    _save(fig, output)


def plot_success_heatmap(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = _heatmap(ax, 8, "success")
    ax.set_title("Stufe A: Erfolgsquote k/N  ·  N = 8")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="k / N")
    _footnote(fig)
    _save(fig, output)


def plot_ratio_vs_guidance(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(1, 6)
    for row, color in enumerate(SERIES_COLORS):
        model, prompting = row_to_model_prompt(row)
        medians = [cell_value(guidance, model, prompting, 8, "ratio") for guidance in range(5)]
        xs = [xi for xi, median in zip(x, medians, strict=True) if median is not None]
        ys = [median for median in medians if median is not None]
        marker = "o" if prompting == 0 else "s"
        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            linewidth=1.7,
            markersize=6,
            label=ROW_LABELS[row],
        )
    ax.axhline(1.0, color=NAVY, linestyle="--", linewidth=1.1, label="Ceiling (Referenz = 1)")
    ax.set_xticks(x, LEVELS)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Median throughput_ratio")
    ax.set_title("Stufe B: Nähe zur Referenz über die Guidance-Leiter")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.yaxis.grid(True, color="#EEF1F5", linewidth=0.8)
    ax.set_axisbelow(True)
    _footnote(fig)
    _save(fig, output)


def plot_naive_vs_median(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(5)
    naive = [summarize(guidance, 0, 0, 8).naive_mean_ratio for guidance in range(5)]
    correct = [summarize(guidance, 0, 0, 8).median_ratio for guidance in range(5)]
    ax.plot(x, naive, color=ROSE, marker="x", linewidth=1.7, label="Fails als 0 (Mittelwert)")
    ax.plot(
        x,
        [value if value is not None else np.nan for value in correct],
        color=TEAL,
        marker="o",
        linewidth=1.7,
        label="Fails als NA (Median der Erfolge)",
    )
    ax.set_xticks(x, LEVELS)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("throughput_ratio")
    ax.set_title("Gemini zero-shot: Fails als 0 verzerren die Qualität nach unten")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#EEF1F5", linewidth=0.8)
    ax.set_axisbelow(True)
    _footnote(fig)
    _save(fig, output)


def plot_boxplot_gemini_zs(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    data = [_gemini_zero_shot_ratios(guidance) for guidance in range(5)]
    box = ax.boxplot(
        data,
        tick_labels=list(LEVELS),
        patch_artist=True,
        widths=0.55,
        medianprops={"color": NAVY, "linewidth": 1.6},
        whiskerprops={"color": SLATE},
        capprops={"color": SLATE},
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": ORANGE,
            "markeredgecolor": ORANGE,
        },
    )
    for patch, guidance in zip(box["boxes"], range(5), strict=True):
        patch.set_facecolor(CMAP(NORM(0.2 + guidance * 0.18)))
        patch.set_edgecolor("white")
        patch.set_alpha(0.9)
    ax.axhline(1.0, color=NAVY, linestyle="--", linewidth=1.0)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("throughput_ratio (Stufe B)")
    ax.set_title("Gemini zero-shot: Spread über 8 Wiederholungen")
    ax.yaxis.grid(True, color="#EEF1F5", linewidth=0.8)
    ax.set_axisbelow(True)
    _footnote(fig)
    _save(fig, output)


def plot_latency(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(1, 6)
    for row, color in enumerate(SERIES_COLORS):
        model, prompting = row_to_model_prompt(row)
        xs: list[int] = []
        ys: list[float] = []
        for guidance in range(5):
            ratio = cell_value(guidance, model, prompting, 8, "ratio")
            if ratio is None:
                continue
            xs.append(guidance + 1)
            ys.append(latency_p95_ratio(ratio))
        ax.plot(
            xs,
            ys,
            color=color,
            marker="o" if row % 2 == 0 else "s",
            linewidth=1.6,
            markersize=5.5,
            label=ROW_LABELS[row],
        )
    ax.axhline(1.0, color=NAVY, linestyle="--", linewidth=1.1, label="Ceiling (p95 Referenz)")
    ax.set_xticks(x, LEVELS)
    ax.set_ylabel("latency_p95_ratio")
    ax.set_title("Latenz p95 relativ zur Referenz  ·  abgeleitet aus der Ratio")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.yaxis.grid(True, color="#EEF1F5", linewidth=0.8)
    ax.set_axisbelow(True)
    _footnote(fig)
    _save(fig, output)


def plot_proxy_vs_server(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    proxy, server = _proxy_server_pairs()
    ax.scatter(proxy, server, c=TEAL, s=42, alpha=0.85, edgecolors="white", linewidths=0.6)
    lo = min(proxy + server)
    hi = max(proxy + server)
    ax.plot([lo, hi], [lo, hi], color=NAVY, linestyle="--", linewidth=1.1, label="Identität")
    ax.set_xlabel("Proxy ops/s (primär)")
    ax.set_ylabel("Server /metrics ops/s (Cross-Check)")
    ax.set_title("Methoden-Validität: Proxy vs. Server")
    ax.legend(frameon=False)
    ax.set_aspect("equal", adjustable="box")
    ax.yaxis.grid(True, color="#EEF1F5", linewidth=0.8)
    ax.xaxis.grid(True, color="#EEF1F5", linewidth=0.8)
    ax.set_axisbelow(True)
    _footnote(fig)
    _save(fig, output)


def plot_zs_vs_ms(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(5)
    width = 0.36
    zero_shot = [cell_value(guidance, 0, 0, 8, "ratio") or 0.0 for guidance in range(5)]
    multi_shot = [cell_value(guidance, 0, 1, 8, "ratio") or 0.0 for guidance in range(5)]
    ax.bar(x - width / 2, zero_shot, width, color=NAVY, label="Gemini zero-shot", edgecolor="white")
    ax.bar(
        x + width / 2, multi_shot, width, color=TEAL, label="Gemini multi-shot", edgecolor="white"
    )
    ax.axhline(1.0, color=SLATE, linestyle="--", linewidth=1.0)
    ax.set_xticks(x, LEVELS)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Median throughput_ratio")
    ax.set_title("zero-shot vs. multi-shot  ·  Gemini  ·  N = 8")
    ax.legend(frameon=False)
    ax.yaxis.grid(True, color="#EEF1F5", linewidth=0.8)
    ax.set_axisbelow(True)
    _footnote(fig)
    _save(fig, output)


def plot_run_table(output: Path) -> None:
    summary = summarize(1, 0, 0, 8)
    fig, ax = plt.subplots(figsize=(10.4, 4.2))
    ax.axis("off")
    ax.set_title("L2 · Gemini · zero-shot: acht Seed-Wiederholungen", loc="left", pad=12)
    headers = ["Rep", "syntax", "execute", "n_requests", "ops/s", "ratio", "Stufe B"]
    rows: list[list[str]] = []
    tones: list[str] = []
    for index, rep in enumerate(summary.reps, start=1):
        rows.append(
            [
                str(index),
                "true" if rep.syntax_ok else "false",
                "true" if rep.executed_ok else "false",
                str(rep.n_requests),
                "NA" if rep.ops_per_sec is None else f"{rep.ops_per_sec:.1f}",
                "NA" if rep.ratio is None else f"{rep.ratio:.2f}",
                "ja" if rep.in_stage_b else "nein",
            ]
        )
        if not rep.syntax_ok:
            tones.append("#F8D7DA")
        elif not rep.executed_ok:
            tones.append("#FCE8CC")
        elif rep.n_requests < 100:
            tones.append("#F5E6B8")
        else:
            tones.append("#D8F0EC")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.55)
    for col in range(len(headers)):
        table[0, col].set_facecolor(NAVY)
        table[0, col].set_text_props(color="white", weight="semibold")
    for row, tone in enumerate(tones, start=1):
        for col in range(len(headers)):
            table[row, col].set_facecolor(tone)
            table[row, col].set_edgecolor("white")
    median_txt = "NA" if summary.median_ratio is None else f"{summary.median_ratio:.2f}"
    ax.text(
        0.0,
        -0.08,
        f"Median der Stufe-B-Ratios: {median_txt}"
        f"   ·   Erfolge {summary.success_count}/8   ·   Fails bleiben NA",
        transform=ax.transAxes,
        color=SLATE,
        fontsize=8.5,
    )
    _footnote(fig)
    _save(fig, output)


def plot_board(output: Path) -> None:
    fig = plt.figure(figsize=(13.6, 9.2))
    axes = fig.subplots(2, 3)
    fig.suptitle("Statistisches Set: Seed-Überblick", fontsize=14, color=NAVY, y=0.98)

    ax = axes[0, 0]
    x = np.arange(5)
    bottoms = np.zeros(5)
    funnel_keys = ("syntax", "execute", "coverage", "success")
    funnel_colors = (ROSE, ORANGE, GOLD, TEAL)
    for key, color in zip(funnel_keys, funnel_colors, strict=True):
        values = np.array([_funnel_counts(guidance)[key] for guidance in range(5)])
        ax.bar(x, values, bottom=bottoms, color=color, width=0.62, edgecolor="white")
        bottoms = bottoms + values
    ax.set_xticks(x, LEVELS)
    ax.set_title("Stufe-A-Trichter")
    ax.set_ylabel("N = 8")

    ax = axes[0, 1]
    for row, color in enumerate(SERIES_COLORS):
        model, prompting = row_to_model_prompt(row)
        medians = [cell_value(guidance, model, prompting, 8, "ratio") for guidance in range(5)]
        xs = [guidance + 1 for guidance, median in enumerate(medians) if median is not None]
        ys = [median for median in medians if median is not None]
        ax.plot(xs, ys, color=color, marker="o", linewidth=1.4, markersize=4.5)
    ax.axhline(1.0, color=NAVY, linestyle="--", linewidth=0.9)
    ax.set_xticks(range(1, 6), LEVELS)
    ax.set_ylim(0, 1.08)
    ax.set_title("Ratio vs. Guidance")

    ax = axes[0, 2]
    naive = [summarize(guidance, 0, 0, 8).naive_mean_ratio for guidance in range(5)]
    correct = [summarize(guidance, 0, 0, 8).median_ratio or np.nan for guidance in range(5)]
    ax.plot(range(5), naive, color=ROSE, marker="x", label="Fail = 0")
    ax.plot(range(5), correct, color=TEAL, marker="o", label="Fail = NA")
    ax.set_xticks(range(5), LEVELS)
    ax.set_ylim(0, 1.05)
    ax.set_title("Fail = 0 vs. Median")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 0]
    data = [_gemini_zero_shot_ratios(guidance) for guidance in range(5)]
    ax.boxplot(
        data,
        tick_labels=list(LEVELS),
        widths=0.5,
        patch_artist=True,
        medianprops={"color": NAVY},
    )
    ax.set_ylim(0, 1.08)
    ax.set_title("Spread Gemini ZS")

    ax = axes[1, 1]
    proxy, server = _proxy_server_pairs()
    ax.scatter(proxy, server, c=TEAL, s=22, alpha=0.85, edgecolors="white")
    lo, hi = min(proxy + server), max(proxy + server)
    ax.plot([lo, hi], [lo, hi], color=NAVY, linestyle="--", linewidth=0.9)
    ax.set_title("Proxy vs. Server")
    ax.set_aspect("equal", adjustable="box")

    ax = axes[1, 2]
    _heatmap(ax, 8, "ratio", annotate=False)
    ax.set_title("Heatmap Median-Ratio")

    for axis in axes.ravel():
        axis.yaxis.grid(True, color="#EEF1F5", linewidth=0.7)
        axis.set_axisbelow(True)
    axes[1, 2].grid(False)

    fig.legend(
        handles=[
            Patch(facecolor=ROSE, label="Syntax-Fail"),
            Patch(facecolor=ORANGE, label="Execute-Fail"),
            Patch(facecolor=GOLD, label="Coverage"),
            Patch(facecolor=TEAL, label="Stufe B"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    _footnote(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _save(fig, output)


def _cell_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for guidance in range(5):
        for model in range(3):
            for prompting in range(2):
                summary = summarize(guidance, model, prompting, 8)
                rows.append(
                    {
                        SYNTHETIC_COLUMN: True,
                        "guidance": LEVELS[guidance],
                        "model": MODELS[model],
                        "prompting": PROMPTS[prompting],
                        "n": 8,
                        "k_stage_b": summary.success_count,
                        "success_rate": round(summary.success_rate, 3),
                        "median_ratio": (
                            "" if summary.median_ratio is None else round(summary.median_ratio, 4)
                        ),
                        "iqr_ratio": (
                            "" if summary.iqr_ratio is None else round(summary.iqr_ratio, 4)
                        ),
                        "naive_mean_fail_as_0": round(summary.naive_mean_ratio, 4),
                        "latency_p95_ratio": (
                            ""
                            if summary.median_ratio is None
                            else round(latency_p95_ratio(summary.median_ratio), 4)
                        ),
                    }
                )
    return rows


def write_cell_summary(path: Path) -> None:
    rows = _cell_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_run_table(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                SYNTHETIC_COLUMN,
                "rep",
                "syntax_ok",
                "executed_ok",
                "n_requests",
                "ops_per_sec",
                "ratio",
                "stage_b",
            ]
        )
        for index, rep in enumerate(summarize(1, 0, 0, 8).reps, start=1):
            writer.writerow(
                [
                    True,
                    index,
                    rep.syntax_ok,
                    rep.executed_ok,
                    rep.n_requests,
                    "" if rep.ops_per_sec is None else rep.ops_per_sec,
                    "" if rep.ratio is None else rep.ratio,
                    rep.in_stage_b,
                ]
            )


def _rank_biserial(u: float, n1: int, n2: int) -> float:
    return (2.0 * u) / (n1 * n2) - 1.0


def write_inferential(path: Path) -> None:
    groups: list[list[float]] = []
    group_labels: list[str] = []
    for guidance in range(5):
        values: list[float] = []
        for model in range(3):
            for prompting in range(2):
                for rep in generate_reps(guidance, model, prompting):
                    if rep.ratio is not None:
                        values.append(rep.ratio)
        if values:
            groups.append(values)
            group_labels.append(LEVELS[guidance])
    kruskal = stats.kruskal(*groups)
    guidance_indices: list[float] = []
    median_ratios: list[float] = []
    for guidance in range(5):
        for model in range(3):
            for prompting in range(2):
                median_ratio = summarize(guidance, model, prompting, 8).median_ratio
                if median_ratio is not None:
                    guidance_indices.append(float(guidance + 1))
                    median_ratios.append(median_ratio)
    spearman = stats.spearmanr(guidance_indices, median_ratios)
    zero_shot: list[float] = []
    multi_shot: list[float] = []
    for guidance in range(1, 5):
        zero_shot.extend(
            rep.ratio for rep in generate_reps(guidance, 0, 0) if rep.ratio is not None
        )
        multi_shot.extend(
            rep.ratio for rep in generate_reps(guidance, 0, 1) if rep.ratio is not None
        )
    mannwhitney = stats.mannwhitneyu(zero_shot, multi_shot, alternative="two-sided")
    rbc = _rank_biserial(float(mannwhitney.statistic), len(zero_shot), len(multi_shot))
    l2 = summarize(1, 0, 0, 8)
    median_txt = "NA" if l2.median_ratio is None else f"{l2.median_ratio:.2f}"
    iqr_txt = "NA" if l2.iqr_ratio is None else f"{l2.iqr_ratio:.2f}"
    filled = sum(1 for row in _cell_rows() if row["median_ratio"] != "")
    lines = [
        "INFERENZ AUF SYNTHETISCHEN SEED-DATEN: kein Thesis-Ergebnis.",
        "N=8, write, Proxy. Tests nur als Layout für statistics.py, sobald metrics.csv existiert.",
        "",
        f"Zellen mit Median-Ratio (N=8): {filled}/30",
        f"L2 Gemini ZS: k={l2.success_count}/8, Median-Ratio={median_txt}, IQR={iqr_txt}",
        "",
        "Kruskal-Wallis: throughput_ratio ~ Guidance (alle Modelle/Prompting, nur Stufe B)",
        f"  Gruppen: {', '.join(group_labels)}",
        f"  H = {kruskal.statistic:.3f}, p = {kruskal.pvalue:.4g}",
        "",
        "Spearman: Guidance-Index vs. Zellen-Median-Ratio",
        f"  ρ = {spearman.statistic:.3f}, p = {spearman.pvalue:.4g}",
        "",
        "Mann-Whitney U: Gemini zero-shot vs. multi-shot (L2–L5, Stufe-B-Ratios)",
        f"  n_ZS = {len(zero_shot)}, n_MS = {len(multi_shot)}",
        f"  U = {mannwhitney.statistic:.1f}, p = {mannwhitney.pvalue:.4g}",
        f"  Rang-biseriale Korrelation r = {rbc:.3f}  (positiv: ZS > MS)",
        "",
        "Hinweis: Bei echten Daten N≥5, Failures als NA (nicht 0), primär Proxy,",
        "/metrics nur als Cross-Check. Parametrische t-Tests erst nach Verteilungscheck.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cover_page() -> Figure:
    fig = plt.figure(figsize=(11.2, 7.4))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    l2 = summarize(1, 0, 0, 8)
    filled = sum(1 for row in _cell_rows() if row["median_ratio"] != "")
    ax.text(0.08, 0.88, "Benchmark-Agent", fontsize=13, color=SLATE)
    ax.text(0.08, 0.78, "Statistisches Set", fontsize=28, color=NAVY, fontweight="semibold")
    ax.text(0.08, 0.70, "Synthetische Seed-Daten  ·  kein Thesis-Ergebnis", fontsize=12, color=ROSE)
    ax.text(
        0.08,
        0.62,
        "write  ·  Proxy primär  ·  N = 8  ·  5 Guidance × 3 Modelle × 2 Prompting",
        fontsize=11,
        color=SLATE,
    )
    median_txt = "NA" if l2.median_ratio is None else f"{l2.median_ratio:.2f}"
    iqr_txt = "NA" if l2.iqr_ratio is None else f"{l2.iqr_ratio:.2f}"
    bullets = [
        f"Zellen mit Median-Ratio: {filled}/30",
        f"L2 Gemini zero-shot: {l2.success_count}/8 Erfolge, "
        f"Median-Ratio {median_txt}, IQR {iqr_txt}",
        "Fails bleiben NA (nicht 0), Stufe B nur bei ausführbaren Läufen",
        "Kruskal-Wallis Guidance: H = 93.88, p ≈ 2e-19  (Seed, nicht Messung)",
        "Spearman Guidance vs. Median-Ratio: ρ = 0.96",
        "Mann-Whitney Gemini ZS vs. MS (L2–L5): p = 0.35",
    ]
    y = 0.50
    for line in bullets:
        ax.text(0.08, y, line, fontsize=12, color=NAVY)
        y -= 0.055
    ax.text(0.08, 0.08, DISCLAIMER, fontsize=9, color=SLATE)
    return fig


def write_combined_pdf(output_dir: Path, figures: list[Path]) -> Path:
    pdf_path = output_dir / "Benchmark-Agent_Seed-Statistik.pdf"
    with PdfPages(pdf_path) as pdf:
        cover = _cover_page()
        pdf.savefig(cover)
        plt.close(cover)
        for png_path in figures:
            image = plt.imread(png_path)
            height, width = image.shape[:2]
            fig_w = 11.2
            fig_h = min(14.0, max(6.0, fig_w * height / width))
            fig = plt.figure(figsize=(fig_w, fig_h))
            axes = fig.add_axes((0.02, 0.05, 0.96, 0.92))
            axes.imshow(image)
            axes.axis("off")
            fig.text(0.02, 0.018, png_path.name, fontsize=8, color=SLATE)
            pdf.savefig(fig)
            plt.close(fig)
    return pdf_path


def render_all(output_dir: Path) -> list[Path]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = output_dir / "tables"
    figures = [
        output_dir / "01_triptych_ratio.png",
        output_dir / "02_triptych_success.png",
        output_dir / "03_funnel_stage_a.png",
        output_dir / "04_heatmap_success.png",
        output_dir / "05_ratio_vs_guidance.png",
        output_dir / "06_naive_vs_median.png",
        output_dir / "07_boxplot_gemini_zs.png",
        output_dir / "08_latency_p95.png",
        output_dir / "09_proxy_vs_server.png",
        output_dir / "10_zs_vs_ms.png",
        output_dir / "11_l2_gemini_runs.png",
        output_dir / "12_board.png",
    ]
    plot_triptych(figures[0], "ratio")
    plot_triptych(figures[1], "success")
    plot_funnel(figures[2])
    plot_success_heatmap(figures[3])
    plot_ratio_vs_guidance(figures[4])
    plot_naive_vs_median(figures[5])
    plot_boxplot_gemini_zs(figures[6])
    plot_latency(figures[7])
    plot_proxy_vs_server(figures[8])
    plot_zs_vs_ms(figures[9])
    plot_run_table(figures[10])
    plot_board(figures[11])
    write_cell_summary(tables / "cell_summary.csv")
    write_run_table(tables / "l2_gemini_zs.csv")
    write_inferential(tables / "inferential.txt")
    pdf_path = write_combined_pdf(output_dir, figures)
    return [*figures, pdf_path]
