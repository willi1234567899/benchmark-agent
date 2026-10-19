from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_results as tr
from scipy import stats as sps

plt = tr.plt
Line2D = tr.Line2D
Patch = tr.Patch

STAGE_B_LEVELS = tr.LEVELS[1:]
ROWS = [(m, p) for m in tr.MAIN_MODELS for p in ("zero_shot", "multi_shot")]
ROW_LABEL = {(m, p): f"{tr.MODEL_LABEL[m]} {'zs' if p == 'zero_shot' else 'ms'}" for m, p in ROWS}
N_BOOT = 2000
SEED = tr.JITTER_SEED
DIVERGING = tr.LinearSegmentedColormap.from_list(
    "div_red_blue", ["#b2302f", "#e98a89", "#f0efec", "#86b6ef", "#1c5cab"]
)


def tint(hex_color: str, amount: float) -> tuple[float, float, float]:
    hex_digits = hex_color.lstrip("#")
    rgb = np.array([int(hex_digits[i : i + 2], 16) / 255 for i in (0, 2, 4)])
    red, green, blue = rgb + (1 - rgb) * amount
    return red, green, blue


def arm_for(model: str, prompting: str, dialect: str) -> dict[str, Any]:
    return next(
        arm
        for arm in tr.ARMS
        if arm["model"] == model and arm["prompting"] == prompting and arm["dialect"] == dialect
    )


def _bootstrap_slopes(
    per_level: dict[str, np.ndarray[Any, Any]], rng: np.random.Generator
) -> list[float]:
    slopes: list[float] = []
    for _ in range(N_BOOT):
        xs, ys = [], []
        for i, values in enumerate(per_level.values()):
            if len(values):
                sample = rng.choice(values, size=len(values), replace=True)
                xs.append(np.full(len(values), i + 2))
                ys.append(sample)
        boot_x, boot_y = np.concatenate(xs), np.concatenate(ys)
        slopes.append(sps.theilslopes(np.log(boot_y), boot_x).slope)
    return slopes


def level_trend(b_cells: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    rows = []
    for arm in tr.ARMS:
        for op in ("write", "read", "delete"):
            for metric in ("tp", "p50"):
                per_level = {
                    lvl: np.asarray(b_cells[(arm["key"], lvl, op)][f"{metric}_s"], dtype=float)
                    for lvl in STAGE_B_LEVELS
                }
                x = np.concatenate(
                    [np.full(len(values), i + 2) for i, values in enumerate(per_level.values())]
                )
                y = np.concatenate(list(per_level.values()))
                row: dict[str, Any] = {
                    "arm": arm["key"],
                    "model": tr.MODEL_FULL[arm["model"]],
                    "prompting": arm["prompting"],
                    "dialect": arm["dialect"],
                    "epoch": arm["epoch"],
                    "op": op,
                    "metric": "throughput" if metric == "tp" else "latency_p50",
                    "n": len(y),
                    **{f"n_{lvl}": len(values) for lvl, values in per_level.items()},
                    **{
                        f"median_{lvl}": float(np.median(values)) if len(values) else None
                        for lvl, values in per_level.items()
                    },
                }
                base = row["median_L2"]
                for lvl in STAGE_B_LEVELS:
                    median = row[f"median_{lvl}"]
                    row[f"ratio_{lvl}_to_L2"] = (median / base) if (median and base) else None
                if len(set(x)) >= 2 and len(y) >= 6:
                    slope = sps.theilslopes(np.log(y), x).slope
                    boot_slopes = _bootstrap_slopes(per_level, rng)
                    lo, hi = np.percentile(boot_slopes, [2.5, 97.5])
                    spearman = sps.spearmanr(x, y)
                    row.update(
                        {
                            "pct_per_level": (np.exp(slope) - 1) * 100,
                            "ci_lo": (np.exp(lo) - 1) * 100,
                            "ci_hi": (np.exp(hi) - 1) * 100,
                            "pct_L2_to_L5": (np.exp(3 * slope) - 1) * 100,
                            "spearman_rho": float(spearman.statistic),
                            "spearman_p": float(spearman.pvalue),
                        }
                    )
                    better_up = metric == "tp"
                    if lo > 0:
                        row["direction"] = "faster with level" if better_up else "slower with level"
                    elif hi < 0:
                        row["direction"] = "slower with level" if better_up else "faster with level"
                    else:
                        row["direction"] = "no clear change"
                rows.append(row)
    return rows


def fig_level_trend(rows: list[dict[str, Any]], out: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(tr.TEXT_WIDTH_IN, 5.0), sharey=True)
    arms = tr.ARMS
    for row_index, metric in enumerate(("throughput", "latency_p50")):
        for col_index, op in enumerate(("write", "read")):
            ax = axes[row_index][col_index]
            for i, arm in enumerate(arms):
                row = next(
                    candidate
                    for candidate in rows
                    if candidate["arm"] == arm["key"]
                    and candidate["op"] == op
                    and candidate["metric"] == metric
                )
                if row.get("pct_per_level") is None:
                    continue
                color = tr.MODEL_COLOR[arm["model"]]
                ax.plot([row["ci_lo"], row["ci_hi"]], [i, i], color=color, lw=1.1, zorder=2)
                clear = row["direction"] != "no clear change"
                ax.scatter(
                    [row["pct_per_level"]],
                    [i],
                    marker=tr.PROMPT_MARKER[arm["prompting"]],
                    s=26,
                    facecolor=color if clear else tr.SURFACE,
                    edgecolor=color,
                    linewidth=1.0,
                    zorder=3,
                )
            ax.axvline(0, color=tr.INK_2, lw=0.9, ls=(0, (4, 2)), zorder=1)
            tr._model_separators(ax, arms, -0.5)
            ax.set_title(f"{op} {'throughput' if metric == 'throughput' else 'latency p50'}")
            ax.grid(False)
            ax.xaxis.grid(True)
            ax.tick_params(axis="y", length=0)
            ax.spines["bottom"].set_color(tr.BASELINE)
            if row_index == 1:
                ax.set_xlabel("change per guidance level (%)")
            lim = max(
                abs(bound)
                for trend_row in rows
                if trend_row["op"] == op and trend_row["metric"] == metric
                for bound in (trend_row.get("ci_lo") or 0, trend_row.get("ci_hi") or 0)
            )
            ax.set_xlim(-lim * 1.08, lim * 1.08)
    axes[0][0].set_yticks(range(len(arms)), [arm["label"] for arm in arms])
    axes[0][0].invert_yaxis()
    handles: list[Any] = [
        Line2D(
            [],
            [],
            marker="o",
            color=tr.MODEL_COLOR[model],
            linestyle="",
            markersize=5,
            label=tr.MODEL_FULL[model],
        )
        for model in tr.MAIN_MODELS
    ]
    handles += [
        Line2D([], [], marker="s", color=tr.INK_2, linestyle="", markersize=5, label="zero-shot"),
        Line2D([], [], marker="o", color=tr.INK_2, linestyle="", markersize=5, label="multi-shot"),
        Line2D(
            [],
            [],
            marker="o",
            color=tr.INK_2,
            markerfacecolor=tr.SURFACE,
            linestyle="",
            markersize=5,
            label="hollow: 95 % interval includes 0",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), handlelength=1.2
    )
    fig.subplots_adjust(wspace=0.08, hspace=0.38, top=0.9)
    tr._header(fig, "Stage B, Stage-A successes only: change of the measurement from L2 to L5")
    return tr.save(fig, out, "stage_b_level_trend")


def noise_band(
    reference: dict[str, dict[str, dict[str, Any]]], op: str = "write"
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    centred = []
    for series in reference.values():
        logs = np.log(np.asarray(series[op]["tp"], dtype=float))
        centred.append(logs - np.median(logs))
    pool = np.concatenate(centred)
    ratios = [
        np.exp(np.median(rng.choice(pool, 10)) - np.median(rng.choice(pool, 10)))
        for _ in range(20000)
    ]
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return float(lo - 1), float(hi - 1)


def fig_level_profile(rows: list[dict[str, Any]], band: tuple[float, float], out: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(tr.TEXT_WIDTH_IN, 4.2), sharex=True, sharey=True)
    x = np.arange(len(STAGE_B_LEVELS))
    for row_index, op in enumerate(("write", "read")):
        for col_index, dialect in enumerate(tr.DIALECTS):
            ax = axes[row_index][col_index]
            ax.axhspan(1 + band[0], 1 + band[1], color=tr.GRID, alpha=0.6, lw=0, zorder=0)
            ax.axhline(1, color=tr.INK_2, lw=0.8, zorder=1)
            for arm in tr.ARMS:
                if arm["dialect"] != dialect:
                    continue
                row = next(
                    candidate
                    for candidate in rows
                    if candidate["arm"] == arm["key"]
                    and candidate["op"] == op
                    and candidate["metric"] == "throughput"
                )
                ys = [row[f"ratio_{lvl}_to_L2"] for lvl in STAGE_B_LEVELS]
                ax.plot(
                    x,
                    ys,
                    color=tr.MODEL_COLOR[arm["model"]],
                    lw=1.3,
                    ls="-" if arm["prompting"] == "multi_shot" else (0, (3, 2)),
                    marker=tr.PROMPT_MARKER[arm["prompting"]],
                    markersize=4,
                    markeredgecolor=tr.SURFACE,
                    zorder=3,
                )
            ax.set_yscale("log")
            ax.set_yticks([0.5, 0.75, 1, 1.5, 2], ["0.5", "0.75", "1", "1.5", "2"])
            ax.minorticks_off()
            ax.set_xticks(x, STAGE_B_LEVELS)
            ax.tick_params(axis="x", length=0)
            ax.spines["bottom"].set_color(tr.BASELINE)
            if row_index == 0:
                ax.set_title(dialect)
            if col_index == 0:
                ax.set_ylabel(f"{op} throughput,\nmedian relative to L2")
    handles: list[Any] = [
        Line2D([], [], color=tr.MODEL_COLOR[model], lw=1.3, label=tr.MODEL_FULL[model])
        for model in tr.MAIN_MODELS
    ]
    handles += [
        Line2D(
            [],
            [],
            color=tr.INK_2,
            lw=1.3,
            ls=(0, (3, 2)),
            marker="s",
            markersize=4,
            label="zero-shot",
        ),
        Line2D([], [], color=tr.INK_2, lw=1.3, marker="o", markersize=4, label="multi-shot"),
        Patch(
            facecolor=tr.GRID,
            alpha=0.6,
            label=f"noise band, reference benchmark, median of 10 vs 10 ({band[0] * 100:+.0f} to {band[1] * 100:+.0f} %)",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), handlelength=1.8
    )
    fig.subplots_adjust(wspace=0.08, hspace=0.3, top=0.88)
    tr._header(fig, "Stage B, Stage-A successes only: cell median relative to the arm's own L2")
    return tr.save(fig, out, "stage_b_level_profile")


def load_tost() -> dict[tuple[str, str, str], dict[str, Any]]:
    path = tr.Path(
        "results/reports/thesis_results/extra/stage_b_client/tables/delta_latency_vs_reference.csv"
    )
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["metric"] == "tp_harness":
                out[(row["arm"], row["level"], row["op"])] = row
    return out


def criteria_rows(
    a_cells: dict[tuple[str, str], dict[str, Any]],
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
    reference: dict[str, dict[str, dict[str, Any]]],
    tost: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for arm in tr.ARMS:
        for lvl in tr.LEVELS:
            stage_a = a_cells[(arm["key"], lvl)]
            row: dict[str, Any] = {
                "arm": arm["key"],
                "model": tr.MODEL_FULL[arm["model"]],
                "prompting": arm["prompting"],
                "dialect": arm["dialect"],
                "epoch": arm["epoch"],
                "level": lvl,
                "stage_a_success": stage_a["success"],
                "stage_a_n": stage_a["n"],
                "stage_a_reliable": stage_a["success"] == stage_a["n"],
                "reference": arm["ref"] or "",
            }
            for op in ("write", "read"):
                cell = b_cells[(arm["key"], lvl, op)]
                values = np.asarray(cell["tp_s"], dtype=float)
                if lvl == "L1" or len(values) == 0:
                    row[f"{op}_n"] = len(values)
                    continue
                q1 = np.percentile(values, 25)
                med = float(np.median(values))
                q3 = np.percentile(values, 75)
                rel_iqr = (q3 - q1) / med
                row[f"{op}_n"] = len(values)
                row[f"{op}_median"] = med
                row[f"{op}_rel_iqr"] = rel_iqr
                if arm["ref"]:
                    ref_values = np.asarray(reference[arm["ref"]][op]["tp"], dtype=float)
                    ref_q1 = np.percentile(ref_values, 25)
                    ref_med = float(np.median(ref_values))
                    ref_q3 = np.percentile(ref_values, 75)
                    ref_rel_iqr = (ref_q3 - ref_q1) / ref_med
                    tost_row = tost.get((arm["key"], lvl, op))
                    row[f"{op}_delta_pct"] = (med - ref_med) / ref_med * 100
                    row[f"{op}_ref_rel_iqr"] = ref_rel_iqr
                    row[f"{op}_consistent"] = rel_iqr <= ref_rel_iqr
                    row[f"{op}_equivalent"] = bool(
                        tost_row and tost_row["tost_eq_success"] == "True"
                    )
                    row[f"{op}_hl_ratio"] = (
                        float(tost_row["hl_ratio_success"])
                        if tost_row and tost_row["hl_ratio_success"]
                        else None
                    )
                    row[f"{op}_both"] = row[f"{op}_consistent"] and row[f"{op}_equivalent"]
            rows.append(row)
    return rows


def point_of(rows: list[dict[str, Any]], arm_key: str, key: str) -> str | None:
    cells = {row["level"]: row for row in rows if row["arm"] == arm_key}
    for i, lvl in enumerate(tr.LEVELS):
        if all(cells[later].get(key) is True for later in tr.LEVELS[i:]):
            return lvl
    return None


def fig_overview_3d(rows: list[dict[str, Any]], dialect: str, out: Path) -> str:
    fig = plt.figure(figsize=(tr.TEXT_WIDTH_IN, 3.4))
    panels = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    dx = dy = 0.62
    norm = tr.matplotlib.colors.TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60)
    for panel_index, ax in enumerate(panels):
        for y_idx, (model, prompting) in enumerate(ROWS):
            y_i = len(ROWS) - 1 - y_idx
            arm = arm_for(model, prompting, dialect)
            for x_i, lvl in enumerate(tr.LEVELS):
                cell = next(
                    candidate
                    for candidate in rows
                    if candidate["arm"] == arm["key"] and candidate["level"] == lvl
                )
                if panel_index == 0:
                    z = cell["stage_a_success"]
                    color = tint(tr.MODEL_COLOR[model], 0.45 if prompting == "zero_shot" else 0.0)
                    ax.bar3d(
                        x_i - dx / 2,
                        y_i - dy / 2,
                        0,
                        dx,
                        dy,
                        max(z, 0.05),
                        color=color,
                        edgecolor=tr.SURFACE,
                        linewidth=0.3,
                        shade=True,
                        alpha=0.95,
                    )
                    if lvl == point_of(rows, arm["key"], "stage_a_reliable"):
                        ax.scatter([x_i], [y_i], [z + 0.6], color=tr.INK, s=10, depthshade=False)
                else:
                    delta = cell.get("write_delta_pct")
                    if delta is None:
                        ax.scatter(
                            [x_i], [y_i], [0], marker="x", color=tr.MUTED, s=10, depthshade=False
                        )
                        continue
                    color = DIVERGING(norm(max(min(delta, 60), -60)))
                    ax.bar3d(
                        x_i - dx / 2,
                        y_i - dy / 2,
                        min(delta, 0),
                        dx,
                        dy,
                        abs(delta) if abs(delta) > 0.5 else 0.5,
                        color=color,
                        edgecolor=tr.SURFACE,
                        linewidth=0.3,
                        shade=True,
                        alpha=0.95,
                    )
                    if cell.get("write_both"):
                        ax.scatter(
                            [x_i], [y_i], [max(delta, 0) + 4], color=tr.INK, s=10, depthshade=False
                        )
        ax.set_xticks(range(len(tr.LEVELS)), tr.LEVELS, fontsize=6)
        ax.set_yticks(range(len(ROWS)), [ROW_LABEL[row] for row in reversed(ROWS)], fontsize=5.5)
        ax.tick_params(axis="z", labelsize=6, pad=0)
        ax.tick_params(axis="y", pad=-2)
        ax.view_init(elev=24, azim=-58)
        ax.set_box_aspect((1.25, 1.35, 0.8))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1, 1, 1, 0))
            axis.pane.set_edgecolor(tr.GRID)
    stage_a_ax, stage_b_ax = panels
    stage_a_ax.set_zlim(0, 10.5)
    stage_a_ax.set_zlabel("Stage A successes (of 10)", fontsize=6.5, labelpad=-1)
    stage_a_ax.set_title("Stage A", fontsize=8, pad=-6)
    lim = (
        max(abs(row.get("write_delta_pct") or 0) for row in rows if row["dialect"] == dialect) * 1.1
    )
    stage_b_ax.set_zlim(-lim, lim)
    xx, yy = np.meshgrid([-0.5, len(tr.LEVELS) - 0.5], [-0.5, len(ROWS) - 0.5])
    for zplane, alpha in ((0, 0.25), (20, 0.1), (-20, 0.1)):
        stage_b_ax.plot_surface(
            xx, yy, np.full_like(xx, zplane, dtype=float), color=tr.INK_2, alpha=alpha, lw=0
        )
    stage_b_ax.set_zlabel("write throughput vs reference (%)", fontsize=6.5, labelpad=-1)
    stage_b_ax.set_title("Stage B (write)", fontsize=8, pad=-6)
    handles: list[Any] = [
        Patch(facecolor=tr.MODEL_COLOR[model], label=tr.MODEL_FULL[model])
        for model in tr.MAIN_MODELS
    ]
    handles += [
        Patch(facecolor=tint(tr.INK_2, 0.45), label="lighter: zero-shot"),
        Patch(facecolor=DIVERGING(0.95), label="Stage B: faster than the reference"),
        Patch(facecolor=DIVERGING(0.05), label="Stage B: slower than the reference"),
        Line2D(
            [],
            [],
            marker="o",
            color=tr.INK,
            linestyle="",
            markersize=4,
            label="Stage A: from here 10/10; Stage B: equivalent and consistent",
        ),
        Line2D(
            [],
            [],
            marker="x",
            color=tr.MUTED,
            linestyle="",
            markersize=4,
            label="no Stage-B value or no same-day reference",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.17),
        fontsize=6.5,
        handlelength=1.2,
    )
    fig.subplots_adjust(left=0.0, right=0.97, wspace=0.05, top=0.9, bottom=0.08)
    fig.text(
        0.01,
        0.97,
        f"{dialect} dialect: guidance level x model and prompting, Stage A and Stage B side by side",
        fontsize=7.5,
        fontweight="bold",
        ha="left",
        va="top",
        color=tr.INK,
    )
    return tr.save(fig, out, f"overview_3d_{dialect.lower()}")


def fig_overview_map(rows: list[dict[str, Any]], out: Path) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(tr.TEXT_WIDTH_IN, 4.3), sharey=True)
    norm = tr.matplotlib.colors.TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60)
    for dialect_index, dialect in enumerate(tr.DIALECTS):
        for col_index, panel in enumerate(("stage_a", "write", "read")):
            ax = axes[dialect_index][col_index]
            for y_i, (model, prompting) in enumerate(ROWS):
                arm = arm_for(model, prompting, dialect)
                for x_i, lvl in enumerate(tr.LEVELS):
                    cell = next(
                        candidate
                        for candidate in rows
                        if candidate["arm"] == arm["key"] and candidate["level"] == lvl
                    )
                    if panel == "stage_a":
                        share = cell["stage_a_success"] / cell["stage_a_n"]
                        color = tr.SEQ_CMAP(share)
                        text = f"{cell['stage_a_success']}"
                        txt_color = "white" if share > 0.6 else tr.INK
                    else:
                        delta = cell.get(f"{panel}_delta_pct")
                        if delta is None:
                            color = tr.SURFACE
                            text = "n/a" if lvl != "L1" else ""
                            txt_color = tr.MUTED
                        else:
                            color = DIVERGING(norm(max(min(delta, 60), -60)))
                            equivalent_mark = "=" if cell.get(f"{panel}_equivalent") else ""
                            consistent_mark = "c" if cell.get(f"{panel}_consistent") else ""
                            marks = equivalent_mark + consistent_mark
                            text = f"{delta:+.0f}" + (f"\n{marks}" if marks else "")
                            txt_color = tr.INK
                    ax.add_patch(
                        plt.Rectangle(
                            (x_i, y_i), 1, 1, facecolor=color, edgecolor=tr.SURFACE, lw=1.5
                        )
                    )
                    ax.text(
                        x_i + 0.5,
                        y_i + 0.5,
                        text,
                        ha="center",
                        va="center",
                        fontsize=5.5,
                        color=txt_color,
                        linespacing=0.9,
                    )
                key = "stage_a_reliable" if panel == "stage_a" else f"{panel}_both"
                point = point_of(rows, arm["key"], key)
                if point is not None:
                    x0 = tr.LEVELS.index(point)
                    ax.plot(
                        [x0, x0],
                        [y_i, y_i + 1],
                        color=tr.INK,
                        lw=2.2,
                        solid_capstyle="butt",
                        zorder=5,
                    )
                else:
                    ax.text(
                        len(tr.LEVELS) + 0.08,
                        y_i + 0.5,
                        "none",
                        fontsize=5,
                        color=tr.INK_2,
                        va="center",
                    )
            ax.set_xlim(0, len(tr.LEVELS))
            ax.set_ylim(len(ROWS), 0)
            ax.set_xticks([i + 0.5 for i in range(len(tr.LEVELS))], tr.LEVELS, fontsize=6.5)
            ax.set_yticks(
                [i + 0.5 for i in range(len(ROWS))], [ROW_LABEL[row] for row in ROWS], fontsize=6.5
            )
            ax.tick_params(length=0)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            title = {
                "stage_a": "Stage A successes",
                "write": "Stage B write: deviation (%)",
                "read": "Stage B read: deviation (%)",
            }[panel]
            ax.set_title(f"{dialect}: {title}", fontsize=7)
    handles = [
        Line2D(
            [],
            [],
            color=tr.INK,
            lw=2.2,
            label="point: from this level on every later level meets the criterion",
        ),
        Patch(facecolor=DIVERGING(0.95), label="faster than the same-day reference"),
        Patch(facecolor=DIVERGING(0.05), label="slower than the same-day reference"),
        Line2D([], [], linestyle="", marker="$=$", color=tr.INK, label="equivalent (TOST, +-20 %)"),
        Line2D(
            [],
            [],
            linestyle="",
            marker="$c$",
            color=tr.INK,
            label="consistent (spread not above the reference)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.1),
        fontsize=6.5,
        handlelength=1.4,
    )
    fig.subplots_adjust(wspace=0.22, hspace=0.35, top=0.9, right=0.95)
    tr._header(
        fig,
        "Stage A: 10 of 10; Stage B: equivalent to the reference and at most as scattered (Stage-A successes)",
    )
    return tr.save(fig, out, "overview_map")


def _ordered_keys(rows: list[dict[str, Any]], template: dict[str, Any]) -> list[str]:
    return sorted(
        {key for row in rows for key in row},
        key=lambda key: list(template).index(key) if key in template else 999,
    )


def main() -> None:
    config, runs = tr.load_runs()
    out = config.paths.results_reports_dir / "thesis_results" / "extra" / "overview"
    fig_dir, tab_dir, tex_dir = out / "fig", out / "tables", out / "latex"
    for directory in (fig_dir, tab_dir, tex_dir):
        directory.mkdir(parents=True, exist_ok=True)
    tr._style()
    a_cells = tr.stage_a_cells(runs)
    b_cells = tr.stage_b_cells(runs)
    reference = tr.reference_stats(runs)

    trend = level_trend(b_cells)
    trend_keys = _ordered_keys(trend, trend[0])
    tr.write_csv(
        tab_dir / "stage_b_level_trend.csv",
        trend_keys,
        [[row.get(key) for key in trend_keys] for row in trend],
    )
    band = noise_band(reference)
    crit = criteria_rows(a_cells, b_cells, reference, load_tost())
    crit_keys = _ordered_keys(crit, crit[-1])
    tr.write_csv(
        tab_dir / "overview_criteria.csv",
        crit_keys,
        [[row.get(key) for key in crit_keys] for row in crit],
    )

    figs = [
        (
            fig_level_trend(trend, fig_dir),
            "fig:stage-b-level-trend",
            "Stage B over the guidance levels, Stage-A successes only: change of the throughput (top) and of the latency p50 (bottom) "
            "per guidance level from L2 to L5, per main-matrix arm, estimated as the Theil-Sen slope of the logarithm of the per-run values "
            f"with a bootstrap 95\\,\\% interval ({N_BOOT} resamples within levels). Filled markers: the interval excludes 0. For "
            "throughput a positive value means faster, for latency a positive value means slower. Descriptive.",
        ),
        (
            fig_level_profile(trend, band, fig_dir),
            "fig:stage-b-level-profile",
            "Stage B over the guidance levels, Stage-A successes only: median throughput of every level relative to the median of the "
            "same arm at L2 (log scale). The shaded band is the noise band of a median of ten runs against a median of ten runs of the "
            "same benchmark; changes inside it cannot be told apart from measurement noise.",
        ),
        (
            fig_overview_3d(crit, "Kestrel", fig_dir),
            "fig:overview-3d-kestrel",
            "Kestrel dialect, all factors in one view: guidance level, model and prompting strategy (zero-shot lighter). Left: Stage A "
            "successes out of 10 per cell; a dot marks the level from which every later level reaches 10/10. Right: deviation of the "
            "median write throughput of the Stage-A successes from the same-day reference (planes at 0 and at $\\pm$20\\,\\%); a dot marks "
            "the level from which every later level is equivalent (TOST, $\\pm$20\\,\\%) and at most as scattered as the reference. "
            "Crosses: no Stage-B value (L1) or no same-day reference (flash zero-shot).",
        ),
        (
            fig_overview_3d(crit, "Tessera", fig_dir),
            "fig:overview-3d-tessera",
            "Tessera dialect, all factors in one view, as \\cref{fig:overview-3d-kestrel}.",
        ),
        (
            fig_overview_map(crit, fig_dir),
            "fig:overview-map",
            "All factors in a readable two-dimensional form. Left: Stage A successes per cell; the thick line marks the level from which "
            "every later level reaches 10 of 10. Middle and right: deviation of the median throughput of the Stage-A successes from the "
            "same-day reference (\\%) for write and read; ``='' = equivalent to the reference (TOST, $\\pm$20\\,\\%), ``c'' = spread "
            "(interquartile range relative to the median) not larger than that of the reference; the thick line marks the level from "
            "which every later level meets both. ``none'': no such level; n/a: no same-day reference.",
        ),
    ]
    (tex_dir / "figures.tex").write_text(tr.tex_figures(figs), encoding="utf-8")

    lines = [
        "# Overview and Stage-B level trend",
        "",
        "Regenerate: `uv run python scripts/thesis_overview.py`.",
        "",
    ]
    lines += ["## Level trend (Theil-Sen, % per level, bootstrap 95 %), Stage-A successes", ""]
    for op in ("write", "read"):
        for metric in ("throughput", "latency_p50"):
            selected = [
                row
                for row in trend
                if row["op"] == op
                and row["metric"] == metric
                and row.get("pct_per_level") is not None
            ]
            direction_counts = {
                direction: sum(1 for row in selected if row["direction"] == direction)
                for direction in ("faster with level", "slower with level", "no clear change")
            }
            lines.append(
                f"- {op} {metric}: "
                + ", ".join(f"{direction} {count}" for direction, count in direction_counts.items())
                + f" (of {len(selected)} arms)"
            )
            for row in selected:
                if row["direction"] != "no clear change":
                    lines.append(
                        f"  - {row['arm']}: {row['pct_per_level']:+.1f} % per level [{row['ci_lo']:+.1f}, {row['ci_hi']:+.1f}], L2 to L5 {row['pct_L2_to_L5']:+.0f} %"
                    )
    lines += ["", "## Points (lowest level from which every later level meets the criterion)", ""]
    for arm in tr.ARMS:
        point_a = point_of(crit, arm["key"], "stage_a_reliable")
        point_write = point_of(crit, arm["key"], "write_both") if arm["ref"] else "n/a"
        point_read = point_of(crit, arm["key"], "read_both") if arm["ref"] else "n/a"
        lines.append(
            f"- {arm['key']}: Stage A {point_a}; Stage B write {point_write}; read {point_read}"
        )
    n_eq = {op: sum(1 for row in crit if row.get(f"{op}_equivalent")) for op in ("write", "read")}
    n_co = {op: sum(1 for row in crit if row.get(f"{op}_consistent")) for op in ("write", "read")}
    n_both = {op: sum(1 for row in crit if row.get(f"{op}_both")) for op in ("write", "read")}
    n_cells = sum(1 for row in crit if row.get("write_delta_pct") is not None)
    lines += [
        "",
        f"Cells with a same-day reference: {n_cells}. Equivalent: write {n_eq['write']}, read {n_eq['read']}; consistent: write {n_co['write']}, read {n_co['read']}; both: write {n_both['write']}, read {n_both['read']}.",
        f"Noise band used in the profile: {band[0] * 100:+.0f} to {band[1] * 100:+.0f} %.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
