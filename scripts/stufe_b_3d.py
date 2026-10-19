from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D

from benchmark_agent.config import load_app_config

EPOCHS: tuple[int, ...] = (6, 7, 8, 9, 10, 11, 12, 13)
LEVELS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5")
PLOTTED: tuple[str, ...] = ("L2", "L3", "L4", "L5")
DIALECT: dict[int, str] = {
    6: "Kestrel",
    7: "Tessera",
    8: "Kestrel",
    9: "Tessera",
    10: "Kestrel",
    11: "Tessera",
    12: "Tessera",
    13: "Kestrel",
}
EXEC_FEEDBACK: dict[int, bool] = {epoch: epoch >= 10 for epoch in EPOCHS}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
KESTREL = "#2a78d6"
TESSERA = "#eb6834"
DPI = 150

Y_TOP = 100.0


def _read_tables(
    tables_dir: Path,
) -> tuple[dict[tuple[int, str], float], dict[tuple[int, str], float | None]]:
    per_level = tables_dir / "auswertung_je_level.csv"
    final = tables_dir / "final_level_vs_performance.csv"
    for path in (per_level, final):
        if not path.exists():
            raise SystemExit(f"{path} missing -- run scripts/performance_report.py first.")

    n_runs: dict[tuple[int, str], int] = defaultdict(int)
    n_stage_b: dict[tuple[int, str], int] = defaultdict(int)
    with per_level.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["epoch"]), row["guidance_level"])
            n_runs[key] += int(row["n_runs"])
            n_stage_b[key] += int(row["n_stage_b"])

    throughput: dict[tuple[int, str], float | None] = {}
    with final.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row["write_median_ops_s"]
            throughput[(int(row["epoch"]), row["level"])] = float(value) if value else None

    yields = {key: 100.0 * n_stage_b[key] / n_runs[key] for key in n_runs if n_runs[key]}
    return yields, throughput


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": INK_2,
            "text.color": INK,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
        }
    )


LABEL_NUDGE_Z: dict[int, float] = {6: 6.0, 8: -12.0, 9: -25.0, 10: 11.0, 11: -9.0, 13: -15.0}


def _draw_ceiling(ax: Axes3D) -> None:
    ceiling = [[(1.55, 0.0, 100.0), (5.5, 0.0, 100.0), (5.5, Y_TOP, 100.0), (1.55, Y_TOP, 100.0)]]
    ax.add_collection3d(Poly3DCollection(ceiling, facecolor=MUTED, alpha=0.09, edgecolor="none"))
    for y_edge in (0.0, Y_TOP):
        ax.plot([1.55, 5.5], [y_edge, y_edge], [100, 100], color=MUTED, lw=1.0, ls=(0, (5, 4)))
    ax.text(
        4.15,
        Y_TOP,
        106.5,
        "Decke: 100 % Ausbeute, kein Run der Zelle geht verloren",
        color=INK_2,
        fontsize=10.5,
        ha="center",
    )


def _draw_epoch(
    ax: Axes3D,
    epoch: int,
    x_by_level: dict[str, int],
    yields: dict[tuple[int, str], float],
    throughput: dict[tuple[int, str], float | None],
) -> None:
    colour = KESTREL if DIALECT[epoch] == "Kestrel" else TESSERA
    marker = "^" if EXEC_FEEDBACK[epoch] else "o"
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for level in PLOTTED:
        value = throughput.get((epoch, level))
        if value is None:
            continue
        xs.append(float(x_by_level[level]))
        ys.append(value)
        zs.append(yields[(epoch, level)])
    for x, y, z in zip(xs, ys, zs, strict=True):
        ax.plot([x, x], [y, y], [0, z], color=GRID, lw=0.8, zorder=1)
    ax.plot(xs, ys, zs, color=colour, lw=1.1, alpha=0.6, zorder=3)
    ax.scatter(
        xs,
        ys,
        zs,
        color=colour,
        marker=marker,
        s=70,
        depthshade=False,
        edgecolor=SURFACE,
        linewidth=1.3,
        zorder=4,
    )
    nudge = LABEL_NUDGE_Z.get(epoch, 0.0)
    if abs(nudge) >= 3.0:
        ax.plot(
            [xs[0], xs[0] - 0.15],
            [ys[0], ys[0]],
            [zs[0], zs[0] + nudge],
            color=colour,
            lw=0.8,
            alpha=0.45,
            zorder=3,
        )
    ax.text(
        xs[0] - 0.19,
        ys[0],
        zs[0] + nudge,
        f"E{epoch}",
        color=colour,
        fontsize=10,
        fontweight="bold",
        ha="right",
        va="center",
        zorder=5,
    )


def main() -> None:
    config = load_app_config()
    out_dir = config.paths.results_reports_dir / "performance_gemini"
    yields, throughput = _read_tables(out_dir / "tables")
    _style()

    fig = plt.figure(figsize=(13.5, 7.9))
    ax = cast(Axes3D, fig.add_axes((0.015, 0.075, 0.90, 0.775), projection="3d", proj_type="ortho"))
    ax.view_init(elev=17, azim=-61)
    x_by_level = {level: idx + 1 for idx, level in enumerate(LEVELS)}

    _draw_ceiling(ax)

    for epoch in EPOCHS:
        _draw_epoch(ax, epoch, x_by_level, yields, throughput)

    ax.scatter(
        [1],
        [0],
        [0],
        facecolor="none",
        edgecolor=MUTED,
        marker="o",
        s=115,
        linewidth=1.5,
        depthshade=False,
        zorder=4,
    )
    ax.text(
        1.0,
        0.0,
        13.0,
        "L1: 0 von 100 Runs ausführbar,\nnichts zu messen",
        color=INK_2,
        fontsize=10,
        ha="center",
    )

    ax.set_xlabel("Guidance-Level", labelpad=12, fontsize=11.5)
    ax.set_ylabel("Median write-Throughput (ops/s)", labelpad=16, fontsize=11.5)
    ax.set_zlabel("Messausbeute", labelpad=14, fontsize=11.5)
    ax.set_xticks(list(x_by_level.values()))
    ax.set_xticklabels(LEVELS, fontsize=11)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_zticks([0, 25, 50, 75, 100])
    ax.set_zticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xlim(0.6, 5.7)
    ax.set_ylim(0, Y_TOP)
    ax.set_zlim(0, 108)
    ax.set_box_aspect((1.75, 1.0, 0.88))
    for pane_axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane_axis.pane.set_facecolor(SURFACE)
        pane_axis.pane.set_edgecolor(GRID)
        pane_axis.pane.set_alpha(1.0)
        pane_axis._axinfo["grid"].update({"color": GRID, "linewidth": 0.7})

    def legend_entry(colour: str, marker: str, size: float, label: str) -> Line2D:
        return Line2D([], [], color=colour, marker=marker, ls="none", ms=size, label=label)

    handles = [
        legend_entry(KESTREL, "s", 8, "Kestrel (Ressourcen-REST)"),
        legend_entry(TESSERA, "s", 8, "Tessera (JSON-RPC-artig)"),
        legend_entry(INK_2, "o", 7.5, "nur Syntax-Feedback (E6–E9)"),
        legend_entry(INK_2, "^", 8, "Ausführungs-Feedback (E10–E13)"),
        Line2D([], [], color=INK_2, lw=1.5, label="ein Linienzug je Epoche, L2 → L5"),
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.035, 0.835),
        frameon=False,
        fontsize=10,
        labelspacing=0.55,
        handletextpad=0.7,
    )

    fig.suptitle(
        "Für die Performance-Messung ist L2 nicht der einzige Punkt, der zählt",
        x=0.035,
        y=0.965,
        ha="left",
        fontsize=17.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.035,
        0.915,
        "Der Zugewinn von L3 und vom Feedback in L4/L5 liegt in der Höhe (Ausbeute), "
        "nicht in der Tiefe (Durchsatz).",
        ha="left",
        fontsize=11.5,
        color=INK_2,
    )
    fig.text(
        0.035,
        0.012,
        "Ausbeute = Stufe-B-Runs / Runs je Epoche × Level ("
        "ausgeführt und ≥ 1 fehlerfreier write-Request); Fehlläufe sind NA, nie 0.  ·  "
        "Durchsatz = Median der Stufe-B-Runs derselben Zelle; innerhalb\n"
        "einer Epoche spannen L2–L5 nur Faktor 1,14–1,90 auf, "
        "das liegt im dokumentierten Messrauschen von ±50 % bei identischen Artefakten. "
        "Epochen nie gepoolt; Durchsatz zwischen Epochen ist mit Zeitdrift\nbehaftet.  ·  "
        "Tabellen: tables/auswertung_je_level.csv, tables/final_level_vs_performance.csv",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
        linespacing=1.7,
    )

    out_path = out_dir / "auswertung_3d_ausbeute_vs_durchsatz.png"
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
