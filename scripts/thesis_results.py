from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy import stats as sps

from benchmark_agent.config import load_app_config
from benchmark_agent.dashboard.build import ceiling_summary
from benchmark_agent.evaluation.cost import cost_of_usage
from benchmark_agent.evaluation.runs import SOURCE_PROXY, RunRecord, collect_runs
from benchmark_agent.evaluation.statistics import (
    apply_holm,
    compute_statistics,
    fisher_control_vs_rest,
    kruskal_by_level,
    pairwise_fisher_vs_reference,
    permutation_success_by_level,
    spearman_level_vs_throughput,
)
from benchmark_agent.models import ModelPricing

matplotlib.use("Agg")


LEVELS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5")
OPS: tuple[str, ...] = ("write", "read", "delete")
MAIN_MODELS: tuple[str, ...] = ("gemini", "claude_haiku", "glm")
MODELS: tuple[str, ...] = MAIN_MODELS
PIPELINE_MODEL = "gemini_lite"
ROLE_MAIN = "main matrix"
ROLE_PIPELINE = "pipeline development"
MODEL_LABEL = {
    "gemini": "flash",
    "gemini_lite": "flash-lite",
    "claude_haiku": "haiku",
    "glm": "glm",
}
MODEL_FULL = {
    "gemini": "gemini-3.5-flash",
    "gemini_lite": "gemini-3.5-flash-lite",
    "claude_haiku": "claude-haiku-5-5",
    "glm": "glm-5.3",
}
PROMPT_LABEL = {"zero_shot": "zero-shot", "multi_shot": "multi-shot"}
DIALECT_OF_EPOCH = {
    6: "Kestrel",
    7: "Tessera",
    8: "Kestrel",
    9: "Tessera",
    10: "Kestrel",
    11: "Tessera",
    12: "Tessera",
    13: "Kestrel",
    14: "Kestrel",
    15: "Tessera",
    16: "Kestrel",
    17: "Tessera",
}
REFERENCE_EPOCHS: tuple[tuple[str, int], ...] = (
    ("Tessera", 12),
    ("Kestrel", 13),
    ("Kestrel", 16),
    ("Tessera", 17),
)
MAIN_REFERENCE = {"Kestrel": "Kestrel E13", "Tessera": "Tessera E12"}


def _arm(key: str, model: str, prompting: str, dialect: str, epoch: int) -> dict[str, Any]:
    return {"key": key, "model": model, "prompting": prompting, "dialect": dialect, "epoch": epoch}


ARMS: list[dict[str, Any]] = [
    _arm("flash-zs-K", "gemini", "zero_shot", "Kestrel", 8),
    _arm("flash-zs-T", "gemini", "zero_shot", "Tessera", 9),
    _arm("flash-ms-K", "gemini", "multi_shot", "Kestrel", 13),
    _arm("flash-ms-T", "gemini", "multi_shot", "Tessera", 12),
    _arm("haiku-zs-K", "claude_haiku", "zero_shot", "Kestrel", 14),
    _arm("haiku-zs-T", "claude_haiku", "zero_shot", "Tessera", 15),
    _arm("haiku-ms-K", "claude_haiku", "multi_shot", "Kestrel", 14),
    _arm("haiku-ms-T", "claude_haiku", "multi_shot", "Tessera", 15),
    _arm("glm-zs-K", "glm", "zero_shot", "Kestrel", 16),
    _arm("glm-zs-T", "glm", "zero_shot", "Tessera", 17),
    _arm("glm-ms-K", "glm", "multi_shot", "Kestrel", 16),
    _arm("glm-ms-T", "glm", "multi_shot", "Tessera", 17),
]
PIPELINE_ARMS: list[dict[str, Any]] = [
    _arm("lite-zs-K", "gemini_lite", "zero_shot", "Kestrel", 10),
    _arm("lite-zs-T", "gemini_lite", "zero_shot", "Tessera", 11),
    _arm("lite-ms-K", "gemini_lite", "multi_shot", "Kestrel", 13),
    _arm("lite-ms-T", "gemini_lite", "multi_shot", "Tessera", 12),
]
SYNTAX_ONLY_ARMS: list[dict[str, Any]] = [
    _arm("lite-zs-K-syntax", "gemini_lite", "zero_shot", "Kestrel", 6),
    _arm("lite-zs-T-syntax", "gemini_lite", "zero_shot", "Tessera", 7),
]
SAME_DAY_REFERENCE = {
    12: "Tessera E12",
    13: "Kestrel E13",
    14: "Kestrel E16",
    15: "Tessera E17",
    16: "Kestrel E16",
    17: "Tessera E17",
}
for _role, _arms in (
    (ROLE_MAIN, ARMS),
    (ROLE_PIPELINE, PIPELINE_ARMS),
    (ROLE_PIPELINE, SYNTAX_ONLY_ARMS),
):
    for _arm_entry in _arms:
        _arm_entry["role"] = _role
        _arm_entry["ref"] = SAME_DAY_REFERENCE.get(_arm_entry["epoch"])
        _arm_entry["regime"] = (
            "execution feedback" if _arm_entry["epoch"] >= 10 else "syntax feedback only"
        )
        _arm_entry["label"] = (
            f"{MODEL_LABEL[_arm_entry['model']]} · {PROMPT_LABEL[_arm_entry['prompting']]} · "
            f"{_arm_entry['dialect']} (E{_arm_entry['epoch']})"
        )
        _arm_entry["short"] = (
            f"{MODEL_LABEL[_arm_entry['model']]} · {PROMPT_LABEL[_arm_entry['prompting']]}"
        )
ARM_BY_KEY = {arm["key"]: arm for arm in ARMS + PIPELINE_ARMS + SYNTAX_ONLY_ARMS}
RETEST_PAIRS: tuple[tuple[int, int, str], ...] = ((6, 10, "Kestrel"), (7, 11, "Tessera"))
DIALECTS: tuple[str, ...] = ("Kestrel", "Tessera")
VIEWS: tuple[str, ...] = ("all", "success")
VIEW_SUBSET = {"all": "measured", "success": "success"}


def view_label(n_runs: int, n_failed: int, view: str, marker: str = "×") -> str:
    if view in ("all", "measured"):
        failed = (
            f"{n_failed} of them failed Stage A ({marker})"
            if n_failed
            else "none of them failed Stage A"
        )
        return f"View: all measured runs, {n_runs} runs, {failed}"
    if view == "success":
        return f"View: Stage-A successes only, {n_runs} runs"
    raise ValueError(view)


OUTCOMES: tuple[tuple[str, str], ...] = (
    ("success", "Success (all four checks)"),
    ("generation_failed", "No artefact (JSON contract)"),
    ("syntax_error", "Syntax error (Gherkin or Python)"),
    ("api_rejected", "Request rejected by the facade"),
    ("runtime_error", "Step exception or failed scenario"),
    ("nonconformant", "Workload not conformant"),
)
OUTCOME_MAP = {
    "none": "success",
    "generation_failed": "generation_failed",
    "syntax_error": "syntax_error",
    "api_rejected": "api_rejected",
    "quoted_literal": "api_rejected",
    "step_error": "runtime_error",
    "scenario_failed": "runtime_error",
    "timeout": "runtime_error",
    "nonconformant": "nonconformant",
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1d9a6c"
VIOLET = "#8a5cd1"
MODEL_COLOR = {"gemini": BLUE, "gemini_lite": ORANGE, "claude_haiku": GREEN, "glm": VIOLET}
PROMPT_MARKER = {"zero_shot": "s", "multi_shot": "o"}
DIALECT_MARKER = {"Kestrel": "o", "Tessera": "^"}
RAMP_5 = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")
RAMP_2 = ("#86b6ef", "#1c5cab")
OUTCOME_COLOR = {
    "success": GRID,
    "generation_failed": RAMP_5[0],
    "syntax_error": RAMP_5[1],
    "api_rejected": RAMP_5[2],
    "runtime_error": RAMP_5[3],
    "nonconformant": RAMP_5[4],
}
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
TEXT_WIDTH_IN = 5.9
PDF_DPI = 200
JITTER_SEED = 20260715


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )


ALL_ARMS: list[dict[str, Any]] = ARMS + PIPELINE_ARMS + SYNTAX_ONLY_ARMS


def arm_runs(runs: Sequence[RunRecord], arm: dict[str, Any]) -> list[RunRecord]:
    return [
        run
        for run in runs
        if run.epoch == arm["epoch"]
        and run.model == arm["model"]
        and run.prompting == arm["prompting"]
        and not run.is_fixture
    ]


def by_level(runs: Sequence[RunRecord]) -> dict[str, list[RunRecord]]:
    grouped: dict[str, list[RunRecord]] = {lvl: [] for lvl in LEVELS}
    for run in runs:
        if run.guidance_level in grouped:
            grouped[run.guidance_level].append(run)
    return grouped


STAGE_B_SUBSETS = ("measured", "success", "stage_a_failed")


def _in_subset(run: RunRecord, subset: str) -> bool:
    if subset == "measured":
        return True
    if subset == "success":
        return run.is_fixture or run.success
    if subset == "stage_a_failed":
        return not run.is_fixture and not run.success
    raise ValueError(subset)


def sb(runs: Sequence[RunRecord], op: str, metric: str, subset: str = "measured") -> list[float]:
    values = [
        run.stage_b_value(SOURCE_PROXY, op, metric) for run in runs if _in_subset(run, subset)
    ]
    return [value for value in values if value is not None]


LIST_PRICE_OVERRIDES = {
    "claude_haiku": ModelPricing(
        input_usd_per_m=0.10, output_usd_per_m=0.50, cached_usd_per_m=0.01, verified_on="2026-10-09"
    )
}


def pricing_for(model: str | None, config: Any) -> ModelPricing | None:
    if model in LIST_PRICE_OVERRIDES:
        return LIST_PRICE_OVERRIDES[model]
    if model in config.models.models:
        return config.models.models[model].pricing
    return None


def quart(values: Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, 25)), float(np.median(arr)), float(np.percentile(arr, 75))


def fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}"


def fmt_p_plain(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt(x: float | None, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{nd}f}"


def holm(pvals: Sequence[float | None]) -> list[float | None]:
    present = [(i, p) for i, p in enumerate(pvals) if p is not None]
    m = len(present)
    adjusted: list[float | None] = [None] * len(pvals)
    running = 0.0
    for rank, (i, p) in enumerate(sorted(present, key=lambda pair: pair[1])):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)
        adjusted[i] = running
    return adjusted


def fisher(a_s: int, a_n: int, b_s: int, b_n: int) -> float:
    _, p = sps.fisher_exact([[a_s, a_n - a_s], [b_s, b_n - b_s]])
    return float(p)


def mwu(a: Sequence[float], b: Sequence[float]) -> tuple[float | None, float | None]:
    if len(a) < 2 or len(b) < 2:
        return None, None
    result = sps.mannwhitneyu(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def spearman_ci(rho: float, n: int) -> tuple[float, float]:
    if n < 4 or abs(rho) >= 1:
        return rho, rho
    z = math.atanh(rho)
    se = 1 / math.sqrt(n - 3)
    return math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)


def tex(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("·", r"$\cdot$")
        .replace("→", r"$\rightarrow$")
        .replace("Δ", r"$\Delta$")
        .replace("ρ", r"$\rho$")
        .replace("≥", r"$\geq$")
    )


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(["" if value is None else value for value in row])


def latex_table(
    caption: str,
    label: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    align: str,
    note: str = "",
    size: str = r"\small",
    midrules_after: Sequence[int] = (),
    tabcolsep: str | None = None,
) -> str:
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        f"    {size}",
        *([f"    \\setlength{{\\tabcolsep}}{{{tabcolsep}}}"] if tabcolsep else []),
        f"    \\begin{{tabular}}{{{align}}}",
        r"        \toprule",
        "        " + " & ".join(header) + r" \\",
        r"        \midrule",
    ]
    for i, row in enumerate(rows):
        lines.append("        " + " & ".join(row) + r" \\")
        if i in midrules_after:
            lines.append(r"        \midrule")
    lines += [r"        \bottomrule", r"    \end{tabular}"]
    if note:
        lines += [
            r"    \par\smallskip",
            r"    \begin{minipage}{\linewidth}\footnotesize",
            f"    {note}",
            r"    \end{minipage}",
        ]
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def save(fig: Figure, out_dir: Path, name: str) -> str:
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", pad_inches=0.05, dpi=PDF_DPI)
    plt.close(fig)
    return name


def load_runs() -> tuple[Any, list[RunRecord]]:
    config = load_app_config()
    collection = collect_runs(
        csv_path=config.paths.results_raw_dir / "metrics.csv",
        artifacts_dir=config.paths.results_artifacts_dir,
        project_root=config.paths.project_root,
    )
    runs = [run for run in collection.runs if not run.synthetic and run.status != "not_evaluated"]
    return config, runs


def stage_a_cells(runs: Sequence[RunRecord]) -> dict[tuple[str, str], dict[str, Any]]:
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for arm in ALL_ARMS:
        grouped = by_level(arm_runs(runs, arm))
        for lvl, members in grouped.items():
            outcomes = Counter(OUTCOME_MAP.get(run.failure_class, "other") for run in members)
            cells[(arm["key"], lvl)] = {
                "arm": arm["key"],
                "level": lvl,
                "n": len(members),
                "success": sum(1 for run in members if run.success),
                "syntax_ok": sum(1 for run in members if run.syntax_ok),
                "executed_ok": sum(1 for run in members if run.syntax_ok and run.executed_ok),
                "stage_b_write": len(sb(members, "write", "throughput_ops_s")),
                **{f"measured_{op}": len(sb(members, op, "throughput_ops_s")) for op in OPS},
                **{
                    f"measured_stage_a_failed_{op}": len(
                        sb(members, op, "throughput_ops_s", "stage_a_failed")
                    )
                    for op in OPS
                },
                **{
                    f"stage_b_missing_{op}": sum(1 for run in members if run.success)
                    - len(sb(members, op, "throughput_ops_s", "success"))
                    for op in OPS
                },
                "outcomes": outcomes,
                "first_draft_ok": sum(
                    1 for run in members if run.success and run.feedback_iterations == 0
                ),
                "repaired": sum(
                    1 for run in members if run.success and run.feedback_iterations > 0
                ),
                "unrepaired": sum(
                    1 for run in members if not run.success and run.feedback_iterations > 0
                ),
                "failed_no_feedback": sum(
                    1 for run in members if not run.success and run.feedback_iterations == 0
                ),
                "max_rounds": max((run.feedback_iterations for run in members), default=0),
                "exec_rounds_runs": sum(1 for run in members if run.execution_feedback_rounds > 0),
                "failure_details": Counter(
                    f"{run.failure_class}: {run.failure_detail[:48]}"
                    for run in members
                    if not run.success
                ),
            }
    return cells


def stage_a_tests(runs: Sequence[RunRecord]) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        members = arm_runs(runs, arm)
        control = fisher_control_vs_rest(members)
        permutation = permutation_success_by_level(members)
        pairwise = pairwise_fisher_vs_reference(members)
        apply_holm(pairwise)
        row = {
            "arm": arm["key"],
            "fisher_control_p": control.p_value,
            "fisher_control_note": control.note,
            "permutation_chi2": permutation.statistic,
            "permutation_p": permutation.p_value,
            "permutation_V": permutation.effect,
        }
        for test in pairwise:
            lvl = test.groups[1]
            row[f"fisher_L2_vs_{lvl}_p"] = test.p_value
            row[f"fisher_L2_vs_{lvl}_holm"] = test.p_adjusted
        rows.append(row)
    return rows


def stage_b_cells(runs: Sequence[RunRecord]) -> dict[tuple[str, str, str], dict[str, Any]]:
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for arm in ALL_ARMS:
        grouped = by_level(arm_runs(runs, arm))
        for lvl, members in grouped.items():
            for op in OPS:
                measured = [
                    run
                    for run in members
                    if run.stage_b_value(SOURCE_PROXY, op, "throughput_ops_s") is not None
                ]
                tp = sb(measured, op, "throughput_ops_s")
                p50 = sb(measured, op, "latency_p50_ms")
                p95 = sb(measured, op, "latency_p95_ms")
                p99 = sb(measured, op, "latency_p99_ms")
                ok = [run.success for run in measured]
                run_ids = [(run.epoch, run.run_id) for run in measured]
                tp_s = sb(measured, op, "throughput_ops_s", "success")
                p50_s = sb(measured, op, "latency_p50_ms", "success")
                cell: dict[str, Any] = {
                    "arm": arm["key"],
                    "level": lvl,
                    "op": op,
                    "n": len(tp),
                    "n_success": len(tp_s),
                    "n_stage_a_failed": len(tp) - len(tp_s),
                    "tp": tp,
                    "p50": p50,
                    "p95": p95,
                    "p99": p99,
                    "ok": ok,
                    "run_ids": run_ids,
                    "tp_s": tp_s,
                    "p50_s": p50_s,
                }
                if tp:
                    cell["tp_q1"], cell["tp_med"], cell["tp_q3"] = quart(tp)
                    cell["p50_q1"], cell["p50_med"], cell["p50_q3"] = quart(p50)
                    cell["p95_med"] = float(np.median(p95))
                    cell["p99_med"] = float(np.median(p99))
                if tp_s:
                    cell["tp_s_q1"], cell["tp_s_med"], cell["tp_s_q3"] = quart(tp_s)
                    cell["p50_s_q1"], cell["p50_s_med"], cell["p50_s_q3"] = quart(p50_s)
                cells[(arm["key"], lvl, op)] = cell
    return cells


def cell_view(
    cell: dict[str, Any], metric: str, view: str
) -> tuple[list[float], tuple[float, float, float] | None]:
    key = metric if view == "all" else f"{metric}_s"
    values = cell[key]
    if not values:
        return values, None
    return values, (cell[f"{key}_q1"], cell[f"{key}_med"], cell[f"{key}_q3"])


def view_counts(
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
    arms: Sequence[dict[str, Any]],
    ops: Sequence[str] = OPS,
    levels: Sequence[str] = LEVELS,
) -> tuple[int, int]:
    shown: set[tuple[Any, str]] = set()
    failed: set[tuple[Any, str]] = set()
    for arm in arms:
        for lvl in levels:
            for op in ops:
                cell = b_cells[(arm["key"], lvl, op)]
                for run_id, ok in zip(cell["run_ids"], cell["ok"], strict=True):
                    shown.add(run_id)
                    if not ok:
                        failed.add(run_id)
    return len(shown), len(failed)


def figure_view_label(
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
    arms: Sequence[dict[str, Any]],
    view: str,
    ops: Sequence[str] = OPS,
    marker: str = "×",
) -> str:
    n, n_failed = view_counts(b_cells, arms, ops)
    return view_label(n if view == "all" else n - n_failed, n_failed, view, marker)


def reference_stats(runs: Sequence[RunRecord]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for dialect, epoch in REFERENCE_EPOCHS:
        key = f"{dialect} E{epoch}"
        epoch_runs = [run for run in runs if run.epoch == epoch]
        summary = ceiling_summary(epoch_runs).get(SOURCE_PROXY, {})
        gold = [
            run for run in epoch_runs if run.is_fixture and run.condition_name.startswith("_gold_")
        ]
        out[key] = {}
        for op in OPS:
            tp = sb(gold, op, "throughput_ops_s")
            p50 = sb(gold, op, "latency_p50_ms")
            p95 = sb(gold, op, "latency_p95_ms")
            p99 = sb(gold, op, "latency_p99_ms")
            tp_q1, tp_med, tp_q3 = quart(tp)
            p50_q1, p50_med, p50_q3 = quart(p50)
            assert abs(tp_med - summary[op]["median_throughput_ops_s"]) < 1e-9
            out[key][op] = {
                "dialect": dialect,
                "epoch": epoch,
                "n": len(tp),
                "tp": tp,
                "tp_q1": tp_q1,
                "tp_med": tp_med,
                "tp_q3": tp_q3,
                "tp_min": min(tp),
                "tp_max": max(tp),
                "p50": p50,
                "p50_q1": p50_q1,
                "p50_med": p50_med,
                "p50_q3": p50_q3,
                "p95_med": float(np.median(p95)),
                "p99_med": float(np.median(p99)),
                "evaluated": sorted(run.evaluated_at or "" for run in gold),
            }
    return out


def delta_rows(
    sb_cells: dict[tuple[str, str, str], dict[str, Any]],
    reference: dict[str, dict[str, dict[str, Any]]],
    arms: Sequence[dict[str, Any]] = ARMS,
) -> list[dict[str, Any]]:
    def pct(x: float | None, ref: float) -> float | None:
        return None if x is None else (x - ref) / ref * 100

    rows = []
    for arm in arms:
        same_day = arm["ref"] is not None
        ref_key = arm["ref"] or MAIN_REFERENCE[arm["dialect"]]
        for lvl in LEVELS[1:]:
            for op in OPS:
                cell = sb_cells[(arm["key"], lvl, op)]
                ref = reference[ref_key][op]
                if cell["n"] == 0:
                    continue
                row: dict[str, Any] = {
                    "arm": arm["key"],
                    "model": MODEL_FULL[arm["model"]],
                    "level": lvl,
                    "op": op,
                    "n": cell["n"],
                    "n_success": cell["n_success"],
                    "n_stage_a_failed": cell["n_stage_a_failed"],
                    "same_day_as_reference": same_day,
                    "reference": ref_key,
                    "ref_tp_med": ref["tp_med"],
                    "ref_p50_med": ref["p50_med"],
                }
                for view, suffix in (("all", ""), ("success", "_success")):
                    tp_values, tp_quartiles = cell_view(cell, "tp", view)
                    _, p50_quartiles = cell_view(cell, "p50", view)
                    row[f"tp_med{suffix}"] = tp_quartiles[1] if tp_quartiles else None
                    row[f"p50_med{suffix}"] = p50_quartiles[1] if p50_quartiles else None
                    for i, quartile in enumerate(("q1", "med", "q3")):
                        row[f"delta_tp_{quartile}{suffix}"] = (
                            pct(tp_quartiles[i], ref["tp_med"]) if tp_quartiles else None
                        )
                        row[f"delta_p50_{quartile}{suffix}"] = (
                            pct(p50_quartiles[i], ref["p50_med"]) if p50_quartiles else None
                        )
                    row[f"mwu_vs_reference_p{suffix}"] = mwu(tp_values, ref["tp"])[1]
                rows.append(row)
    return rows


def stage_b_tests(
    runs: Sequence[RunRecord], arms: Sequence[dict[str, Any]] = ARMS
) -> list[dict[str, Any]]:
    rows = []
    for arm in arms:
        members = arm_runs(runs, arm)
        for op in OPS:
            row: dict[str, Any] = {"arm": arm["key"], "model": MODEL_FULL[arm["model"]], "op": op}
            successes = [run for run in members if run.success]
            for suffix, subset in (("", members), ("_success", successes)):
                kruskal = kruskal_by_level(subset, op, SOURCE_PROXY)
                spearman = spearman_level_vs_throughput(subset, op, SOURCE_PROXY)
                apply_holm([kruskal, spearman])
                ci_low: float | None = None
                ci_high: float | None = None
                if spearman.applicable:
                    ci_low, ci_high = spearman_ci(spearman.statistic or 0.0, spearman.n)
                row.update(
                    {
                        f"n{suffix}": spearman.n,
                        f"kw_H{suffix}": kruskal.statistic,
                        f"kw_p{suffix}": kruskal.p_value,
                        f"kw_holm{suffix}": kruskal.p_adjusted,
                        f"kw_eps2{suffix}": kruskal.effect,
                        f"kw_groups{suffix}": ",".join(kruskal.groups),
                        f"rho{suffix}": spearman.statistic,
                        f"rho{suffix}_ci_low": ci_low,
                        f"rho{suffix}_ci_high": ci_high,
                        f"rho{suffix}_p": spearman.p_value,
                        f"rho{suffix}_holm": spearman.p_adjusted,
                    }
                )
            rows.append(row)
    return rows


def add_global_holm(rows: list[dict[str, Any]]) -> None:
    for row, adjusted in zip(rows, holm([row["rho_p"] for row in rows]), strict=True):
        row["rho_holm_all"] = adjusted
    for row, adjusted in zip(rows, holm([row["rho_success_p"] for row in rows]), strict=True):
        row["rho_success_holm_all"] = adjusted


def epoch_battery(runs: Sequence[RunRecord]) -> list[dict[str, Any]]:
    rows = []
    epochs = sorted({arm["epoch"] for arm in ARMS})
    for epoch in epochs:
        members = [run for run in runs if run.epoch == epoch and run.model in MAIN_MODELS]
        for op in OPS:
            report = compute_statistics(members, operation=op, source=SOURCE_PROXY)
            tests = {test.name: test for test in report.tests}
            kruskal = tests["Kruskal-Wallis"]
            spearman = tests["Spearman"]
            mann_whitney = tests["Mann-Whitney U"]
            permutation = tests["Permutationstest L2–L5"]
            models = sorted({MODEL_FULL[run.model] for run in members if run.model in MODEL_FULL})
            rows.append(
                {
                    "epoch": epoch,
                    "models": "+".join(models),
                    "op": op,
                    "n_stage_b": spearman.n,
                    "kw_H": kruskal.statistic,
                    "kw_p": kruskal.p_value,
                    "kw_holm": kruskal.p_adjusted,
                    "rho": spearman.statistic,
                    "rho_p": spearman.p_value,
                    "rho_holm": spearman.p_adjusted,
                    "mwu_U": mann_whitney.statistic,
                    "mwu_p": mann_whitney.p_value,
                    "mwu_holm": mann_whitney.p_adjusted,
                    "mwu_groups": "/".join(str(size) for size in mann_whitney.group_sizes),
                    "permutation_p": permutation.p_value,
                    "permutation_note": permutation.note,
                }
            )
    return rows


def _arm_of(model: str, prompting: str, dialect: str) -> dict[str, Any]:
    return next(
        arm
        for arm in ARMS
        if arm["model"] == model and arm["prompting"] == prompting and arm["dialect"] == dialect
    )


def prompting_comparison(
    runs: Sequence[RunRecord], a_cells: dict[tuple[str, str], dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage_a: list[dict[str, Any]] = []
    for model in MAIN_MODELS:
        for dialect in DIALECTS:
            zs_arm = _arm_of(model, "zero_shot", dialect)
            ms_arm = _arm_of(model, "multi_shot", dialect)
            p_values = []
            for lvl in LEVELS:
                zs_cell, ms_cell = a_cells[(zs_arm["key"], lvl)], a_cells[(ms_arm["key"], lvl)]
                p_values.append(
                    fisher(zs_cell["success"], zs_cell["n"], ms_cell["success"], ms_cell["n"])
                )
            adjusted = holm(p_values)
            for lvl, p_value, p_holm in zip(LEVELS, p_values, adjusted, strict=True):
                zs_cell, ms_cell = a_cells[(zs_arm["key"], lvl)], a_cells[(ms_arm["key"], lvl)]
                stage_a.append(
                    {
                        "model": model,
                        "dialect": dialect,
                        "level": lvl,
                        "zs_epoch": zs_arm["epoch"],
                        "ms_epoch": ms_arm["epoch"],
                        "zs_success": zs_cell["success"],
                        "zs_n": zs_cell["n"],
                        "ms_success": ms_cell["success"],
                        "ms_n": ms_cell["n"],
                        "fisher_p": p_value,
                        "fisher_holm": p_holm,
                    }
                )
    stage_b: list[dict[str, Any]] = []
    for epoch, model in (
        (8, "gemini"),
        (9, "gemini"),
        (14, "claude_haiku"),
        (15, "claude_haiku"),
        (16, "glm"),
        (17, "glm"),
    ):
        members = [
            run
            for run in runs
            if run.epoch == epoch
            and run.model == model
            and run.guidance_level == "L3"
            and not run.is_fixture
        ]
        zs_runs = [run for run in members if run.prompting == "zero_shot"]
        ms_runs = [run for run in members if run.prompting == "multi_shot"]
        for op in OPS:
            row: dict[str, Any] = {
                "epoch": epoch,
                "dialect": DIALECT_OF_EPOCH[epoch],
                "model": model,
                "op": op,
            }
            for suffix, subset in (("", "measured"), ("_success", "success")):
                zs_values = sb(zs_runs, op, "throughput_ops_s", subset)
                ms_values = sb(ms_runs, op, "throughput_ops_s", subset)
                u_statistic, mwu_p = mwu(zs_values, ms_values)
                row.update(
                    {
                        f"zs_n{suffix}": len(zs_values),
                        f"zs_median{suffix}": float(np.median(zs_values)) if zs_values else None,
                        f"ms_n{suffix}": len(ms_values),
                        f"ms_median{suffix}": float(np.median(ms_values)) if ms_values else None,
                        f"mwu_U{suffix}": u_statistic,
                        f"mwu_p{suffix}": mwu_p,
                    }
                )
            stage_b.append(row)
    return stage_a, stage_b


def retest_pairs(runs: Sequence[RunRecord], project_root: Path) -> list[dict[str, Any]]:
    rows = []
    for first, second, dialect in RETEST_PAIRS:
        first_runs = {
            (run.condition_name, run.seed): run
            for run in runs
            if run.epoch == first and not run.is_fixture
        }
        second_runs = {
            (run.condition_name, run.seed): run
            for run in runs
            if run.epoch == second and not run.is_fixture
        }
        for key, first_run in first_runs.items():
            second_run = second_runs.get(key)
            if (
                second_run is None
                or first_run.run_directory is None
                or second_run.run_directory is None
            ):
                continue
            first_dir = project_root / first_run.run_directory
            second_dir = project_root / second_run.run_directory
            try:
                identical = all(
                    (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
                    for name in ("benchmark.feature", "steps.py")
                )
            except OSError:
                identical = False
            tp_first = first_run.stage_b_value(SOURCE_PROXY, "write", "throughput_ops_s")
            tp_second = second_run.stage_b_value(SOURCE_PROXY, "write", "throughput_ops_s")
            p50_first = first_run.stage_b_value(SOURCE_PROXY, "write", "latency_p50_ms")
            p50_second = second_run.stage_b_value(SOURCE_PROXY, "write", "latency_p50_ms")
            rows.append(
                {
                    "pair": f"E{first}->E{second}",
                    "dialect": dialect,
                    "condition": first_run.condition_name,
                    "level": first_run.guidance_level,
                    "seed": first_run.seed,
                    "identical_artifact": identical,
                    "run_first": first_run.run_id,
                    "run_second": second_run.run_id,
                    "success_first": first_run.success,
                    "success_second": second_run.success,
                    "tp_first": tp_first,
                    "tp_second": tp_second,
                    "tp_ratio": (tp_second / tp_first) if (tp_first and tp_second) else None,
                    "p50_first": p50_first,
                    "p50_second": p50_second,
                    "p50_ratio": (p50_second / p50_first) if (p50_first and p50_second) else None,
                }
            )
    return rows


def epoch_role(epoch: int, model: str | None = None) -> str:
    if epoch <= 5:
        return "pilot"
    if model == PIPELINE_MODEL or epoch in (6, 7, 10, 11):
        if epoch in (6, 7):
            return "pipeline development (flash-lite), feedback-regime contrast"
        return "pipeline development (flash-lite)"
    if epoch in (12, 13) and model is None:
        return "main matrix (flash) + pipeline development (flash-lite)"
    return ROLE_MAIN


ROLE_SHORT = {
    "pilot": "pilot",
    "pipeline development (flash-lite), feedback-regime contrast": "pipe.$^c$",
    "pipeline development (flash-lite)": "pipe.",
    "main matrix (flash) + pipeline development (flash-lite)": "main + pipe.",
    ROLE_MAIN: "main",
}


def _epoch_members(runs: Sequence[RunRecord], epoch: int) -> list[RunRecord]:
    if epoch == 1:
        return [
            run
            for run in runs
            if run.epoch is None and not run.is_fixture and run.criteria_version == 2
        ]
    return [run for run in runs if run.epoch == epoch and not run.is_fixture]


def pilot_rows(runs: Sequence[RunRecord]) -> list[dict[str, Any]]:
    rows = []
    for epoch in range(1, 18):
        all_members = _epoch_members(runs, epoch)
        if not all_members:
            continue
        groups: list[tuple[str | None, list[RunRecord]]]
        if epoch in (12, 13):
            groups = [
                (model, [run for run in all_members if run.model == model])
                for model in ("gemini", PIPELINE_MODEL)
            ]
        else:
            groups = [(None, all_members)]
        for model, members in groups:
            grouped = by_level(members)
            api = sorted({run.sut_api or "influx" for run in members})
            rows.append(
                {
                    "epoch": epoch,
                    "model": MODEL_FULL.get(model, "") if model else "",
                    "role": epoch_role(epoch, model) if model else epoch_role(epoch),
                    "api": "/".join(api),
                    "runs": len(members),
                    **{
                        f"{lvl}_success": sum(1 for run in grouped[lvl] if run.success)
                        for lvl in LEVELS
                    },
                    **{f"{lvl}_n": len(grouped[lvl]) for lvl in LEVELS},
                    "canary_runs": sum(
                        1
                        for run in members
                        if any(rejection.startswith("influx_") for rejection in run.api_rejections)
                    ),
                    "legacy_requests": sum(run.api_legacy_requests or 0 for run in members),
                    "vocab_leak_total": sum(run.vocab_leak or 0 for run in members),
                    "vocab_leak_runs": sum(1 for run in members if (run.vocab_leak or 0) > 0),
                }
            )
    return rows


def token_rows(runs: Sequence[RunRecord], config: Any) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        pricing = pricing_for(arm["model"], config)
        grouped = by_level(arm_runs(runs, arm))
        for lvl, members in grouped.items():
            prompt_tokens: list[int] = []
            output_tokens: list[int] = []
            costs: list[float] = []
            llm_calls: list[int] = []
            probe_calls: list[int] = []
            for run in members:
                usage = run.usage_total or run.usage
                if not usage:
                    continue
                prompt_tokens.append(int(usage.get("prompt_token_count") or 0))
                if pricing is None:
                    output_tokens.append(int(usage.get("candidates_token_count") or 0))
                else:
                    run_cost, _billed, run_output_tokens, _cached = cost_of_usage(usage, pricing)
                    output_tokens.append(run_output_tokens)
                    costs.append(run_cost)
                llm_calls.append(run.llm_calls or 1)
                probe_calls.append(run.probe_calls or 0)
            n_success = sum(1 for run in members if run.success)
            rows.append(
                {
                    "arm": arm["key"],
                    "level": lvl,
                    "n": len(members),
                    "n_priced": len(costs),
                    "price_basis": (
                        "list price, not billed (subscription)"
                        if arm["model"] in LIST_PRICE_OVERRIDES
                        else "list price"
                    ),
                    "prompt_tokens_median": (
                        float(np.median(prompt_tokens)) if prompt_tokens else None
                    ),
                    "output_tokens_median": (
                        float(np.median(output_tokens)) if output_tokens else None
                    ),
                    "llm_calls_mean": float(np.mean(llm_calls)) if llm_calls else None,
                    "probe_calls_mean": float(np.mean(probe_calls)) if probe_calls else None,
                    "cost_usd_total": float(sum(costs)) if costs else None,
                    "cost_usd_per_run": float(np.mean(costs)) if costs else None,
                    "cost_usd_per_success": (
                        (float(sum(costs)) / n_success) if (costs and n_success) else None
                    ),
                    "success": n_success,
                }
            )
    return rows


def dataset_rows(runs: Sequence[RunRecord], config: Any) -> list[dict[str, Any]]:
    rows = []
    for epoch in range(1, 18):
        members = _epoch_members(runs, epoch)
        if not members:
            continue
        gold = [
            run
            for run in runs
            if run.epoch == epoch and run.is_fixture and run.condition_name.startswith("_gold_")
        ]
        models = sorted({MODEL_FULL.get(run.model or "", run.model or "") for run in members})
        prompt_set = {run.prompting for run in members}
        ms_levels = sorted(
            {run.guidance_level or "" for run in members if run.prompting == "multi_shot"}
        )
        if prompt_set == {"zero_shot", "multi_shot"} and ms_levels == ["L3"]:
            prompts = ["zero-shot (L3 also multi-shot)"]
        else:
            prompts = sorted(
                {PROMPT_LABEL.get(prompting or "", prompting or "") for prompting in prompt_set}
            )
        cost_total, complete = 0.0, True
        subscription = all(run.model in LIST_PRICE_OVERRIDES for run in members)
        for run in members:
            pricing = pricing_for(run.model, config)
            usage = run.usage_total or run.usage
            if pricing is None or not usage:
                complete = False
                continue
            if not run.usage_total:
                complete = False
            cost_total += cost_of_usage(usage, pricing)[0]
        dates = sorted((run.generated_at or "")[:10] for run in members if run.generated_at)
        cell_sizes = Counter((run.guidance_level, run.model, run.prompting) for run in members)
        levels_present = {cell[0] for cell in cell_sizes}
        rows.append(
            {
                "epoch": epoch,
                "role": epoch_role(epoch),
                "date": dates[0] if dates else "",
                "models": " + ".join(models),
                "dialect": "/".join(sorted({(run.sut_api or "influx") for run in members})),
                "prompting": " + ".join(prompts),
                "levels": "/".join(lvl for lvl in LEVELS if lvl in levels_present),
                "feedback_regime": (
                    "execution" if epoch >= 10 else ("syntax only" if epoch >= 2 else "syntax only")
                ),
                "workload": "v3" if epoch >= 6 else ("v2" if epoch >= 4 else "v1"),
                "runs": len(members),
                "n_per_cell": "/".join(str(size) for size in sorted(set(cell_sizes.values()))),
                "gold_runs": len(gold),
                "cost_usd": round(cost_total, 2),
                "billed": not subscription,
                "cost_complete": complete,
                "interleaved": epoch >= 8,
            }
        )
    return rows


PIPELINE_HEADER = "Pipeline development (gemini-3.5-flash-lite), not part of the model comparison"


class _CanvasWithRenderer(Protocol):
    def get_renderer(self) -> RendererBase: ...


def _header(fig: Figure, text: str) -> None:
    fig.canvas.draw()
    renderer = cast(_CanvasWithRenderer, fig.canvas).get_renderer()
    all_boxes = [ax.get_tightbbox(renderer) for ax in fig.axes if ax.get_visible()]
    boxes = [box for box in all_boxes if box is not None]
    to_figure = fig.transFigure.inverted()
    x0 = min(to_figure.transform((box.x0, box.y0))[0] for box in boxes) if boxes else 0.0
    y1 = max(to_figure.transform((box.x1, box.y1))[1] for box in boxes) if boxes else 1.0
    fig.text(
        x0, y1 + 0.015, text, ha="left", va="bottom", fontsize=7.5, color=INK, fontweight="bold"
    )


def _model_separators(
    ax: Axes, arms: Sequence[dict[str, Any]], offset: float = 0.0, key: str = "model"
) -> None:
    for i in range(1, len(arms)):
        if arms[i][key] != arms[i - 1][key]:
            ax.axhline(i + offset, color=BASELINE, lw=0.8, zorder=5)


def fig_stage_a_matrix(
    a_cells: dict[tuple[str, str], dict[str, Any]],
    out: Path,
    arms: Sequence[dict[str, Any]] = ARMS,
    name: str = "stage_a_success_matrix",
    header: str | None = None,
    group_key: str = "model",
) -> str:
    data = np.array(
        [
            [
                a_cells[(arm["key"], lvl)]["success"] / a_cells[(arm["key"], lvl)]["n"]
                for lvl in LEVELS
            ]
            for arm in arms
        ]
    )
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 0.6 + 0.29 * len(arms)))
    mesh = ax.pcolormesh(data, cmap=SEQ_CMAP, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
    ax.invert_yaxis()
    for i, arm in enumerate(arms):
        for j, lvl in enumerate(LEVELS):
            cell = a_cells[(arm["key"], lvl)]
            rate = cell["success"] / cell["n"]
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{cell['success']}/{cell['n']}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if rate > 0.6 else INK,
            )
    ax.set_xticks([j + 0.5 for j in range(len(LEVELS))], LEVELS)
    ax.set_yticks([i + 0.5 for i in range(len(arms))], [arm["label"] for arm in arms])
    _model_separators(ax, arms, key=group_key)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Guidance level")
    colorbar = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 0.5, 1])
    colorbar.set_label("Stage A success rate")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0)
    if header:
        _header(fig, header)
    return save(fig, out, name)


def fig_failure_classes(a_cells: dict[tuple[str, str], dict[str, Any]], out: Path) -> str:
    fig, axes = plt.subplots(
        1, len(LEVELS), figsize=(TEXT_WIDTH_IN, 1.3 + 0.27 * len(ARMS)), sharey=True
    )
    for ax, lvl in zip(axes, LEVELS, strict=True):
        for i, arm in enumerate(ARMS):
            left = 0
            counts = a_cells[(arm["key"], lvl)]["outcomes"]
            for key, _label in OUTCOMES:
                n = counts.get(key, 0)
                if n:
                    ax.barh(
                        i,
                        n,
                        left=left,
                        height=0.62,
                        color=OUTCOME_COLOR[key],
                        edgecolor=SURFACE,
                        linewidth=1.2,
                    )
                    left += n
        _model_separators(ax, ARMS, -0.5)
        ax.set_title(lvl)
        ax.set_xlim(0, 10)
        ax.set_xticks([0, 5, 10])
        ax.grid(False)
        if ax is axes[0]:
            ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.spines["bottom"].set_color(BASELINE)
    axes[0].set_yticks(range(len(ARMS)), [arm["label"] for arm in ARMS])
    axes[2].set_xlabel("Runs per cell (N = 10)")
    handles = [Patch(facecolor=OUTCOME_COLOR[key], label=label) for key, label in OUTCOMES]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=1.2,
        columnspacing=1.4,
    )
    fig.subplots_adjust(wspace=0.18, bottom=1.0 / (1.3 + 0.27 * len(ARMS)))
    return save(fig, out, "stage_a_failure_classes")


STATUS: tuple[tuple[str, str], ...] = (
    ("success", "Stage A success, measured"),
    ("measured_failed_a", "measured, but failed Stage A (kept, marked)"),
    ("partial", "ran with errors: partial traffic, no Stage-B value"),
    ("no_traffic", "no accepted request (no artefact, syntax error, all requests rejected)"),
)
STATUS_COLOR = {
    "success": GRID,
    "measured_failed_a": RAMP_5[1],
    "partial": RAMP_5[3],
    "no_traffic": RAMP_5[4],
}


def run_status(r: RunRecord) -> str:
    if r.success:
        return "success"
    if r.stage_b_value(SOURCE_PROXY, "write", "throughput_ops_s") is not None:
        return "measured_failed_a"
    for op in OPS:
        stats = r.op(SOURCE_PROXY, op)
        if stats is not None and stats.exercised:
            return "partial"
    return "no_traffic"


def status_rows(runs: Sequence[RunRecord]) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        for lvl, members in by_level(arm_runs(runs, arm)).items():
            counts = Counter(run_status(run) for run in members)
            partial_classes = Counter(
                run.failure_class for run in members if run_status(run) == "partial"
            )
            rows.append(
                {
                    "arm": arm["key"],
                    "epoch": arm["epoch"],
                    "level": lvl,
                    "n": len(members),
                    **{key: counts.get(key, 0) for key, _ in STATUS},
                    "partial_classes": "; ".join(
                        f"{failure_class} x{count}"
                        for failure_class, count in partial_classes.items()
                    ),
                }
            )
    return rows


def fig_status(rows: Sequence[dict[str, Any]], out: Path) -> str:
    fig, axes = plt.subplots(
        1, len(LEVELS), figsize=(TEXT_WIDTH_IN, 1.3 + 0.27 * len(ARMS)), sharey=True
    )
    for ax, lvl in zip(axes, LEVELS, strict=True):
        for i, arm in enumerate(ARMS):
            row = next(
                entry for entry in rows if entry["arm"] == arm["key"] and entry["level"] == lvl
            )
            left = 0
            for key, _label in STATUS:
                n = row[key]
                if n:
                    ax.barh(
                        i,
                        n,
                        left=left,
                        height=0.62,
                        color=STATUS_COLOR[key],
                        edgecolor=SURFACE,
                        linewidth=1.2,
                    )
                    left += n
        _model_separators(ax, ARMS, -0.5)
        ax.set_title(lvl)
        ax.set_xlim(0, 10)
        ax.set_xticks([0, 5, 10])
        ax.grid(False)
        ax.tick_params(axis="y", length=0)
        ax.spines["bottom"].set_color(BASELINE)
    axes[0].set_yticks(range(len(ARMS)), [arm["label"] for arm in ARMS])
    axes[0].invert_yaxis()
    axes[2].set_xlabel("Runs per cell (N = 10)")
    handles = [Patch(facecolor=STATUS_COLOR[key], label=label) for key, label in STATUS]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=1.2,
        columnspacing=1.4,
    )
    fig.subplots_adjust(wspace=0.18, bottom=1.0 / (1.3 + 0.27 * len(ARMS)))
    return save(fig, out, "stage_measurement_status")


def fig_feedback(
    a_cells: dict[tuple[str, str], dict[str, Any]],
    out: Path,
    rows: Sequence[dict[str, Any]] = ARMS,
    name: str = "stage_a_feedback_effect",
    header: str | None = None,
    row_labels: Sequence[str] | None = None,
    group_key: str = "model",
) -> str:
    height = 1.1 + 0.27 * len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, height), sharey=True)
    for ax, lvl in zip(axes, ("L4", "L5"), strict=True):
        for i, arm in enumerate(rows):
            cell = a_cells[(arm["key"], lvl)]
            first, final = cell["first_draft_ok"], cell["success"]
            ax.plot([first, final], [i, i], color=BASELINE, lw=1.4, zorder=1)
            ax.scatter(
                [first], [i], s=30, color=RAMP_2[0], edgecolor=SURFACE, linewidth=0.8, zorder=3
            )
            ax.scatter(
                [final], [i], s=30, color=RAMP_2[1], edgecolor=SURFACE, linewidth=0.8, zorder=4
            )
            ax.text(10.45, i, f"{first} → {final}", va="center", ha="left", fontsize=7, color=INK_2)
        _model_separators(ax, rows, -0.5, key=group_key)
        ax.set_title(lvl)
        ax.set_xlim(-0.4, 12.4)
        ax.set_xticks(range(0, 11, 2))
        ax.set_xlabel("Successful runs out of 10")
        ax.grid(False)
        ax.xaxis.grid(True)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(rows)), list(row_labels or [arm["label"] for arm in rows]))
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color=RAMP_2[0],
            linestyle="",
            markersize=6,
            label="first draft accepted",
        ),
        Line2D(
            [],
            [],
            marker="o",
            color=RAMP_2[1],
            linestyle="",
            markersize=6,
            label="final (after feedback)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.subplots_adjust(wspace=0.32, bottom=0.78 / height)
    if header:
        _header(fig, header)
    return save(fig, out, name)


def fig_stage_b(
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
    reference: dict[str, dict[str, dict[str, Any]]],
    dialect: str,
    op: str,
    view: str,
    out: Path,
) -> str:
    arms = [arm for arm in ARMS if arm["dialect"] == dialect]
    rng = np.random.default_rng(JITTER_SEED)
    fig, axes = plt.subplots(2, len(arms), figsize=(TEXT_WIDTH_IN, 4.1), sharey="row")
    any_failed = False
    for j, arm in enumerate(arms):
        top, bottom = axes[0][j], axes[1][j]
        ref = reference[arm["ref"]][op] if arm["ref"] else None
        color = MODEL_COLOR[arm["model"]]
        for k, lvl in enumerate(LEVELS[1:]):
            cell = b_cells[(arm["key"], lvl, op)]
            for ax, metric in ((top, "tp"), (bottom, "p50")):
                values, quartiles = cell_view(cell, metric, view)
                if not values or quartiles is None:
                    continue
                x = k + rng.uniform(-0.2, 0.2, size=len(values))
                y = np.asarray(values, dtype=float)
                if view == "all":
                    passed = np.asarray(cell["ok"], dtype=bool)
                else:
                    passed = np.ones(len(values), dtype=bool)
                ax.scatter(
                    x[passed],
                    y[passed],
                    s=10,
                    color=color,
                    edgecolor=SURFACE,
                    linewidth=0.5,
                    zorder=3,
                    alpha=0.95,
                )
                if (~passed).any():
                    any_failed = True
                    ax.scatter(
                        x[~passed],
                        y[~passed],
                        s=16,
                        marker="x",
                        color=color,
                        linewidth=0.9,
                        zorder=3,
                    )
                q1, med, q3 = quartiles
                ax.hlines(med, k - 0.3, k + 0.3, color=INK, lw=1.6, zorder=4)
                ax.vlines(k - 0.38, q1, q3, color=INK_2, lw=0.9, zorder=2)
        if ref is not None:
            for ax, key in ((top, "tp"), (bottom, "p50")):
                ax.axhspan(
                    ref[f"{key}_q1"], ref[f"{key}_q3"], color=GRID, alpha=0.55, zorder=0, lw=0
                )
                ax.axhline(ref[f"{key}_med"], color=INK_2, lw=0.9, ls=(0, (4, 2)), zorder=1)
        strategy = "zero-shot" if arm["prompting"] == "zero_shot" else "multi-shot"
        top.set_title(f"{MODEL_LABEL[arm['model']]}\n{strategy} · E{arm['epoch']}", fontsize=7)
        for ax in (top, bottom):
            ax.set_xticks(range(len(LEVELS) - 1), LEVELS[1:])
            ax.set_xlim(-0.6, len(LEVELS) - 1.4)
            ax.tick_params(axis="x", length=0, labelsize=6.5)
            ax.spines["bottom"].set_color(BASELINE)
        bottom.set_xlabel(
            "Level" if ref is not None else "Level\n(no same-day ref.)",
            fontsize=7,
            color=INK_2,
        )
    for row in axes:
        row[0].set_ylim(bottom=0)
    axes[0][0].set_ylabel(f"{op} throughput (ops/s)")
    axes[1][0].set_ylabel(f"{op} latency p50 (ms)")
    handles: list[Any] = [
        Line2D(
            [],
            [],
            marker="o",
            color=MODEL_COLOR[model],
            linestyle="",
            markersize=5,
            label=f"{MODEL_FULL[model]} run",
        )
        for model in MAIN_MODELS
    ]
    if any_failed:
        handles.append(
            Line2D(
                [],
                [],
                marker="x",
                color=INK_2,
                linestyle="",
                markersize=5,
                label="measured run that failed Stage A (kept)",
            )
        )
    handles += [
        Line2D(
            [],
            [],
            color=INK,
            lw=1.6,
            label="median, all measured runs" if view == "all" else "median, Stage-A successes",
        ),
        Line2D([], [], color=INK_2, lw=0.9, label="interquartile range"),
        Line2D(
            [], [], color=INK_2, lw=0.9, ls=(0, (4, 2)), label="same-day reference median (N = 10)"
        ),
        Patch(facecolor=GRID, alpha=0.55, label="reference interquartile range"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.15 if len(handles) > 6 else -0.11),
        handlelength=1.6,
    )
    fig.subplots_adjust(wspace=0.14, hspace=0.42, top=0.88)
    _header(fig, figure_view_label(b_cells, arms, view, ops=(op,)))
    return save(fig, out, f"stage_b_{op}_{dialect.lower()}_{view}")


def fig_delta(
    rows: Sequence[dict[str, Any]],
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
    dialect: str,
    view: str,
    out: Path,
) -> str:
    arms = [arm for arm in ARMS if arm["dialect"] == dialect and arm["ref"] is not None]
    suffix = "" if view == "all" else "_success"
    offsets = np.linspace(-0.36, 0.36, len(arms))
    fig, axes = plt.subplots(2, len(OPS), figsize=(TEXT_WIDTH_IN, 4.6), sharex="row", sharey=True)
    any_dagger = False
    for row_index, (metric, label) in enumerate((("tp", "throughput"), ("p50", "latency p50"))):
        for column_index, op in enumerate(OPS):
            ax = axes[row_index][column_index]
            for k, lvl in enumerate(LEVELS[1:]):
                if k:
                    ax.axhline(k - 0.5, color=GRID, lw=0.6, zorder=0)
                for arm, offset in zip(arms, offsets, strict=True):
                    row = next(
                        (
                            entry
                            for entry in rows
                            if entry["arm"] == arm["key"]
                            and entry["level"] == lvl
                            and entry["op"] == op
                        ),
                        None,
                    )
                    if row is None or row[f"delta_{metric}_med{suffix}"] is None:
                        continue
                    y = k + offset
                    color = MODEL_COLOR[arm["model"]]
                    med = row[f"delta_{metric}_med{suffix}"]
                    ax.plot(
                        [row[f"delta_{metric}_q1{suffix}"], row[f"delta_{metric}_q3{suffix}"]],
                        [y, y],
                        color=color,
                        lw=0.9,
                        alpha=0.55,
                        zorder=2,
                    )
                    ax.scatter(
                        [med],
                        [y],
                        marker=PROMPT_MARKER[arm["prompting"]],
                        s=16,
                        color=color,
                        edgecolor=SURFACE,
                        linewidth=0.5,
                        zorder=3,
                    )
                    if view == "all" and row["n_stage_a_failed"]:
                        any_dagger = True
                        ax.annotate(
                            "\u2020",
                            (med, y),
                            xytext=(3, 0.5),
                            textcoords="offset points",
                            fontsize=7,
                            color=INK,
                            va="center",
                            ha="left",
                            zorder=4,
                        )
            ax.axvline(0, color=INK_2, lw=0.9, ls=(0, (4, 2)), zorder=1)
            ax.set_yticks(range(len(LEVELS) - 1), LEVELS[1:])
            ax.grid(False)
            ax.xaxis.grid(True)
            ax.tick_params(axis="y", length=0)
            ax.spines["bottom"].set_color(BASELINE)
            if row_index == 0:
                ax.set_title(op)
            ax.set_xlabel(f"Δ {label} (%)")
    axes[0][0].invert_yaxis()
    handles: list[Any] = [
        Line2D(
            [],
            [],
            marker=PROMPT_MARKER[arm["prompting"]],
            color=MODEL_COLOR[arm["model"]],
            linestyle="",
            markersize=5,
            label=f"{arm['short']} (E{arm['epoch']}, ref. {arm['ref'].split()[1]})",
        )
        for arm in arms
    ]
    handles.append(
        Line2D([], [], color=INK_2, lw=0.9, ls=(0, (4, 2)), label="reference median (Δ = 0)")
    )
    if any_dagger:
        handles.append(
            Line2D(
                [],
                [],
                marker=r"$\dagger$",
                color=INK,
                linestyle="",
                markersize=6,
                label="cell contains measured runs that failed Stage A",
            )
        )
    fig.legend(
        handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), handlelength=1.6
    )
    fig.subplots_adjust(wspace=0.12, hspace=0.55, top=0.9)
    _header(fig, figure_view_label(b_cells, arms, view, ops=OPS, marker="cells marked †"))
    return save(fig, out, f"stage_b_delta_vs_reference_{dialect.lower()}_{view}")


def fig_comparison(
    panels: Sequence[tuple[str, dict[str, Any], dict[str, Any]]],
    a_cells: dict[tuple[str, str], dict[str, Any]],
    colors: tuple[str, str],
    legend: tuple[str, str],
    sig: dict[tuple[str, str], float | None],
    name: str,
    out: Path,
    ncols: int = 4,
) -> str:
    ncols = min(ncols, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, grid = plt.subplots(
        nrows, ncols, figsize=(TEXT_WIDTH_IN, 0.3 + 1.8 * nrows), sharey=True, squeeze=False
    )
    axes = list(grid.flat)
    x = np.arange(len(LEVELS))
    for ax, (title, arm_a, arm_b) in zip(axes, panels, strict=True):
        successes_a = [a_cells[(arm_a["key"], lvl)]["success"] for lvl in LEVELS]
        successes_b = [a_cells[(arm_b["key"], lvl)]["success"] for lvl in LEVELS]
        ax.plot(
            x - 0.08,
            successes_a,
            color=colors[0],
            lw=1.2,
            marker="o",
            markersize=4.5,
            markeredgecolor=SURFACE,
            zorder=3,
        )
        ax.plot(
            x + 0.08,
            successes_b,
            color=colors[1],
            lw=1.2,
            marker="o",
            markersize=4.5,
            markeredgecolor=SURFACE,
            zorder=3,
        )
        for k, lvl in enumerate(LEVELS):
            p = sig.get((title, lvl))
            if p is not None and successes_a[k] != successes_b[k]:
                ax.text(k, 10.7, f"{p:.2f}", ha="center", va="bottom", fontsize=6, color=INK_2)
        ax.set_title(title)
        ax.set_xticks(x, LEVELS)
        ax.set_ylim(-0.5, 11.8)
        ax.set_yticks([0, 5, 10])
        ax.tick_params(axis="x", length=0)
        ax.spines["bottom"].set_color(BASELINE)
    for row in grid:
        row[0].set_ylabel("Successful runs (of 10)")
    handles = [
        Line2D([], [], color=colors[0], marker="o", markersize=4.5, lw=1.2, label=legend[0]),
        Line2D([], [], color=colors[1], marker="o", markersize=4.5, lw=1.2, label=legend[1]),
        Line2D(
            [],
            [],
            linestyle="",
            marker="$0.10$",
            color=INK_2,
            markersize=12,
            label="Fisher exact p, Holm over the five levels (where the two differ)",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.12 / nrows),
        handlelength=1.6,
    )
    fig.subplots_adjust(wspace=0.15, hspace=0.45)
    return save(fig, out, name)


def fig_retest(rows: Sequence[dict[str, Any]], out: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.8))
    for ax, (first, second, dialect) in zip(axes, RETEST_PAIRS, strict=True):
        pairs = [
            row
            for row in rows
            if row["pair"] == f"E{first}->E{second}"
            and row["identical_artifact"]
            and row["tp_ratio"]
        ]
        xs = np.array([pair["tp_first"] for pair in pairs])
        ys = np.array([pair["tp_second"] for pair in pairs])
        lim = max(float(xs.max()), float(ys.max())) * 1.08 if len(pairs) else 100
        line_x = np.linspace(0, lim, 50)
        ax.fill_between(line_x, line_x * 0.5, line_x * 1.5, color=GRID, alpha=0.5, lw=0, zorder=0)
        ax.plot(line_x, line_x, color=INK_2, lw=0.9, zorder=1)
        ax.scatter(xs, ys, s=16, color=ORANGE, edgecolor=SURFACE, linewidth=0.6, zorder=3)
        ratios = np.array([pair["tp_ratio"] for pair in pairs])
        ax.set_title(f"{dialect} · flash-lite · E{first} → E{second}")
        ax.set_xlabel(f"write throughput, first measurement E{first} (ops/s)")
        ax.set_ylabel(f"second measurement E{second} (ops/s)")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.spines["bottom"].set_color(BASELINE)
        if len(pairs):
            q1, med, q3 = quart(ratios)
            ax.text(
                0.97,
                0.04,
                f"n = {len(pairs)} identical artefacts\nratio median {med:.2f}, IQR {q1:.2f}–{q3:.2f}\nrange {ratios.min():.2f}–{ratios.max():.2f}",
                transform=ax.transAxes,
                va="bottom",
                ha="right",
                fontsize=7,
                color=INK_2,
            )
    handles = [
        Line2D([], [], color=INK_2, lw=0.9, label="identity (ratio 1.0)"),
        Patch(facecolor=GRID, alpha=0.5, label="ratio 0.5–1.5"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(wspace=0.35)
    _header(
        fig,
        "Measurement noise only: flash-lite pipeline-development runs (not part of the model comparison)",
    )
    return save(fig, out, "stage_b_test_retest_noise")


def fig_spearman(
    rows: Sequence[dict[str, Any]],
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
    view: str,
    out: Path,
) -> str:
    key = "rho" if view == "all" else "rho_success"
    fig, axes = plt.subplots(
        1, len(OPS), figsize=(TEXT_WIDTH_IN, 0.7 + 0.25 * len(ARMS)), sharey=True
    )
    for ax, op in zip(axes, OPS, strict=True):
        for i, arm in enumerate(ARMS):
            row = next(entry for entry in rows if entry["arm"] == arm["key"] and entry["op"] == op)
            if row[key] is None:
                continue
            color = MODEL_COLOR[arm["model"]]
            ax.plot(
                [row[f"{key}_ci_low"], row[f"{key}_ci_high"]], [i, i], color=color, lw=1.0, zorder=2
            )
            filled = row[f"{key}_p"] is not None and row[f"{key}_p"] < 0.05
            ax.scatter(
                [row[key]],
                [i],
                marker=DIALECT_MARKER[arm["dialect"]],
                s=26,
                facecolor=color if filled else SURFACE,
                edgecolor=color,
                linewidth=1.0,
                zorder=3,
            )
        _model_separators(ax, ARMS, -0.5)
        ax.axvline(0, color=INK_2, lw=0.9, ls=(0, (4, 2)), zorder=1)
        ax.set_title(op)
        ax.set_xlim(-1, 1)
        ax.set_xticks([-0.5, 0, 0.5])
        if op == "read":
            ax.set_xlabel(
                "Spearman ρ between guidance level (L2 = 2 … L5 = 5) and throughput per run"
            )
        ax.grid(False)
        ax.xaxis.grid(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["bottom"].set_color(BASELINE)
    axes[0].set_yticks(range(len(ARMS)), [arm["label"] for arm in ARMS])
    axes[0].invert_yaxis()
    handles: list[Any] = [
        Line2D(
            [],
            [],
            marker="o",
            color=MODEL_COLOR[model],
            linestyle="",
            markersize=5,
            label=MODEL_FULL[model],
        )
        for model in MAIN_MODELS
    ]
    handles += [
        Line2D([], [], marker="o", color=INK_2, linestyle="", markersize=5, label="Kestrel"),
        Line2D([], [], marker="^", color=INK_2, linestyle="", markersize=5, label="Tessera"),
        Line2D(
            [],
            [],
            marker="o",
            color=INK_2,
            markerfacecolor=SURFACE,
            linestyle="",
            markersize=5,
            label="hollow: p ≥ 0.05 (uncorrected)",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), handlelength=1.4
    )
    fig.subplots_adjust(wspace=0.12)
    _header(fig, figure_view_label(b_cells, ARMS, view, ops=OPS, marker="included in ρ"))
    return save(fig, out, f"stage_b_spearman_by_arm_{view}")


ARTEFACT_CLASSES: tuple[tuple[str, str], ...] = (
    ("success", "Stage A success (enters Stage B)"),
    ("feature", "Feature file: Gherkin does not parse"),
    ("binding", "Feature/step binding: undefined step or quoted literal in a request"),
    ("step_syntax", "Step code: syntax or load error"),
    ("step_runtime", "Step code: runtime error or failed scenario"),
    ("api", "API contract: request rejected by the facade"),
    ("workload", "Workload not conformant (executed, wrong amount of data)"),
    ("no_artefact", "No artefact: reply broke the JSON contract"),
)
ARTEFACT_COLOR = {
    "success": GRID,
    "feature": "#4a3aa7",
    "binding": "#eda100",
    "step_syntax": "#e34948",
    "step_runtime": "#e34948",
    "api": "#008300",
    "workload": "#e87ba4",
    "no_artefact": MUTED,
}
ARTEFACT_HATCH = {"step_runtime": "////"}
OUTLIER_EDGE = "#d03b3b"


def artefact_class(r: RunRecord) -> str:
    if r.success:
        return "success"
    failure_class, detail = r.failure_class, (r.failure_detail or "")
    if failure_class == "generation_failed":
        return "no_artefact"
    if failure_class == "syntax_error":
        if "Feature could not be parsed" in detail or "Gherkin" in detail:
            return "feature"
        if "no matching step definition" in detail:
            return "binding"
        return "step_syntax"
    if failure_class == "quoted_literal":
        return "binding"
    if failure_class == "api_rejected":
        return "api"
    if failure_class == "nonconformant":
        return "workload"
    return "step_runtime"


def failure_artefact_rows(runs: Sequence[RunRecord]) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        for lvl, members in by_level(arm_runs(runs, arm)).items():
            counts = Counter(artefact_class(run) for run in members)
            rows.append(
                {
                    "arm": arm["key"],
                    "model": MODEL_FULL[arm["model"]],
                    "prompting": arm["prompting"],
                    "dialect": arm["dialect"],
                    "epoch": arm["epoch"],
                    "level": lvl,
                    "n": len(members),
                    **{key: counts.get(key, 0) for key, _ in ARTEFACT_CLASSES},
                }
            )
    return rows


def fig_failure_artefacts(rows: Sequence[dict[str, Any]], out: Path) -> str:
    height = 1.55 + 0.27 * len(ARMS)
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(TEXT_WIDTH_IN, height), sharey=True)
    for ax, lvl in zip(axes, LEVELS, strict=True):
        for i, arm in enumerate(ARMS):
            row = next(
                entry for entry in rows if entry["arm"] == arm["key"] and entry["level"] == lvl
            )
            left = 0
            for key, _label in ARTEFACT_CLASSES:
                n = row[key]
                if not n:
                    continue
                ax.barh(
                    i,
                    n,
                    left=left,
                    height=0.62,
                    color=ARTEFACT_COLOR[key],
                    hatch=ARTEFACT_HATCH.get(key),
                    edgecolor=SURFACE,
                    linewidth=1.2,
                )
                left += n
        _model_separators(ax, ARMS, -0.5)
        ax.set_title(lvl)
        ax.set_xlim(0, 10)
        ax.set_xticks([0, 5, 10])
        ax.grid(False)
        ax.tick_params(axis="y", length=0)
        ax.spines["bottom"].set_color(BASELINE)
    axes[0].set_yticks(range(len(ARMS)), [arm["label"] for arm in ARMS])
    axes[0].invert_yaxis()
    axes[2].set_xlabel("Runs per cell (N = 10)")
    plt.rcParams["hatch.color"] = SURFACE
    handles = [
        Patch(
            facecolor=ARTEFACT_COLOR[key],
            hatch=ARTEFACT_HATCH.get(key),
            edgecolor=SURFACE,
            label=label,
        )
        for key, label in ARTEFACT_CLASSES
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=1.6,
        columnspacing=1.4,
    )
    fig.subplots_adjust(wspace=0.18, bottom=1.25 / height)
    return save(fig, out, "stage_a_failure_artefacts")


def tukey_outliers(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) < 4:
        return np.zeros(len(arr), dtype=bool)
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    return (arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)


def _success_runs_label(
    b_cells: dict[tuple[str, str, str], dict[str, Any]], arms: Sequence[dict[str, Any]], op: str
) -> str:
    n = sum(b_cells[(arm["key"], lvl, op)]["n_success"] for arm in arms for lvl in LEVELS[1:])
    return f"Stage B, Stage-A successes only: {n} runs"


def fig_stage_b_runs(
    b_cells: dict[tuple[str, str, str], dict[str, Any]], dialect: str, op: str, out: Path
) -> tuple[str, int]:
    arms = [arm for arm in ARMS if arm["dialect"] == dialect]
    rng = np.random.default_rng(JITTER_SEED)
    fig, axes = plt.subplots(2, len(arms), figsize=(TEXT_WIDTH_IN, 4.0), sharey="row")
    n_outliers = 0
    for j, arm in enumerate(arms):
        color = MODEL_COLOR[arm["model"]]
        for row_index, metric in enumerate(("tp", "p50")):
            ax = axes[row_index][j]
            for k, lvl in enumerate(LEVELS[1:]):
                values = b_cells[(arm["key"], lvl, op)][f"{metric}_s"]
                if not values:
                    continue
                y = np.asarray(values, dtype=float)
                x = k + rng.uniform(-0.2, 0.2, size=len(y))
                is_outlier = tukey_outliers(y)
                n_outliers += int(is_outlier.sum())
                ax.scatter(
                    x[~is_outlier],
                    y[~is_outlier],
                    s=10,
                    color=color,
                    edgecolor=SURFACE,
                    linewidth=0.5,
                    zorder=3,
                )
                ax.scatter(
                    x[is_outlier],
                    y[is_outlier],
                    s=16,
                    color=color,
                    edgecolor=OUTLIER_EDGE,
                    linewidth=1.2,
                    zorder=4,
                )
                ax.hlines(float(np.median(y)), k - 0.3, k + 0.3, color=INK, lw=1.4, zorder=5)
            ax.set_xticks(range(len(LEVELS) - 1), LEVELS[1:])
            ax.set_xlim(-0.6, len(LEVELS) - 1.4)
            ax.tick_params(axis="x", length=0, labelsize=6.5)
            ax.spines["bottom"].set_color(BASELINE)
        strategy = "zero-shot" if arm["prompting"] == "zero_shot" else "multi-shot"
        axes[0][j].set_title(
            f"{MODEL_LABEL[arm['model']]}\n{strategy}, E{arm['epoch']}", fontsize=7
        )
        axes[1][j].set_xlabel("Level", fontsize=7, color=INK_2)
    for row in axes:
        row[0].set_ylim(bottom=0)
    axes[0][0].set_ylabel(f"{op} throughput (ops/s)")
    axes[1][0].set_ylabel(f"{op} latency p50 (ms)")
    handles: list[Any] = [
        Line2D(
            [],
            [],
            marker="o",
            color=MODEL_COLOR[model],
            linestyle="",
            markersize=5,
            label=f"{MODEL_FULL[model]} run",
        )
        for model in MAIN_MODELS
    ]
    handles += [
        Line2D(
            [],
            [],
            marker="o",
            color=MUTED,
            markeredgecolor=OUTLIER_EDGE,
            markeredgewidth=1.2,
            linestyle="",
            markersize=5,
            label="outlier: beyond 1.5 IQR of its cell",
        ),
        Line2D([], [], color=INK, lw=1.4, label="median per level"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), handlelength=1.6
    )
    fig.subplots_adjust(wspace=0.14, hspace=0.42, top=0.88)
    _header(fig, _success_runs_label(b_cells, arms, op))
    return save(fig, out, f"stage_b_{op}_{dialect.lower()}_runs"), n_outliers


def _candle(ax: Axes, x: float, values: Sequence[float], color: str, width: float = 0.5) -> None:
    arr = np.asarray(values, dtype=float)
    q1, med, q3 = np.percentile(arr, 25), float(np.median(arr)), np.percentile(arr, 75)
    iqr = q3 - q1
    inside = arr[(arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)]
    whisker_low, whisker_high = float(inside.min()), float(inside.max())
    ax.vlines(x, whisker_low, whisker_high, color=color, lw=0.9, zorder=2)
    ax.add_patch(
        Rectangle(
            (x - width / 2, q1),
            width,
            max(q3 - q1, 1e-9),
            facecolor=color,
            edgecolor=color,
            alpha=0.35,
            lw=0.9,
            zorder=3,
        )
    )
    ax.hlines(med, x - width / 2, x + width / 2, color=INK, lw=1.4, zorder=4)
    outliers = arr[(arr < q1 - 1.5 * iqr) | (arr > q3 + 1.5 * iqr)]
    if len(outliers):
        ax.scatter(
            [x] * len(outliers),
            outliers,
            s=12,
            facecolor=SURFACE,
            edgecolor=OUTLIER_EDGE,
            linewidth=1.0,
            zorder=5,
        )


def fig_stage_b_candles(
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
    reference: dict[str, dict[str, dict[str, Any]]],
    dialect: str,
    op: str,
    out: Path,
) -> str:
    arms = [arm for arm in ARMS if arm["dialect"] == dialect]
    fig, axes = plt.subplots(2, len(arms), figsize=(TEXT_WIDTH_IN, 3.9), sharey="row")
    for j, arm in enumerate(arms):
        color = MODEL_COLOR[arm["model"]]
        ref = reference[arm["ref"]][op] if arm["ref"] else None
        for row_index, metric in enumerate(("tp", "p50")):
            ax = axes[row_index][j]
            for k, lvl in enumerate(LEVELS[1:]):
                values = b_cells[(arm["key"], lvl, op)][f"{metric}_s"]
                if values:
                    _candle(ax, k, values, color)
            labels = list(LEVELS[1:])
            if ref is not None:
                _candle(ax, len(LEVELS) - 1 + 0.2, ref[metric], INK_2)
                ax.axhline(
                    float(np.median(ref[metric])), color=INK_2, lw=0.7, ls=(0, (3, 2)), zorder=1
                )
                labels.append("ref")
                ax.set_xticks([*range(len(LEVELS) - 1), len(LEVELS) - 1 + 0.2], labels)
            else:
                ax.set_xticks(range(len(LEVELS) - 1), labels)
            ax.set_xlim(-0.6, len(LEVELS) - 0.2)
            ax.tick_params(axis="x", length=0, labelsize=6.5)
            ax.spines["bottom"].set_color(BASELINE)
        strategy = "zero-shot" if arm["prompting"] == "zero_shot" else "multi-shot"
        axes[0][j].set_title(
            f"{MODEL_LABEL[arm['model']]}\n{strategy}, E{arm['epoch']}", fontsize=7
        )
        axes[1][j].set_xlabel(
            "Level" if ref is not None else "Level\n(no same-day ref.)", fontsize=7, color=INK_2
        )
    for row in axes:
        row[0].set_ylim(bottom=0)
    axes[0][0].set_ylabel(f"{op} throughput (ops/s)")
    axes[1][0].set_ylabel(f"{op} latency p50 (ms)")
    handles: list[Any] = [
        Patch(
            facecolor=MODEL_COLOR[model],
            alpha=0.35,
            edgecolor=MODEL_COLOR[model],
            label=MODEL_FULL[model],
        )
        for model in MAIN_MODELS
    ]
    handles += [
        Patch(facecolor=INK_2, alpha=0.35, edgecolor=INK_2, label="same-day reference (N = 10)"),
        Line2D([], [], color=INK, lw=1.4, label="median; box = interquartile range"),
        Line2D(
            [],
            [],
            marker="o",
            color=SURFACE,
            markeredgecolor=OUTLIER_EDGE,
            linestyle="",
            markersize=5,
            label="outlier outside the Tukey fences",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), handlelength=1.6
    )
    fig.subplots_adjust(wspace=0.14, hspace=0.45, top=0.88)
    _header(fig, _success_runs_label(b_cells, arms, op) + ", against the reference of the same day")
    return save(fig, out, f"stage_b_{op}_{dialect.lower()}_candles")


def flash_identical_pairs(runs: Sequence[RunRecord], project_root: Path) -> list[dict[str, Any]]:
    flash = [
        run
        for run in runs
        if run.model == "gemini" and not run.is_fixture and run.success and run.run_directory
    ]

    def key(r: RunRecord) -> bytes | None:
        directory = project_root / str(r.run_directory)
        try:
            return (
                (directory / "benchmark.feature").read_bytes()
                + b"\0"
                + (directory / "steps.py").read_bytes()
            )
        except OSError:
            return None

    by_key: dict[bytes, list[RunRecord]] = {}
    for run in flash:
        artefact = key(run)
        if artefact is not None:
            by_key.setdefault(artefact, []).append(run)
    rows = []
    for group in by_key.values():
        days = sorted({(run.evaluated_at or "")[:10] for run in group})
        if len(days) < 2:
            continue
        first = min(group, key=lambda run: run.evaluated_at or "")
        second = max(group, key=lambda run: run.evaluated_at or "")
        row: dict[str, Any] = {
            "run_first": first.run_id,
            "run_second": second.run_id,
            "epoch_first": first.epoch,
            "epoch_second": second.epoch,
            "level": first.guidance_level,
            "day_first": (first.evaluated_at or "")[:10],
            "day_second": (second.evaluated_at or "")[:10],
        }
        for op in ("write", "read"):
            tp_first = first.stage_b_value(SOURCE_PROXY, op, "throughput_ops_s")
            tp_second = second.stage_b_value(SOURCE_PROXY, op, "throughput_ops_s")
            row[f"{op}_first"], row[f"{op}_second"] = tp_first, tp_second
            row[f"{op}_ratio"] = (tp_second / tp_first) if (tp_first and tp_second) else None
        rows.append(row)
    return rows


def fig_repeatability(
    reference: dict[str, dict[str, dict[str, Any]]], pairs: Sequence[dict[str, Any]], out: Path
) -> str:
    fig, axes = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_IN, 2.9), gridspec_kw={"width_ratios": [1.5, 1]}
    )
    ax = axes[0]
    rng = np.random.default_rng(JITTER_SEED)
    order = ["Kestrel E13", "Kestrel E16", "Tessera E12", "Tessera E17"]
    day = {
        "Kestrel E13": "13 Sep",
        "Kestrel E16": "9 Oct",
        "Tessera E12": "13 Sep",
        "Tessera E17": "9 Oct",
    }
    for i, series in enumerate(order):
        for offset, op, marker in ((-0.16, "write", "o"), (0.16, "read", "^")):
            values = np.asarray(reference[series][op]["tp"], dtype=float)
            is_outlier = tukey_outliers(values)
            x = i + offset + rng.uniform(-0.05, 0.05, size=len(values))
            ax.scatter(
                x[~is_outlier],
                values[~is_outlier],
                s=13,
                marker=marker,
                color=INK_2,
                edgecolor=SURFACE,
                linewidth=0.5,
                zorder=3,
            )
            ax.scatter(
                x[is_outlier],
                values[is_outlier],
                s=18,
                marker=marker,
                color=INK_2,
                edgecolor=OUTLIER_EDGE,
                linewidth=1.2,
                zorder=4,
            )
            ax.hlines(
                float(np.median(values)),
                i + offset - 0.1,
                i + offset + 0.1,
                color=INK,
                lw=1.4,
                zorder=5,
            )
    ax.set_xticks(range(len(order)), [f"{series.split()[0]}\n{day[series]}" for series in order])
    ax.axvline(1.5, color=BASELINE, lw=0.8)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("throughput (ops/s)")
    ax.set_title("Reference benchmark, 10 runs per series")
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_color(BASELINE)
    ax2 = axes[1]
    measured_pairs = [pair for pair in pairs if pair["write_ratio"]]
    if measured_pairs:
        xs = np.array([pair["write_first"] for pair in measured_pairs])
        ys = np.array([pair["write_second"] for pair in measured_pairs])
        lim = max(float(xs.max()), float(ys.max())) * 1.1
        line_x = np.linspace(0, lim, 50)
        ax2.fill_between(line_x, line_x * 0.8, line_x * 1.25, color=GRID, alpha=0.6, lw=0, zorder=0)
        ax2.plot(line_x, line_x, color=INK_2, lw=0.9, zorder=1)
        ax2.scatter(
            xs, ys, s=18, color=MODEL_COLOR["gemini"], edgecolor=SURFACE, linewidth=0.6, zorder=3
        )
        ratios = ys / xs
        ax2.text(
            0.97,
            0.04,
            f"n = {len(measured_pairs)} pairs\nratio {np.median(ratios):.2f} (median)\n"
            f"range {ratios.min():.2f} to {ratios.max():.2f}",
            transform=ax2.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color=INK_2,
        )
        ax2.set_xlim(0, lim)
        ax2.set_ylim(0, lim)
        ax2.set_aspect("equal")
    ax2.set_title("Identical flash artefacts, two days")
    ax2.set_xlabel("write throughput, first day (ops/s)")
    ax2.set_ylabel("second day (ops/s)")
    ax2.spines["bottom"].set_color(BASELINE)
    handles = [
        Line2D([], [], marker="o", color=INK_2, linestyle="", markersize=5, label="write"),
        Line2D([], [], marker="^", color=INK_2, linestyle="", markersize=5, label="read"),
        Line2D([], [], color=INK, lw=1.4, label="median"),
        Line2D(
            [],
            [],
            marker="o",
            color=INK_2,
            markeredgecolor=OUTLIER_EDGE,
            markeredgewidth=1.2,
            linestyle="",
            markersize=5,
            label="outlier beyond 1.5 IQR of the series",
        ),
        Line2D([], [], color=INK_2, lw=0.9, label="identity"),
        Patch(facecolor=GRID, alpha=0.6, label="ratio 0.8 to 1.25"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.16), handlelength=1.4
    )
    fig.subplots_adjust(wspace=0.35)
    return save(fig, out, "stage_b_repeatability")


def arm_tex(arm: dict[str, Any]) -> str:
    return f"{MODEL_LABEL[arm['model']]} & {PROMPT_LABEL[arm['prompting']]} & {arm['dialect']} & E{arm['epoch']}"


def arm_short_tex(arm: dict[str, Any]) -> str:
    strategy = "zs" if arm["prompting"] == "zero_shot" else "ms"
    return f"{MODEL_LABEL[arm['model']]} $\\cdot$ {strategy} $\\cdot$ {arm['dialect'][0]} $\\cdot$ E{arm['epoch']}"


ARM_LEGEND = "zs = zero-shot, ms = multi-shot, K = Kestrel, T = Tessera, flash = gemini-3.5-flash, haiku = claude-haiku-5-5, glm = glm-5.3."
PIPELINE_LEGEND = "flash-lite = gemini-3.5-flash-lite, used only to develop and test the pipeline cheaply; never part of the model comparison."
MODEL_MIDRULES: tuple[int, ...] = (3, 7)


def tex_dataset(rows: Sequence[dict[str, Any]]) -> str:
    table = []
    for row in rows:
        models = (
            row["models"]
            .replace("gemini-3.5-flash-lite", "flash-lite")
            .replace("gemini-3.5-flash", "flash")
            .replace("claude-haiku-5-5", "haiku")
            .replace("glm-5.3", "glm")
        )
        prompting = (
            row["prompting"]
            .replace("zero-shot (L3 also multi-shot)", "zs (L3 also ms)")
            .replace("zero-shot", "zs")
            .replace("multi-shot", "ms")
        )
        dialect = {"influx": "InfluxDB", "tessera": "Tessera", "kestrel": "Kestrel"}.get(
            row["dialect"], row["dialect"]
        )
        table.append(
            [
                f"E{row['epoch']}",
                ROLE_SHORT[row["role"]],
                row["date"],
                tex(models),
                dialect,
                tex(prompting),
                row["workload"],
                row["feedback_regime"].replace("syntax only", "syntax"),
                str(row["runs"]),
                row["n_per_cell"],
                (
                    f"{row['cost_usd']:.2f}"
                    + ("" if row["cost_complete"] else "$^*$")
                    + ("" if row["billed"] else "$^\\dagger$")
                ),
            ]
        )
    return latex_table(
        "Dataset: one row per epoch (same prompt, workload, criteria and feedback regime; epochs are never pooled). "
        "E1--E5 are pilots with one request per operation; E6--E17 use workload v3 (10 write batches, 20 reads, 1 delete). "
        "The main matrix (three models, one Gemini model) consists of E8, E9 and E12--E17; gemini-3.5-flash-lite (E6, E7, E10, E11 and the "
        "flash-lite half of E12/E13) only served to develop and test the pipeline.",
        "tab:dataset",
        [
            "Epoch",
            "Role",
            "Date",
            "Model(s)",
            "API",
            "Prompting",
            "Workload",
            "Feedb.",
            "Runs",
            "N/cell",
            "USD",
        ],
        table,
        "lllllllllrr",
        note="Role: main = main matrix; pipe. = pipeline development with gemini-3.5-flash-lite (cheap model to test the pipeline, not part of the "
        "model comparison); pipe.$^c$ = pipeline development and contrast for the feedback regime (syntax-only feedback); main + pipe. = the epoch "
        "interleaves the gemini-3.5-flash runs of the main matrix with flash-lite pipeline runs, which are always analysed apart. "
        "Feedb. = feedback regime of the L4/L5 loop. "
        "zs = zero-shot, ms = multi-shot, flash = gemini-3.5-flash, flash-lite = gemini-3.5-flash-lite, haiku = claude-haiku-5-5, "
        "glm = glm-5.3; InfluxDB = its API without facade. $^*$ list-price estimate from the final model turn only (token usage of "
        "earlier turns was not recorded before E6); $^\\dagger$ list-price equivalent, not billed (Claude Code CLI on a subscription). E12, E13, E16 and "
        "E17 additionally contain the reference benchmark, 10 runs each (no model call).",
        size=r"\scriptsize",
        tabcolsep="2.5pt",
    )


def tex_stage_a(
    a_cells: dict[tuple[str, str], dict[str, Any]], tests: Sequence[dict[str, Any]]
) -> str:
    rows = []
    for arm in ARMS:
        arm_tests = next(entry for entry in tests if entry["arm"] == arm["key"])
        cells = [
            f"{a_cells[(arm['key'], lvl)]['success']}/{a_cells[(arm['key'], lvl)]['n']}"
            for lvl in LEVELS
        ]
        rows.append(
            [
                arm_short_tex(arm),
                *cells,
                fmt_p(arm_tests["fisher_control_p"]),
                fmt_p(arm_tests["permutation_p"]),
                fmt_p(arm_tests["fisher_L2_vs_L3_holm"]),
                fmt_p(arm_tests["fisher_L2_vs_L4_holm"]),
                fmt_p(arm_tests["fisher_L2_vs_L5_holm"]),
            ]
        )
    return latex_table(
        "Stage A: successful runs per guidance level (N = 10 per cell) for the 12 arms of the main matrix (three models, two prompting strategies, two dialects), "
        "with the exact tests per arm: Fisher's exact test L1 against L2--L5 pooled, the permutation test over L2--L5 "
        "(20\\,000 permutations), and Fisher's exact tests of L3, L4 and L5 against L2 (Holm-corrected within the arm).",
        "tab:stage-a",
        [
            "Arm",
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
            "L1 vs. rest",
            "L2--L5",
            "L3 vs. L2",
            "L4 vs. L2",
            "L5 vs. L2",
        ],
        rows,
        "lrrrrrrrrrr",
        note=ARM_LEGEND + " In E8 and E9 the L4/L5 feedback loop reacted to "
        "syntax failures only; no first draft failed there, so the later execution-feedback regime would not have changed a run. "
        "The last five columns are $p$-values; -- = test not applicable (no failure in L2--L5).",
        size=r"\scriptsize",
        midrules_after=MODEL_MIDRULES,
        tabcolsep="2.5pt",
    )


def tex_failure_classes(a_cells: dict[tuple[str, str], dict[str, Any]]) -> str:
    rows = []
    midrules = []
    for lvl in LEVELS:
        for key, label in OUTCOMES[1:]:
            counts = [a_cells[(arm["key"], lvl)]["outcomes"].get(key, 0) for arm in ARMS]
            if any(counts):
                rows.append([lvl, label, *[str(count) if count else "--" for count in counts]])
        if rows and rows[-1][0] == lvl:
            midrules.append(len(rows) - 1)
    header = [
        "Level",
        "Outcome class",
        *[f"\\rotatebox{{90}}{{{tex(arm['short'])} {arm['dialect'][0]}}}" for arm in ARMS],
    ]
    return latex_table(
        "Why runs fail: outcome classes of the unsuccessful runs per level and arm (counts of 10 runs per cell). "
        "Classes follow the pipeline stage at which the run stops.",
        "tab:failure-classes",
        header,
        rows,
        "ll" + "r" * len(ARMS),
        note="Columns: the 12 arms of the main matrix, model $\\cdot$ prompting, K = Kestrel, T = Tessera (same order as \\cref{tab:stage-a}). "
        "\\emph{Request rejected by the facade} includes requests whose URL path carried a quoted Gherkin literal. "
        "Requires \\texttt{graphicx} (loaded) for \\texttt{rotatebox}.",
        size=r"\scriptsize",
        midrules_after=midrules[:-1],
        tabcolsep="2pt",
    )


def _feedback_rows(
    a_cells: dict[tuple[str, str], dict[str, Any]], arms: Sequence[dict[str, Any]]
) -> list[list[str]]:
    rows = []
    for arm in arms:
        for lvl in ("L4", "L5"):
            cell = a_cells[(arm["key"], lvl)]
            l3_cell = a_cells[(arm["key"], "L3")]
            rows.append(
                [
                    arm_short_tex(arm) if lvl == "L4" else "",
                    ("execution" if arm["regime"].startswith("execution") else "syntax only")
                    if lvl == "L4"
                    else "",
                    lvl,
                    f"{l3_cell['success']}/{l3_cell['n']}",
                    str(cell["first_draft_ok"]),
                    str(cell["repaired"]),
                    str(cell["unrepaired"]),
                    str(cell["failed_no_feedback"]),
                    f"{cell['success']}/{cell['n']}",
                    str(cell["max_rounds"]),
                ]
            )
    return rows


FEEDBACK_HEADER = [
    "Arm",
    "Feedback",
    "Level",
    "L3",
    "First",
    "Rep.",
    "Not rep.",
    "No feedb.",
    "Final",
    "Rounds",
]
FEEDBACK_NOTE = (
    "First = first draft accepted, Rep. = repaired by feedback, Not rep. = still failing after feedback, No feedb. = failed "
    "without triggering feedback, Rounds = maximum feedback rounds in the cell (validator and execution rounds, at most five)."
)


def tex_feedback(a_cells: dict[tuple[str, str], dict[str, Any]]) -> str:
    return latex_table(
        "Effect of the feedback loop on L4 and L5 in the main matrix: runs whose first draft already passed, runs repaired by feedback, "
        "runs that stayed broken after feedback, and runs that failed without triggering feedback. "
        "The first L4 prompt equals the L3 prompt except for the two label lines, so the L3 column is the counterfactual without feedback.",
        "tab:feedback",
        FEEDBACK_HEADER,
        _feedback_rows(a_cells, ARMS),
        "lllrrrrrrr",
        note=ARM_LEGEND
        + " "
        + FEEDBACK_NOTE
        + " In E8 and E9 the loop reacted to syntax failures only; no first draft failed there. The feedback-regime contrast "
        "(syntax-only against execution feedback) was run with gemini-3.5-flash-lite during pipeline development and is reported "
        "separately (\\cref{tab:pipeline}).",
        size=r"\scriptsize",
        midrules_after=(7, 15),
        tabcolsep="3pt",
    )


def tex_pipeline(a_cells: dict[tuple[str, str], dict[str, Any]]) -> str:
    rows = []
    order = PIPELINE_ARMS + SYNTAX_ONLY_ARMS
    for arm in order:
        cells = [
            f"{a_cells[(arm['key'], lvl)]['success']}/{a_cells[(arm['key'], lvl)]['n']}"
            for lvl in LEVELS
        ]
        l4, l5 = a_cells[(arm["key"], "L4")], a_cells[(arm["key"], "L5")]
        rows.append(
            [
                arm_short_tex(arm),
                "execution" if arm["regime"].startswith("execution") else "syntax only",
                *cells,
                f"{l4['first_draft_ok']} $\\rightarrow$ {l4['success']}",
                f"{l5['first_draft_ok']} $\\rightarrow$ {l5['success']}",
            ]
        )
    return latex_table(
        "Pipeline development with gemini-3.5-flash-lite (not part of the model comparison): Stage A successes per level (N = 10 per cell) "
        "and, for L4/L5, successful first drafts $\\rightarrow$ final successes after the feedback loop. E6/E7 ran the same prompts and seeds "
        "as E10/E11 with syntax-only feedback; the pair shows what execution feedback adds to the pipeline.",
        "tab:pipeline",
        [
            "Arm",
            "L4/L5 feedback",
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
            "L4 first $\\rightarrow$ final",
            "L5 first $\\rightarrow$ final",
        ],
        rows,
        "llrrrrrrr",
        note=PIPELINE_LEGEND
        + " zs = zero-shot, ms = multi-shot, K = Kestrel, T = Tessera. The flash-lite multi-shot arms share E12/E13 with the gemini-3.5-flash "
        "multi-shot arms of the main matrix (interleaved), but are analysed apart. These runs serve as pipeline and measurement evidence only "
        "(feedback regime, test-retest noise in \\cref{tab:retest}), never as model evidence.",
        size=r"\scriptsize",
        midrules_after=(3,),
        tabcolsep="3pt",
    )


def _sb_cell_tex(cell: dict[str, Any], view: str) -> str:
    tp_values, tp_quartiles = cell_view(cell, "tp", view)
    _, p50_quartiles = cell_view(cell, "p50", view)
    if not tp_values or tp_quartiles is None or p50_quartiles is None:
        return "--"
    mark = (
        f"$^{{\\dagger {cell['n_stage_a_failed']}}}$"
        if view == "all" and cell["n_stage_a_failed"]
        else ""
    )
    return f"{tp_quartiles[1]:.1f} ({tp_quartiles[0]:.0f}--{tp_quartiles[2]:.0f}) / {p50_quartiles[1]:.1f}{mark}"


def tex_stage_b(
    sb_cells: dict[tuple[str, str, str], dict[str, Any]],
    reference: dict[str, dict[str, dict[str, Any]]],
    op: str,
    retest_range: str = "",
    missing: int = 0,
) -> str:
    rows = []
    midrules = []
    for i, arm in enumerate(ARMS):
        cells = [sb_cells[(arm["key"], lvl, op)] for lvl in LEVELS[1:]]
        rows.append([arm_short_tex(arm), *[_sb_cell_tex(cell, "all") for cell in cells]])
        if any(cell["n_stage_a_failed"] for cell in cells):
            rows.append(
                ["\\quad successes only", *[_sb_cell_tex(cell, "success") for cell in cells]]
            )
        if i in MODEL_MIDRULES:
            midrules.append(len(rows) - 1)
    ref_lines = []
    for series, ops in reference.items():
        ref = ops[op]
        ref_lines.append(
            f"{series}: {ref['tp_med']:.1f} ({ref['tp_q1']:.1f}--{ref['tp_q3']:.1f}) ops/s, p50 {ref['p50_med']:.1f} ms (N = {ref['n']})"
        )
    n_all, n_failed = view_counts(sb_cells, ARMS, (op,))
    return latex_table(
        f"Stage B, {op} operation, main matrix: median (interquartile range) of the throughput in ops/s / median of the latency p50 in ms "
        "per arm and guidance level, in two views. Arm rows: all measured runs, i.e.\\ every run that executed and produced a value for the "
        "operation, including runs that failed Stage A after executing (marked $^{\\dagger k}$, $k$ such runs in the cell). "
        "Rows \\emph{successes only}: the same cells restricted to the Stage-A successes (given only where the two views differ). "
        f"{n_all} measured runs, {n_failed} of them failed Stage A. L1 has no measured run in any arm.",
        f"tab:stage-b-{op}",
        ["Arm", "L2", "L3", "L4", "L5"],
        rows,
        "lllll",
        note=ARM_LEGEND
        + " Reference benchmark through the same pipeline: "
        + "; ".join(ref_lines)
        + ". "
        "Measurement days: E8/E9 on 28 Aug 2026, E12/E13 and the references E12/E13 on 13 Sep 2026, E14--E17 and the references "
        "E16/E17 on 9 Oct 2026; byte-identical artefacts measured twice differ by a factor of "
        + retest_range
        + " in write throughput (\\cref{tab:retest}), so values from different days are not compared. Stage-B failures (Stage-A successes "
        f"without a value for the operation): {missing}.",
        size=r"\scriptsize",
        midrules_after=tuple(midrules),
        tabcolsep="3pt",
    )


def tex_reference(reference: dict[str, dict[str, dict[str, Any]]]) -> str:
    rows = []
    for ops in reference.values():
        for op in OPS:
            stats = ops[op]
            rows.append(
                [
                    stats["dialect"] if op == "write" else "",
                    f"E{stats['epoch']}" if op == "write" else "",
                    op,
                    str(stats["n"]),
                    f"{stats['tp_med']:.1f}",
                    f"{stats['tp_q1']:.1f}--{stats['tp_q3']:.1f}",
                    f"{stats['tp_min']:.1f}--{stats['tp_max']:.1f}",
                    f"{stats['p50_med']:.1f}",
                    f"{stats['p95_med']:.1f}",
                    f"{stats['p99_med']:.1f}",
                ]
            )
    return latex_table(
        "The reference benchmark through the same pipeline: ten runs per dialect, each on a fresh InfluxDB. "
        "Throughput in ops/s (median, interquartile range, range), latency percentiles in ms (median over the ten runs).",
        "tab:reference",
        ["Dialect", "Epoch", "Operation", "N", "Throughput", "IQR", "Range", "p50", "p95", "p99"],
        rows,
        "lllrrrrrrr",
        note="The delete operation consists of one request per run, so its three percentiles coincide. "
        "Each series of ten reference runs was measured as one block of about 100 s right after the generated runs of its day "
        "(E12/E13 on 13 Sep 2026, E16/E17 on 9 Oct 2026), not interleaved with them.",
        size=r"\footnotesize",
        midrules_after=(2, 5, 8),
    )


def tex_delta(
    rows: Sequence[dict[str, Any]], sb_cells: dict[tuple[str, str, str], dict[str, Any]]
) -> str:
    arms = [arm for arm in ARMS if arm["ref"] is not None]
    n_all, n_failed = view_counts(sb_cells, arms)
    sub_header = (
        r"        & & $\Delta_{\text{tp}}$ & $\Delta_{\text{p50}}$ & $\Delta_{\text{tp}}$ & $\Delta_{\text{p50}}$ & "
        r"$\Delta_{\text{tp}}$ & $\Delta_{\text{p50}}$ \\"
    )
    header = [
        "Arm",
        "Level",
        "\\multicolumn{2}{c}{write}",
        "\\multicolumn{2}{c}{read}",
        "\\multicolumn{2}{c}{delete}",
    ]
    out = []
    for view in VIEWS:
        suffix = "" if view == "all" else "_success"
        table = []
        for arm in arms:
            for lvl in LEVELS[1:]:
                line = [arm_short_tex(arm) if lvl == "L2" else "", lvl]
                for op in OPS:
                    row = next(
                        (
                            entry
                            for entry in rows
                            if entry["arm"] == arm["key"]
                            and entry["level"] == lvl
                            and entry["op"] == op
                        ),
                        None,
                    )
                    if row is None or row[f"delta_tp_med{suffix}"] is None:
                        line += ["--", "--"]
                    else:
                        mark = "$^{\\dagger}$" if view == "all" and row["n_stage_a_failed"] else ""
                        line += [
                            f"{row[f'delta_tp_med{suffix}']:+.0f}{mark}",
                            f"{row[f'delta_p50_med{suffix}']:+.0f}{mark}",
                        ]
                table.append(line)
        if view == "all":
            caption = (
                "Deviation $\\Delta$ of the generated benchmarks from the reference (\\cref{eq:deviation}) in percent, view \\emph{all measured runs} "
                f"({n_all} runs, {n_failed} of them failed Stage A), for every main-matrix arm with a reference measured on the same day. "
                "Each entry is the deviation of the cell median from the median of that same-day reference series; throughput ($\\Delta_{\\text{tp}}$) "
                "and latency p50 ($\\Delta_{\\text{p50}}$) per operation. Positive throughput = faster than the reference, positive latency = slower."
            )
            label = "tab:delta"
            note = (
                ARM_LEGEND
                + " References: E12 (Tessera) and E13 (Kestrel) for the flash multi-shot arms, E16 (Kestrel) for haiku E14 and glm E16, "
                "E17 (Tessera) for haiku E15 and glm E17. $^{\\dagger}$ the cell contains measured runs that failed Stage A (see \\cref{tab:delta-success} "
                "for the Stage-A successes alone). The flash zero-shot arms E8/E9 have no same-day reference; their deviations are confounded with the "
                "drift between measurement days and are listed in the CSV only. Descriptive; no test."
            )
        else:
            caption = (
                "Deviation $\\Delta$ from the same-day reference in percent as in \\cref{tab:delta}, view \\emph{Stage-A successes only} "
                f"({n_all - n_failed} runs): the medians are taken over the Stage-A successes of each cell."
            )
            label = "tab:delta-success"
            note = ARM_LEGEND + " -- = no Stage-A success with a value in the cell."
        body = latex_table(
            caption,
            label,
            header,
            table,
            "llrrrrrr",
            note=note,
            size=r"\scriptsize",
            midrules_after=tuple(range(3, 4 * len(arms) - 1, 4)),
            tabcolsep="4pt",
        )
        out.append(
            body.replace(
                r"        \midrule" + "\n", sub_header + "\n" + r"        \midrule" + "\n", 1
            )
        )
    return "\n".join(out)


def tex_stage_b_tests(rows: Sequence[dict[str, Any]]) -> str:
    out = []
    for view in VIEWS:
        suffix = "" if view == "all" else "_success"
        table = []
        for arm in ARMS:
            per_op = {
                op: next(
                    entry for entry in rows if entry["arm"] == arm["key"] and entry["op"] == op
                )
                for op in OPS
            }
            sizes = {per_op[op][f"n{suffix}"] for op in OPS}
            line = [
                arm_short_tex(arm),
                "/".join(str(per_op[op][f"n{suffix}"]) for op in OPS)
                if len(sizes) > 1
                else str(sizes.pop()),
            ]
            for op in OPS:
                row = per_op[op]
                line += [
                    fmt(row[f"kw_H{suffix}"], 1),
                    fmt_p(row[f"kw_holm{suffix}"]),
                    fmt(row[f"rho{suffix}"], 2),
                    fmt_p(row[f"rho{suffix}_holm"]),
                ]
            table.append(line)
        header = [
            "Arm",
            "$n$",
            "\\multicolumn{4}{c}{write}",
            "\\multicolumn{4}{c}{read}",
            "\\multicolumn{4}{c}{delete}",
        ]
        if view == "all":
            caption = (
                "Stage B tests per main-matrix arm and operation, view \\emph{all measured runs}: Kruskal--Wallis $H$ across L2--L5 and Spearman $\\rho$ "
                "between guidance level and throughput, with Holm-corrected $p$-values within the arm and operation ($n$ = measured runs, including "
                "those that failed Stage A). The Mann--Whitney test is not applicable within an arm (one prompting strategy per arm)."
            )
            label = "tab:stage-b-tests"
        else:
            caption = "Stage B tests as in \\cref{tab:stage-b-tests}, view \\emph{Stage-A successes only} ($n$ = Stage-A successes with a value)."
            label = "tab:stage-b-tests-success"
        body = latex_table(
            caption,
            label,
            header,
            table,
            "lr" + "rrrr" * 3,
            note=ARM_LEGEND
            + " Exploratory tests (not fixed in advance). With ten repetitions per level only large effects can be detected. Holm over all 36 "
            "Spearman tests of a view is in \\texttt{stage\\_b\\_tests\\_by\\_arm.csv} (rho\\_holm\\_all, rho\\_success\\_holm\\_all).",
            size=r"\scriptsize",
            midrules_after=MODEL_MIDRULES,
            tabcolsep="2.5pt",
        )
        out.append(
            body.replace(
                r"        \midrule" + "\n",
                r"        & & $H$ & $p$ & $\rho$ & $p$ & $H$ & $p$ & $\rho$ & $p$ & $H$ & $p$ & $\rho$ & $p$ \\"
                + "\n"
                + r"        \midrule"
                + "\n",
                1,
            )
        )
    return "\n".join(out)


def tex_prompting(stage_a: Sequence[dict[str, Any]], stage_b: Sequence[dict[str, Any]]) -> str:
    rows = []
    for model in MAIN_MODELS:
        for dialect in DIALECTS:
            selected = [
                row for row in stage_a if row["model"] == model and row["dialect"] == dialect
            ]
            line = [
                MODEL_LABEL[model],
                dialect,
                f"E{selected[0]['zs_epoch']} / E{selected[0]['ms_epoch']}",
            ]
            for row in selected:
                line.append(
                    f"{row['zs_success']} / {row['ms_success']}"
                    + (
                        "$^*$"
                        if row["fisher_holm"] is not None and row["fisher_holm"] < 0.05
                        else ""
                    )
                )
            rows.append(line)
    table_a = latex_table(
        "Zero-shot against multi-shot prompting, Stage A: successful runs out of 10 per level (zero-shot / multi-shot) for each model of the "
        "main matrix and dialect. Fisher's exact test per level, Holm-corrected over the five levels of a row; $^*$ marks $p < 0.05$.",
        "tab:prompting-stage-a",
        ["Model", "Dialect", "Epochs zs / ms", "L1", "L2", "L3", "L4", "L5"],
        rows,
        "lllrrrrr",
        note="flash = gemini-3.5-flash: the two strategies were generated in different epochs (zero-shot on 28 Aug 2026, multi-shot on 13 Sep 2026) "
        "with the same seeds and the same prompts except for the appended demonstration. haiku = claude-haiku-5-5 and glm = glm-5.3: both "
        "strategies interleaved within one epoch.",
        size=r"\footnotesize",
        midrules_after=(1, 3),
    )
    rows_b = []
    for row in stage_b:
        if row["op"] != "write":
            continue
        rows_b.append(
            [
                f"E{row['epoch']}",
                row["dialect"],
                MODEL_LABEL[row["model"]],
                f"{row['zs_n']}/{row['ms_n']}",
                fmt(row["zs_median"]),
                fmt(row["ms_median"]),
                fmt_p(row["mwu_p"]),
                f"{row['zs_n_success']}/{row['ms_n_success']}",
                fmt(row["zs_median_success"]),
                fmt(row["ms_median_success"]),
                fmt_p(row["mwu_p_success"]),
            ]
        )
    table_b = latex_table(
        "Zero-shot against multi-shot prompting, Stage B: write throughput (ops/s, median) of the L3 runs within one epoch, with the "
        "Mann--Whitney U test, in both views (all measured runs; Stage-A successes only).",
        "tab:prompting-stage-b",
        [
            "Epoch",
            "Dialect",
            "Model",
            "$n$ zs/ms",
            "Med. zs",
            "Med. ms",
            "$p$",
            "$n$ zs/ms",
            "Med. zs",
            "Med. ms",
            "$p$",
        ],
        rows_b,
        "lllrrrrrrrr",
        note="Columns 4--7: all measured runs; columns 8--11: Stage-A successes only. E8/E9 carry both strategies at L3 for flash (the 10 multi-shot L3 runs per epoch belong to no main-matrix arm, flash multi-shot is E12/E13); "
        "haiku (E14/E15) and glm (E16/E17) interleave both strategies at every level. Within-epoch comparison, so free of the drift "
        "between measurement days; uncorrected $p$-values.",
        size=r"\scriptsize",
        midrules_after=(1, 3),
        tabcolsep="3pt",
    )
    return table_a + "\n" + table_b


def tex_retest(
    rows: Sequence[dict[str, Any]], reference: dict[str, dict[str, dict[str, Any]]]
) -> str:
    table = []
    for first, second, dialect in RETEST_PAIRS:
        pair_rows = [row for row in rows if row["pair"] == f"E{first}->E{second}"]
        identical = [row for row in pair_rows if row["identical_artifact"]]
        measured_twice = [row for row in identical if row["tp_ratio"]]
        ratios = np.array([row["tp_ratio"] for row in measured_twice])
        p50_ratios = np.array([row["p50_ratio"] for row in measured_twice if row["p50_ratio"]])
        q1, med, q3 = quart(ratios)
        p50_q1, p50_med, p50_q3 = quart(p50_ratios)
        table.append(
            [
                f"E{first} → E{second}",
                dialect,
                str(len(pair_rows)),
                str(len(identical)),
                str(len(measured_twice)),
                f"{med:.2f}",
                f"{q1:.2f}--{q3:.2f}",
                f"{ratios.min():.2f}--{ratios.max():.2f}",
                f"{p50_med:.2f}",
                f"{p50_q1:.2f}--{p50_q3:.2f}",
            ]
        )
    ref_note = []
    for dialect in ("Kestrel", "Tessera"):
        ref = reference[MAIN_REFERENCE[dialect]]["write"]
        ref_note.append(
            f"{dialect} E{ref['epoch']}: write throughput {ref['tp_min']:.1f}--{ref['tp_max']:.1f} ops/s over ten runs "
            f"(max/min {ref['tp_max'] / ref['tp_min']:.2f})"
        )
    return latex_table(
        "Measurement noise of Stage B: the same artefact measured twice. The pairs are gemini-3.5-flash-lite pipeline-development runs, used only "
        "as measurement-noise evidence, not as model evidence. E10/E11 reused the prompts and seeds of E6/E7, and the model "
        "returned byte-identical artefacts for part of the cells; the ratio of the second to the first measurement of these "
        "identical artefacts is pure measurement noise (write throughput and write latency p50).",
        "tab:retest",
        [
            "Pair",
            "Dialect",
            "Pairs",
            "Identical",
            "Twice",
            "Ratio tp",
            "IQR",
            "Range",
            "Ratio p50",
            "IQR",
        ],
        [[tex(cell) for cell in row] for row in table],
        "llrrrrrrrr",
        note="Identical = \\texttt{benchmark.feature} and \\texttt{steps.py} byte-identical in both epochs; Twice = identical and "
        "with a Stage B value in both epochs; ratios are second over first measurement (tp = write throughput, p50 = write latency "
        "p50), median over the pairs. The pairs are a test-retest of the same client, not independent replications of the "
        "generation. The reference itself spreads the same way: " + "; ".join(ref_note) + ".",
        size=r"\scriptsize",
        tabcolsep="2.5pt",
    )


def tex_pilot(rows: Sequence[dict[str, Any]]) -> str:
    table: list[list[str]] = []
    last_pilot = 0
    for row in rows:
        api = {"influx": "InfluxDB", "tessera": "Tessera", "kestrel": "Kestrel"}.get(
            row["api"], row["api"]
        )
        cells = [
            f"{row[f'{lvl}_success']}/{row[f'{lvl}_n']}" if row[f"{lvl}_n"] else "--"
            for lvl in LEVELS
        ]
        epoch = f"E{row['epoch']}"
        if row["model"]:
            epoch += " (lite)" if "lite" in row["model"] else " (flash)"
        if row["role"] == "pilot":
            last_pilot = len(table)
        table.append(
            [
                epoch,
                ROLE_SHORT[row["role"]],
                api,
                str(row["runs"]),
                *cells,
                str(row["canary_runs"]),
                str(row["legacy_requests"]),
                str(row["vocab_leak_runs"]),
            ]
        )
    return latex_table(
        "Contamination control across all epochs: Stage A success per level (criteria v2), runs with at least one InfluxDB-specific request "
        "form rejected by the facade (canary), requests to the InfluxDB endpoints behind the facade (legacy), and artefacts containing InfluxDB "
        "vocabulary (static scan). E12/E13 are split into their gemini-3.5-flash (main matrix) and gemini-3.5-flash-lite (pipeline development) halves.",
        "tab:contamination",
        [
            "Epoch",
            "Role",
            "API",
            "Runs",
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
            "Canary",
            "Legacy",
            "Vocab.",
        ],
        table,
        "lllrrrrrrrrr",
        note="Role as in \\cref{tab:dataset} (pipe. = pipeline development with gemini-3.5-flash-lite, not part of the model comparison). "
        "API InfluxDB = its API in the prompt and on the wire, no facade. E1 artefacts were re-evaluated under criteria v2; E2 = E1 with "
        "the pinned L1 workload; E3 = first facade with InfluxDB traces left on the wire; E4 onwards neutralised facade. The L1 row of "
        "E2 (3/3) is the memorisation case: without any specification the model wrote a working InfluxDB client. Canary and Vocab. "
        "count runs, Legacy counts requests.",
        size=r"\scriptsize",
        midrules_after=(last_pilot,),
        tabcolsep="2.5pt",
    )


def tex_tokens(rows: Sequence[dict[str, Any]]) -> str:
    table = []
    for arm in ARMS:
        line = [arm_short_tex(arm)]
        for lvl in LEVELS:
            row = next(
                entry for entry in rows if entry["arm"] == arm["key"] and entry["level"] == lvl
            )
            if row["cost_usd_per_run"] is None:
                line.append(
                    f"{row['prompt_tokens_median'] / 1000:.1f}/--/--"
                    if row["prompt_tokens_median"]
                    else "--"
                )
            else:
                per_success = (
                    f"{row['cost_usd_per_success']:.3f}" if row["cost_usd_per_success"] else "--"
                )
                line.append(
                    f"{row['prompt_tokens_median'] / 1000:.1f}/{row['cost_usd_per_run']:.3f}/{per_success}"
                )
        table.append(line)
    return latex_table(
        "Guidance material and cost per level in the main matrix: median prompt tokens of a run (all model turns, in thousands) / list-price cost per run (USD) / "
        "cost per successful run (USD).",
        "tab:tokens",
        ["Arm", "L1", "L2", "L3", "L4", "L5"],
        table,
        "llllll",
        note=ARM_LEGEND
        + " Prices: gemini-3.5-flash 1.50 / 9.00 USD (verified 28 Aug 2026), glm-5.3 1.40 / 4.40 USD "
        "(Z.AI on OpenRouter, verified 9 Oct 2026) per million input / output tokens. claude-haiku-5-5 0.10 / 0.50 USD (cache hits 0.01; verified 9 Oct 2026), not billed: it ran on a subscription through the "
        "Claude Code CLI. Cached prompt tokens are priced at the cache price of each provider. L1 includes the probe turns, L4/L5 the "
        "feedback rounds.",
        size=r"\scriptsize",
        midrules_after=MODEL_MIDRULES,
        tabcolsep="2pt",
    )


def tex_figures(names: Sequence[tuple[str, str, str]]) -> str:
    blocks = []
    for name, label, caption in names:
        blocks.append(
            "\n".join(
                [
                    r"\begin{figure}[H]",
                    r"    \centering",
                    f"    \\includegraphics[width=\\columnwidth]{{fig/{name}.pdf}}",
                    f"    \\caption{{{caption}}}",
                    f"    \\label{{{label}}}",
                    r"\end{figure}",
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


STALE_OUTPUTS: tuple[str, ...] = (
    *(
        f"fig/{name}.{ext}"
        for ext in ("pdf", "png")
        for name in (
            "stage_a_model_comparison",
            "stage_b_delta_vs_reference",
            "stage_b_spearman_by_arm",
            *(f"stage_b_delta_vs_reference_{dialect.lower()}" for dialect in DIALECTS),
            *(f"stage_b_{op}_{dialect.lower()}" for op in OPS for dialect in DIALECTS),
            *(f"stage_b_{op}_{dialect.lower()}_haiku_glm" for op in OPS for dialect in DIALECTS),
        )
    ),
    "tables/model_stage_a.csv",
    "tables/model_stage_b.csv",
    "latex/tab_model.tex",
)


def _rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    write_csv(path, list(rows[0].keys()), [list(row.values()) for row in rows])


def pipeline_rows(
    a_cells: dict[tuple[str, str], dict[str, Any]],
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for arm in PIPELINE_ARMS + SYNTAX_ONLY_ARMS:
        for lvl in LEVELS:
            cell = a_cells[(arm["key"], lvl)]
            write_cell = b_cells[(arm["key"], lvl, "write")]
            rows.append(
                {
                    "role": "pipeline development (flash-lite), not model evidence",
                    "arm": arm["key"],
                    "model": MODEL_FULL[arm["model"]],
                    "prompting": arm["prompting"],
                    "dialect": arm["dialect"],
                    "epoch": arm["epoch"],
                    "feedback_regime": arm["regime"],
                    "level": lvl,
                    "n": cell["n"],
                    "success": cell["success"],
                    "first_draft_ok": cell["first_draft_ok"],
                    "repaired": cell["repaired"],
                    "unrepaired": cell["unrepaired"],
                    "failed_no_feedback": cell["failed_no_feedback"],
                    "max_rounds": cell["max_rounds"],
                    "measured_write": write_cell["n"],
                    "measured_write_stage_a_failed": write_cell["n_stage_a_failed"],
                    "write_tp_median_all": write_cell.get("tp_med"),
                    "write_tp_median_success": write_cell.get("tp_s_med"),
                }
            )
    return rows


def _write_stage_a_cells_csv(tab_dir: Path, a_cells: dict[tuple[str, str], dict[str, Any]]) -> None:
    rows = []
    for arm in ARMS:
        for lvl in LEVELS:
            cell = a_cells[(arm["key"], lvl)]
            rows.append(
                [
                    arm["key"],
                    MODEL_FULL[arm["model"]],
                    arm["prompting"],
                    arm["dialect"],
                    arm["epoch"],
                    lvl,
                    cell["n"],
                    cell["success"],
                    cell["syntax_ok"],
                    cell["executed_ok"],
                    cell["stage_b_write"],
                    *[cell[f"measured_{op}"] for op in OPS],
                    *[cell[f"measured_stage_a_failed_{op}"] for op in OPS],
                    *[cell[f"stage_b_missing_{op}"] for op in OPS],
                    cell["first_draft_ok"],
                    cell["repaired"],
                    cell["unrepaired"],
                    cell["failed_no_feedback"],
                    cell["max_rounds"],
                    cell["exec_rounds_runs"],
                    *[cell["outcomes"].get(key, 0) for key, _ in OUTCOMES],
                    "; ".join(
                        f"{detail} x{count}" for detail, count in cell["failure_details"].items()
                    ),
                ]
            )
    write_csv(
        tab_dir / "stage_a_cells.csv",
        [
            "arm",
            "model",
            "prompting",
            "dialect",
            "epoch",
            "level",
            "n",
            "success",
            "syntax_ok",
            "executed_ok",
            "stage_b_write",
            *[f"measured_{op}" for op in OPS],
            *[f"measured_stage_a_failed_{op}" for op in OPS],
            *[f"stage_b_missing_{op}" for op in OPS],
            "first_draft_ok",
            "repaired",
            "unrepaired",
            "failed_no_feedback",
            "max_rounds",
            "exec_feedback_runs",
            *[key for key, _ in OUTCOMES],
            "failure_details",
        ],
        rows,
    )


def _write_stage_b_cells_csv(
    tab_dir: Path, b_cells: dict[tuple[str, str, str], dict[str, Any]]
) -> None:
    rows = []
    for arm in ARMS:
        for lvl in LEVELS:
            for op in OPS:
                cell = b_cells[(arm["key"], lvl, op)]
                rows.append(
                    [
                        arm["key"],
                        MODEL_FULL[arm["model"]],
                        arm["prompting"],
                        arm["dialect"],
                        arm["epoch"],
                        lvl,
                        op,
                        cell["n"],
                        cell["n_stage_a_failed"],
                        cell["n_success"],
                        cell.get("tp_med"),
                        cell.get("tp_q1"),
                        cell.get("tp_q3"),
                        cell.get("p50_med"),
                        cell.get("p50_q1"),
                        cell.get("p50_q3"),
                        cell.get("p95_med"),
                        cell.get("p99_med"),
                        cell.get("tp_s_med"),
                        cell.get("tp_s_q1"),
                        cell.get("tp_s_q3"),
                        cell.get("p50_s_med"),
                        cell.get("p50_s_q1"),
                        cell.get("p50_s_q3"),
                    ]
                )
    write_csv(
        tab_dir / "stage_b_cells.csv",
        [
            "arm",
            "model",
            "prompting",
            "dialect",
            "epoch",
            "level",
            "operation",
            "n_measured",
            "n_stage_a_failed",
            "n_success",
            "tp_median_all",
            "tp_q1_all",
            "tp_q3_all",
            "p50_median_all",
            "p50_q1_all",
            "p50_q3_all",
            "p95_median_all",
            "p99_median_all",
            "tp_median_success",
            "tp_q1_success",
            "tp_q3_success",
            "p50_median_success",
            "p50_q1_success",
            "p50_q3_success",
        ],
        rows,
    )


def _write_stage_b_runs_csv(tab_dir: Path, runs: Sequence[RunRecord]) -> None:
    rows = []
    for arm in ARMS:
        ordered = sorted(
            arm_runs(runs, arm),
            key=lambda candidate: (candidate.guidance_level or "", candidate.run_id),
        )
        for run in ordered:
            measured = any(
                run.stage_b_value(SOURCE_PROXY, op, "throughput_ops_s") is not None for op in OPS
            )
            values: list[float | None] = []
            for op in OPS:
                stats = run.op(SOURCE_PROXY, op)
                values += [
                    run.stage_b_value(SOURCE_PROXY, op, "throughput_ops_s"),
                    run.stage_b_value(SOURCE_PROXY, op, "latency_p50_ms"),
                    run.stage_b_value(SOURCE_PROXY, op, "latency_p95_ms"),
                    run.stage_b_value(SOURCE_PROXY, op, "latency_p99_ms"),
                    stats.n_requests if stats else 0,
                    stats.error_count if stats else 0,
                ]
            rows.append(
                [
                    arm["epoch"],
                    arm["key"],
                    run.run_id,
                    run.guidance_level,
                    MODEL_FULL[arm["model"]],
                    arm["prompting"],
                    arm["dialect"],
                    run.seed,
                    run.success,
                    measured,
                    measured and run.success,
                    run.feedback_iterations,
                    run.execution_feedback_rounds,
                    run.failure_class,
                    *values,
                ]
            )
    write_csv(
        tab_dir / "stage_b_runs.csv",
        [
            "epoch",
            "arm",
            "run_id",
            "level",
            "model",
            "prompting",
            "dialect",
            "seed",
            "stage_a_success",
            "in_view_all",
            "in_view_success",
            "feedback_iterations",
            "execution_feedback_rounds",
            "failure_class",
            *[
                f"{op}_{metric}"
                for op in OPS
                for metric in (
                    "throughput_ops_s",
                    "p50_ms",
                    "p95_ms",
                    "p99_ms",
                    "n_requests",
                    "error_count",
                )
            ],
        ],
        rows,
    )


def _write_reference_csv(tab_dir: Path, reference: dict[str, dict[str, dict[str, Any]]]) -> None:
    write_csv(
        tab_dir / "reference_runs.csv",
        [
            "dialect",
            "epoch",
            "operation",
            "n",
            "tp_median",
            "tp_q1",
            "tp_q3",
            "tp_min",
            "tp_max",
            "p50_median",
            "p50_q1",
            "p50_q3",
            "p95_median",
            "p99_median",
            "tp_values",
            "evaluated_at",
        ],
        [
            [
                series,
                stats["epoch"],
                op,
                stats["n"],
                stats["tp_med"],
                stats["tp_q1"],
                stats["tp_q3"],
                stats["tp_min"],
                stats["tp_max"],
                stats["p50_med"],
                stats["p50_q1"],
                stats["p50_q3"],
                stats["p95_med"],
                stats["p99_med"],
                ";".join(f"{value:.2f}" for value in stats["tp"]),
                stats["evaluated"][0] + ".." + stats["evaluated"][-1],
            ]
            for series in reference
            for op, stats in reference[series].items()
        ],
    )


def _stage_a_figures(
    a_cells: dict[tuple[str, str], dict[str, Any]], status: Sequence[dict[str, Any]], fig_dir: Path
) -> list[tuple[str, str, str]]:
    figures: list[tuple[str, str, str]] = []
    figures.append(
        (
            fig_stage_a_matrix(a_cells, fig_dir),
            "fig:stage-a-matrix",
            "Stage A: share of successful runs per guidance level for the 12 arms of the main matrix (three models, two prompting "
            "strategies, two dialects; N = 10 runs per cell; numbers are successes out of 10). flash = gemini-3.5-flash, "
            "haiku = claude-haiku-5-5, glm = glm-5.3.",
        )
    )
    figures.append(
        (
            fig_failure_classes(a_cells, fig_dir),
            "fig:failure-classes",
            "Outcome of the 10 runs of every main-matrix cell by pipeline stage: the stage at which an unsuccessful run stops, per level "
            "and arm. Rows in the order of \\cref{fig:stage-a-matrix}.",
        )
    )
    figures.append(
        (
            fig_status(status, fig_dir),
            "fig:measurement-status",
            "Where each main-matrix run leaves the pipeline, separating Stage A from Stage B: Stage-A successes (all measured), runs that "
            "were measured but failed Stage A (kept in the all-measured-runs view of Stage B and marked there), runs that ran with errors "
            "and sent partial traffic but carry no Stage-B value, and runs without any accepted request. Rows in the order of "
            "\\cref{fig:stage-a-matrix}.",
        )
    )
    figures.append(
        (
            fig_feedback(a_cells, fig_dir),
            "fig:feedback",
            "Main matrix, L4 and L5: successful runs out of 10 before (first draft) and after the feedback loop. In E8/E9 the loop "
            "reacted to syntax failures only (no first draft failed there); all other arms ran with execution feedback.",
        )
    )
    return figures


def _stage_b_view_figures(
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
    reference: dict[str, dict[str, dict[str, Any]]],
    fig_dir: Path,
) -> list[tuple[str, str, str]]:
    figures: list[tuple[str, str, str]] = []
    for dialect in DIALECTS:
        for op in OPS:
            for view in VIEWS:
                name = fig_stage_b(b_cells, reference, dialect, op, view, fig_dir)
                n_all, n_failed = view_counts(
                    b_cells, [arm for arm in ARMS if arm["dialect"] == dialect], (op,)
                )
                if view == "all":
                    shown = (
                        f"view \\emph{{all measured runs}} ({n_all} runs, {n_failed} of them failed Stage A and are drawn as $\\times$; "
                        "median and interquartile range over all of them)"
                    )
                else:
                    shown = f"view \\emph{{Stage-A successes only}} ({n_all - n_failed} runs)"
                figures.append(
                    (
                        name,
                        f"fig:stage-b-{op}-{dialect.lower()}-{view}",
                        f"Stage B, {dialect} dialect, {op} operation, {shown}: throughput (top) and latency p50 (bottom) per run by "
                        "guidance level, one panel per main-matrix arm (flash, haiku, glm, each zero- and multi-shot), with the median "
                        "and interquartile range per level. The same-day reference (median and interquartile range of its ten runs) is "
                        "drawn where one was measured on the arm's day; flash zero-shot (E8/E9, 28 Aug 2026) has none. Measurement days: flash multi-shot "
                        "E12/E13 13 Sep 2026, haiku E14/E15 and glm E16/E17 9 Oct 2026. L1 has no "
                        "measured run. Descriptive.",
                    )
                )
    return figures


def _delta_figures(
    deltas: Sequence[dict[str, Any]],
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
    fig_dir: Path,
) -> list[tuple[str, str, str]]:
    figures: list[tuple[str, str, str]] = []
    for dialect in DIALECTS:
        for view in VIEWS:
            name = fig_delta(deltas, b_cells, dialect, view, fig_dir)
            dialect_arms = [
                arm for arm in ARMS if arm["dialect"] == dialect and arm["ref"] is not None
            ]
            n_all, n_failed = view_counts(b_cells, dialect_arms)
            if view == "all":
                shown = (
                    f"view \\emph{{all measured runs}} ({n_all} runs, {n_failed} of them failed Stage A; a dagger marks cells that "
                    "contain such runs)"
                )
            else:
                shown = f"view \\emph{{Stage-A successes only}} ({n_all - n_failed} runs)"
            figures.append(
                (
                    name,
                    f"fig:delta-{dialect.lower()}-{view}",
                    f"Deviation from the same-day reference (\\cref{{eq:deviation}}), {dialect} dialect, {shown}, per operation and "
                    "main-matrix arm with a same-day reference: markers are the deviation of the cell median (circle = multi-shot, "
                    "square = zero-shot), lines the deviation of its first and third quartile. Top: throughput (positive = faster "
                    "than the reference), bottom: latency p50 (positive = slower). Descriptive.",
                )
            )
    return figures


def _prompting_figures(
    prompt_a: Sequence[dict[str, Any]],
    a_cells: dict[tuple[str, str], dict[str, Any]],
    fig_dir: Path,
) -> list[tuple[str, str, str]]:
    figures: list[tuple[str, str, str]] = []
    sig_prompt = {
        (f"{MODEL_LABEL[row['model']]} · {row['dialect']}", row["level"]): row["fisher_holm"]
        for row in prompt_a
    }
    figures.append(
        (
            fig_comparison(
                [
                    (
                        f"{MODEL_LABEL[model]} · {dialect}",
                        _arm_of(model, "zero_shot", dialect),
                        _arm_of(model, "multi_shot", dialect),
                    )
                    for dialect in DIALECTS
                    for model in MAIN_MODELS
                ],
                a_cells,
                RAMP_2,
                ("zero-shot", "multi-shot"),
                sig_prompt,
                "stage_a_prompting_comparison",
                fig_dir,
                ncols=3,
            ),
            "fig:prompting-comparison",
            "Prompting comparison, Stage A: successful runs per level for zero-shot and multi-shot prompting, one panel per model of "
            "the main matrix (columns) and dialect (rows). flash: two epochs (zero-shot 28 Aug, multi-shot 13 Sep 2026); haiku and glm: "
            "both strategies interleaved in one epoch. Asterisks mark levels where Fisher's exact test (Holm-corrected over the five "
            "levels) gives $p < 0.05$.",
        )
    )
    return figures


def _spearman_figures(
    b_tests: Sequence[dict[str, Any]],
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
    any_counts: tuple[int, int],
    fig_dir: Path,
) -> list[tuple[str, str, str]]:
    n_any, n_any_failed = any_counts
    figures: list[tuple[str, str, str]] = []
    for view in VIEWS:
        if view == "all":
            shown = f"view \\emph{{all measured runs}} ({n_any} runs, {n_any_failed} of them failed Stage A)"
            holm_col = "rho\\_holm\\_all"
        else:
            shown = f"view \\emph{{Stage-A successes only}} ({n_any - n_any_failed} runs; $\\rho$ and its interval from the successes' own $n$)"
            holm_col = "rho\\_success\\_holm\\_all"
        n_sig = sum(
            1
            for row in b_tests
            if row["rho_holm_all" if view == "all" else "rho_success_holm_all"] is not None
            and row["rho_holm_all" if view == "all" else "rho_success_holm_all"] < 0.05
        )
        figures.append(
            (
                fig_spearman(b_tests, b_cells, view, fig_dir),
                f"fig:spearman-{view}",
                f"Rank correlation between guidance level (L2 = 2 to L5 = 5) and throughput per run, per main-matrix arm and operation, "
                f"{shown}, with 95\\,\\% confidence intervals (Fisher transformation). Filled markers indicate $p < 0.05$ "
                f"(uncorrected, exploratory); {n_sig} of the 36 tests {'stays' if n_sig == 1 else 'stay'} below 0.05 after Holm over all 36 tests of this view "
                f"(column {holm_col} in the CSV).",
            )
        )
    return figures


def _stage_b_run_figures(
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
    reference: dict[str, dict[str, dict[str, Any]]],
    fig_dir: Path,
) -> list[tuple[str, str, str]]:
    figures: list[tuple[str, str, str]] = []
    for dialect in DIALECTS:
        dialect_arms = [arm for arm in ARMS if arm["dialect"] == dialect]
        for op in OPS:
            name, n_outliers = fig_stage_b_runs(b_cells, dialect, op, fig_dir)
            n_success = sum(
                b_cells[(arm["key"], lvl, op)]["n_success"]
                for arm in dialect_arms
                for lvl in LEVELS[1:]
            )
            figures.append(
                (
                    name,
                    f"fig:stage-b-{op}-{dialect.lower()}-runs",
                    f"Stage B, {dialect} dialect, {op} operation: throughput (top) and latency p50 (bottom) of every run that passed "
                    f"Stage A ({n_success} runs), one panel per main-matrix arm, by guidance level, with the median per level. Runs outside "
                    f"the Tukey fences of their cell (1.5 interquartile ranges beyond the quartiles) are ringed in red ({n_outliers} of "
                    f"{n_success} values in this figure). L1 has no run that passed Stage A. Descriptive.",
                )
            )
            name = fig_stage_b_candles(b_cells, reference, dialect, op, fig_dir)
            figures.append(
                (
                    name,
                    f"fig:stage-b-{op}-{dialect.lower()}-candles",
                    f"Stage B, {dialect} dialect, {op} operation, Stage-A successes: one candle per arm and level (box = interquartile "
                    "range, line = median, wicks = Tukey fences, red rings = outliers) next to the candle of the reference benchmark "
                    "measured on the same day (grey, N = 10; dashed line = its median). flash zero-shot (E8/E9, 28 Aug 2026) has no "
                    "same-day reference. Descriptive.",
                )
            )
    return figures


def _pipeline_figures(
    a_cells: dict[tuple[str, str], dict[str, Any]], fig_dir: Path
) -> list[tuple[str, str, str]]:
    figures: list[tuple[str, str, str]] = []
    figures.append(
        (
            fig_stage_a_matrix(
                a_cells,
                fig_dir,
                arms=PIPELINE_ARMS + SYNTAX_ONLY_ARMS,
                name="pipeline_stage_a_flash_lite",
                header=PIPELINE_HEADER,
                group_key="regime",
            ),
            "fig:pipeline-stage-a",
            "Pipeline development (gemini-3.5-flash-lite), not part of the model comparison: Stage A successes per level of the "
            "flash-lite epochs used to develop and test the pipeline cheaply (E10/E11 zero-shot, the flash-lite half of E12/E13 "
            "multi-shot, and E6/E7 with syntax-only feedback). Shown only as pipeline evidence.",
        )
    )
    regime_rows = [SYNTAX_ONLY_ARMS[0], PIPELINE_ARMS[0], SYNTAX_ONLY_ARMS[1], PIPELINE_ARMS[1]]
    figures.append(
        (
            fig_feedback(
                a_cells,
                fig_dir,
                rows=regime_rows,
                name="pipeline_feedback_regime",
                header=PIPELINE_HEADER,
                group_key="dialect",
                row_labels=[
                    f"{arm['dialect']} E{arm['epoch']} · "
                    + ("syntax-only feedback" if arm["epoch"] < 10 else "execution feedback")
                    for arm in regime_rows
                ],
            ),
            "fig:pipeline-feedback-regime",
            "Pipeline development (gemini-3.5-flash-lite, zero-shot), not part of the model comparison: L4 and L5 successes out of 10 "
            "for the first draft and after the feedback loop, syntax-only feedback (E6/E7) against execution feedback (E10/E11) with "
            "the same prompts and seeds. Evidence for the pipeline design, not for a model.",
        )
    )
    return figures


def main() -> None:
    config, runs = load_runs()
    out = config.paths.results_reports_dir / "thesis_results"
    fig_dir, tab_dir, tex_dir = out / "fig", out / "tables", out / "latex"
    for directory in (fig_dir, tab_dir, tex_dir):
        directory.mkdir(parents=True, exist_ok=True)
    removed = []
    for rel in STALE_OUTPUTS:
        path = out / rel
        if path.exists():
            path.unlink()
            removed.append(rel)
    _style()

    a_cells = stage_a_cells(runs)
    a_tests = stage_a_tests(runs)
    b_cells = stage_b_cells(runs)
    reference = reference_stats(runs)
    deltas = delta_rows(b_cells, reference)
    b_tests = stage_b_tests(runs)
    add_global_holm(b_tests)
    battery = epoch_battery(runs)
    prompt_a, prompt_b = prompting_comparison(runs, a_cells)
    retest = retest_pairs(runs, config.paths.project_root)
    pilots = pilot_rows(runs)
    tokens = token_rows(runs, config)
    dataset = dataset_rows(runs, config)
    pipeline = pipeline_rows(a_cells, b_cells)
    status = status_rows(runs)

    counts = {op: view_counts(b_cells, ARMS, (op,)) for op in OPS}
    counts["any"] = view_counts(b_cells, ARMS, OPS)
    l1_measured = sum(b_cells[(arm["key"], "L1", op)]["n"] for arm in ARMS for op in OPS)
    stage_b_missing = {
        op: sum(
            a_cells[(arm["key"], lvl)][f"stage_b_missing_{op}"] for arm in ARMS for lvl in LEVELS
        )
        for op in OPS
    }

    _rows_csv(tab_dir / "dataset_epochs.csv", dataset)
    _write_stage_a_cells_csv(tab_dir, a_cells)
    _rows_csv(tab_dir / "stage_a_tests.csv", a_tests)
    _write_stage_b_cells_csv(tab_dir, b_cells)
    _write_stage_b_runs_csv(tab_dir, runs)
    _write_reference_csv(tab_dir, reference)
    _rows_csv(tab_dir / "delta_vs_reference.csv", deltas)
    _rows_csv(tab_dir / "stage_b_tests_by_arm.csv", b_tests)
    _rows_csv(tab_dir / "stage_b_battery_by_epoch.csv", battery)
    _rows_csv(tab_dir / "prompting_stage_a.csv", prompt_a)
    _rows_csv(tab_dir / "prompting_stage_b_L3.csv", prompt_b)
    _rows_csv(tab_dir / "retest_pairs.csv", retest)
    _rows_csv(tab_dir / "contamination_by_epoch.csv", pilots)
    _rows_csv(tab_dir / "tokens_cost.csv", tokens)
    _rows_csv(tab_dir / "measurement_status.csv", status)
    _rows_csv(tab_dir / "pipeline_flash_lite.csv", pipeline)

    figures: list[tuple[str, str, str]] = []
    figures += _stage_a_figures(a_cells, status, fig_dir)
    figures += _stage_b_view_figures(b_cells, reference, fig_dir)
    figures += _delta_figures(deltas, b_cells, fig_dir)
    figures += _prompting_figures(prompt_a, a_cells, fig_dir)
    figures.append(
        (
            fig_retest(retest, fig_dir),
            "fig:retest",
            "Measurement noise: write throughput of byte-identical artefacts measured in two epochs (E6 against E10, E7 against E11). "
            "These are gemini-3.5-flash-lite pipeline-development runs, used here only as evidence of measurement noise, not as model "
            "evidence. The band marks a ratio between 0.5 and 1.5.",
        )
    )
    figures += _spearman_figures(b_tests, b_cells, counts["any"], fig_dir)
    artefact_rows = failure_artefact_rows(runs)
    _rows_csv(tab_dir / "stage_a_failure_artefacts.csv", artefact_rows)
    figures.append(
        (
            fig_failure_artefacts(artefact_rows, fig_dir),
            "fig:failure-artefacts",
            "Stage A outcome of the 10 runs of every main-matrix cell, coloured by the artefact that caused the failure: the "
            "feature file, the binding between feature and step code, the step code (solid: syntax or load error, hatched: runtime "
            "error), the API contract of the facade, the workload, or no artefact at all. Only Stage-A successes enter Stage B; "
            "runs with a non-conformant workload were executed and measured but are excluded from Stage B because they did not run "
            "the specified workload. Rows in the order of \\cref{fig:stage-a-matrix}.",
        )
    )
    figures += _stage_b_run_figures(b_cells, reference, fig_dir)
    flash_pairs = flash_identical_pairs(runs, config.paths.project_root)
    if flash_pairs:
        _rows_csv(tab_dir / "repeatability_flash_pairs.csv", flash_pairs)
    figures.append(
        (
            fig_repeatability(reference, flash_pairs, fig_dir),
            "fig:repeatability",
            "Measurement noise without a model in the loop. Left: the reference benchmark, whose step code is byte-identical in "
            "every run of a dialect, measured ten times per series on 13 Sep and on 9 Oct 2026 (write and read throughput; median "
            "as a line, outliers beyond 1.5 IQR ringed in red). Right: byte-identical gemini-3.5-flash artefacts measured on two "
            f"different days ({len([pair for pair in flash_pairs if pair['write_ratio']])} pairs), write throughput of the second against "
            "the first measurement; the band marks a ratio of 0.8 to 1.25.",
        )
    )
    figures += _pipeline_figures(a_cells, fig_dir)

    ratios = [row["tp_ratio"] for row in retest if row["identical_artifact"] and row["tp_ratio"]]
    retest_range = f"{min(ratios):.2f}--{max(ratios):.2f}"
    (tex_dir / "tab_dataset.tex").write_text(tex_dataset(dataset), encoding="utf-8")
    (tex_dir / "tab_stage_a.tex").write_text(tex_stage_a(a_cells, a_tests), encoding="utf-8")
    (tex_dir / "tab_failure_classes.tex").write_text(tex_failure_classes(a_cells), encoding="utf-8")
    (tex_dir / "tab_feedback.tex").write_text(tex_feedback(a_cells), encoding="utf-8")
    for op in OPS:
        (tex_dir / f"tab_stage_b_{op}.tex").write_text(
            tex_stage_b(b_cells, reference, op, retest_range, stage_b_missing[op]), encoding="utf-8"
        )
    (tex_dir / "tab_reference.tex").write_text(tex_reference(reference), encoding="utf-8")
    (tex_dir / "tab_delta.tex").write_text(tex_delta(deltas, b_cells), encoding="utf-8")
    (tex_dir / "tab_stage_b_tests.tex").write_text(tex_stage_b_tests(b_tests), encoding="utf-8")
    (tex_dir / "tab_prompting.tex").write_text(tex_prompting(prompt_a, prompt_b), encoding="utf-8")
    (tex_dir / "tab_retest.tex").write_text(tex_retest(retest, reference), encoding="utf-8")
    (tex_dir / "tab_contamination.tex").write_text(tex_pilot(pilots), encoding="utf-8")
    (tex_dir / "tab_tokens.tex").write_text(tex_tokens(tokens), encoding="utf-8")
    (tex_dir / "tab_pipeline.tex").write_text(tex_pipeline(a_cells), encoding="utf-8")
    (tex_dir / "figures.tex").write_text(tex_figures(figures), encoding="utf-8")

    summary = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs_loaded": len(runs),
        "main_models": [MODEL_FULL[model] for model in MAIN_MODELS],
        "arms": [{k: v for k, v in arm.items() if k != "short"} for arm in ARMS],
        "pipeline_arms": [
            {k: v for k, v in arm.items() if k != "short"}
            for arm in PIPELINE_ARMS + SYNTAX_ONLY_ARMS
        ],
        "stage_b_views": {
            "note": "distinct main-matrix runs with a Stage-B value; view all = measured, view success = measured minus Stage-A failed",
            **{
                op: {"all": n_all, "stage_a_failed": n_failed, "success": n_all - n_failed}
                for op, (n_all, n_failed) in counts.items()
            },
            "l1_measured_values": l1_measured,
            "stage_b_failures_per_op": stage_b_missing,
        },
        "stage_a": {
            f"{arm['key']}/{lvl}": f"{a_cells[(arm['key'], lvl)]['success']}/{a_cells[(arm['key'], lvl)]['n']}"
            for arm in ARMS
            for lvl in LEVELS
        },
        "pipeline_stage_a": {
            f"{arm['key']}/E{arm['epoch']}/{lvl}": f"{a_cells[(arm['key'], lvl)]['success']}/{a_cells[(arm['key'], lvl)]['n']}"
            for arm in PIPELINE_ARMS + SYNTAX_ONLY_ARMS
            for lvl in LEVELS
        },
        "stage_a_tests": a_tests,
        "reference": {
            series: {
                op: {k: v for k, v in stats.items() if k not in ("tp", "p50", "evaluated")}
                for op, stats in ops.items()
            }
            for series, ops in reference.items()
        },
        "delta_same_day": [row for row in deltas if row["same_day_as_reference"]],
        "stage_b_tests": b_tests,
        "prompting_stage_a": prompt_a,
        "prompting_stage_b_L3": prompt_b,
        "retest_flash_lite_measurement_noise": {
            f"E{first}->E{second}": {
                "pairs": sum(1 for row in retest if row["pair"] == f"E{first}->E{second}"),
                "identical": sum(
                    1
                    for row in retest
                    if row["pair"] == f"E{first}->E{second}" and row["identical_artifact"]
                ),
                "identical_by_level": dict(
                    Counter(
                        row["level"]
                        for row in retest
                        if row["pair"] == f"E{first}->E{second}" and row["identical_artifact"]
                    )
                ),
                "ratios": [
                    round(row["tp_ratio"], 3)
                    for row in retest
                    if row["pair"] == f"E{first}->E{second}"
                    and row["identical_artifact"]
                    and row["tp_ratio"]
                ],
            }
            for first, second, _ in RETEST_PAIRS
        },
        "contamination": pilots,
        "tokens": tokens,
        "dataset": dataset,
        "figures": [figure[0] for figure in figures],
        "removed_stale_outputs": removed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_readme(out, figures, summary)
    n_csv = len(list(tab_dir.glob("*.csv")))
    n_tex = len(list(tex_dir.glob("*.tex")))
    print(f"Written {len(figures)} figures, {n_csv} CSV tables, {n_tex} LaTeX files -> {out}")
    if removed:
        print("Removed stale outputs: " + ", ".join(removed))
    _print_headlines(a_cells, a_tests, reference, deltas, b_tests, prompt_a, prompt_b, retest)
    print("\n== Stage-B views (main matrix, distinct runs) ==")
    for op, (n_all, n_failed) in counts.items():
        print(
            f"{op:6s} all measured {n_all} (Stage-A failed {n_failed}) | Stage-A successes {n_all - n_failed}"
        )
    print(f"L1 measured values: {l1_measured}; Stage-B failures per op: {stage_b_missing}")


def _print_headlines(
    a_cells: dict[tuple[str, str], dict[str, Any]],
    a_tests: Sequence[dict[str, Any]],
    reference: dict[str, dict[str, dict[str, Any]]],
    deltas: Sequence[dict[str, Any]],
    b_tests: Sequence[dict[str, Any]],
    prompt_a: Sequence[dict[str, Any]],
    prompt_b: Sequence[dict[str, Any]],
    retest: Sequence[dict[str, Any]],
) -> None:
    print("\n== Stage A per arm (main matrix) ==")
    for arm in ARMS:
        cells = " ".join(
            f"{lvl} {a_cells[(arm['key'], lvl)]['success']}/{a_cells[(arm['key'], lvl)]['n']}"
            for lvl in LEVELS
        )
        stage_b_counts = " ".join(
            str(a_cells[(arm["key"], lvl)]["stage_b_write"]) for lvl in LEVELS
        )
        print(f"{arm['label']:40s} {cells}   stageB-n {stage_b_counts}")
    print("\n== Stage A pipeline development (flash-lite, not model evidence) ==")
    for arm in PIPELINE_ARMS + SYNTAX_ONLY_ARMS:
        cells = " ".join(
            f"{lvl} {a_cells[(arm['key'], lvl)]['success']}/{a_cells[(arm['key'], lvl)]['n']}"
            for lvl in LEVELS
        )
        print(f"{arm['label']:40s} {cells}")
    print("\n== Stage A tests per arm ==")
    for test_row in a_tests:
        print(
            f"{test_row['arm']:12s} L1vsRest {fmt_p_plain(test_row['fisher_control_p'])}  perm {fmt_p_plain(test_row['permutation_p'])} (V={fmt(test_row['permutation_V'], 2)})  "
            f"L2vsL3 {fmt_p_plain(test_row['fisher_L2_vs_L3_holm'])} L2vsL4 {fmt_p_plain(test_row['fisher_L2_vs_L4_holm'])} L2vsL5 {fmt_p_plain(test_row['fisher_L2_vs_L5_holm'])}"
        )
    print("\n== Reference ==")
    for series, ops in reference.items():
        for op, stats in ops.items():
            print(
                f"{series:12s} {op:6s} tp {stats['tp_med']:.1f} ({stats['tp_q1']:.1f}-{stats['tp_q3']:.1f}, range {stats['tp_min']:.1f}-{stats['tp_max']:.1f}) p50 {stats['p50_med']:.1f}"
            )
    print("\n== Delta same day (all measured | Stage-A successes) ==")
    for row in deltas:
        if row["same_day_as_reference"]:
            succ_tp = row["delta_tp_med_success"]
            succ_p50 = row["delta_p50_med_success"]
            print(
                f"{row['arm']:11s} {row['level']} {row['op']:6s} n={row['n']:2d} (failA {row['n_stage_a_failed']}) "
                f"Dtp {row['delta_tp_med']:+.0f}% Dp50 {row['delta_p50_med']:+.0f}% | "
                f"Dtp {'' if succ_tp is None else f'{succ_tp:+.0f}%'} Dp50 {'' if succ_p50 is None else f'{succ_p50:+.0f}%'}"
            )
    print("\n== Stage B tests per arm (all | success) ==")
    for row in b_tests:
        print(
            f"{row['arm']:11s} {row['op']:6s} n={row['n']:2d} rho={fmt(row['rho'], 3)} p={fmt_p_plain(row['rho_p'])} holm_all={fmt_p_plain(row['rho_holm_all'])} | "
            f"n={row['n_success']:2d} rho={fmt(row['rho_success'], 3)} p={fmt_p_plain(row['rho_success_p'])} holm_all={fmt_p_plain(row['rho_success_holm_all'])}"
        )
    print("\n== Prompting Stage A (Fisher per level, Holm) ==")
    for row in prompt_a:
        print(
            f"{MODEL_LABEL[row['model']]:6s} {row['dialect']:8s} {row['level']} zs {row['zs_success']}/{row['zs_n']} ms {row['ms_success']}/{row['ms_n']} p={fmt_p_plain(row['fisher_p'])} holm={fmt_p_plain(row['fisher_holm'])}"
        )
    print("\n== Prompting Stage B L3 within epoch (MWU; all | success) ==")
    for row in prompt_b:
        print(
            f"E{row['epoch']} {MODEL_LABEL[row['model']]:6s} {row['op']:6s} zs n={row['zs_n']} med={fmt(row['zs_median'])} ms n={row['ms_n']} med={fmt(row['ms_median'])} p={fmt_p_plain(row['mwu_p'])} | "
            f"n={row['zs_n_success']}/{row['ms_n_success']} p={fmt_p_plain(row['mwu_p_success'])}"
        )
    print("\n== Retest (flash-lite pipeline runs, measurement noise only) ==")
    for first, second, dialect in RETEST_PAIRS:
        pair_rows = [row for row in retest if row["pair"] == f"E{first}->E{second}"]
        identical = [row for row in pair_rows if row["identical_artifact"]]
        ratios = [row["tp_ratio"] for row in identical if row["tp_ratio"]]
        q1, med, q3 = quart(ratios)
        print(
            f"E{first}->E{second} {dialect}: pairs {len(pair_rows)}, identical {len(identical)}, both measured {len(ratios)}, ratio med {med:.2f} IQR {q1:.2f}-{q3:.2f} range {min(ratios):.2f}-{max(ratios):.2f}"
        )


def _write_readme(
    out: Path, figures: Sequence[tuple[str, str, str]], summary: dict[str, Any]
) -> None:
    views = summary["stage_b_views"]
    lines = [
        "# Results-Paket für die Thesis (Kapitel 4.3–4.5)",
        "",
        f"Erzeugt {summary['generated_at']} mit `uv run python scripts/thesis_results.py`. Quelle: `results/raw/metrics.csv` + "
        "`results/artifacts/**` über `evaluation/runs.py` (derselbe Collector wie das Dashboard), Stufe-A/B-Regeln aus "
        "`evaluation/runs.py` und `evaluation/statistics.py`, Referenz über `dashboard/build.py:ceiling_summary`. "
        f"{summary['runs_loaded']} ausgewertete Runs geladen (Quarantäne ausgeschlossen).",
        "",
        "## Design",
        "",
        "- **Modellvergleich nur mit EINEM Gemini-Modell (gemini-3.5-flash).** Hauptmatrix = 3 Modelle × 2 Prompting-Strategien ×",
        "  2 Dialekte = **12 Arme**, N = 10 je Level, eine Epoche je Arm, Epochen werden nie gepoolt:",
        "  flash zs Kestrel E8 · flash zs Tessera E9 · flash ms Kestrel E13 · flash ms Tessera E12 ·",
        "  haiku (claude-haiku-5-5, Claude-Code-CLI, Abo, kein Seed) zs + ms Kestrel E14 / Tessera E15 ·",
        "  glm (glm-5.3, OpenRouter → Z.AI, reasoning low) zs + ms Kestrel E16 / Tessera E17.",
        "  Bei haiku und glm laufen zero- und multi-shot interleaved in derselben Epoche; bei flash liegen sie in zwei Epochen",
        "  (zs 28.08., ms 13.09.2026).",
        "- **gemini-3.5-flash-lite = nur Pipeline-Entwicklung** (billig die Pipeline testen). Seine Runs (E10, E11, die",
        "  flash-lite-Hälfte von E12/E13 sowie E6/E7 (nur Syntax-Feedback)) gehen in KEINE Modell-Abbildung, -Tabelle, -Rangfolge",
        "  oder -Test ein. Sie stehen nur in den eigens benannten `pipeline_*`-Ausgaben (Stufe-A-Matrix, Feedback-Regime E6/E7 vs.",
        "  E10/E11, `pipeline_flash_lite.csv`, `tab_pipeline.tex`) und als Messrausch-Evidenz (identische Artefakte E6→E10, E7→E11,",
        "  `stage_b_test_retest_noise`, `tab_retest.tex`). E12/E13 enthalten beide Modelle interleaved; ausgewertet wird immer getrennt",
        "  (Kontaminationstabelle: E12/E13 in flash- und flash-lite-Zeile geteilt; `tab_dataset.tex` mit Rollen-Spalte).",
        "- E1–E5 (Piloten, ein Request je Operation) erscheinen nur in der Kontaminationstabelle.",
        "- Referenz (bddbench-Fixture, je 10 Läufe) an zwei Messtagen: 13.09. (E12 Tessera, E13 Kestrel) und 09.10. (E16 Kestrel,",
        "  E17 Tessera). Jeder Arm wird gegen die Referenz seines Messtags gestellt (haiku E14 → Gold E16, haiku E15 → Gold E17);",
        "  flash zs E8/E9 (28.08.) haben keine Referenz vom selben Tag und stehen nur in `delta_vs_reference.csv`",
        "  mit `same_day_as_reference=False`.",
        "- **Stufe B in zwei expliziten Sichten, jede als eigene Abbildung.** `<name>_all` = alle gemessenen Runs (jeder Run, der",
        "  ausgeführt wurde und einen Wert für die Operation hat; Stufe-A-Fehlschläge bleiben drin und sind markiert: × in",
        "  Modellfarbe, † in Δ-Abbildungen und Tabellen); `<name>_success` = nur Stufe-A-Erfolge, als vollständige eigene Abbildung.",
        "  Jede Abbildung trägt oben links eine Sicht-Zeile mit den Zahlen der gezeigten Runs.",
        f"  Hauptmatrix (write): {views['write']['all']} gemessene Runs, davon {views['write']['stage_a_failed']} Stufe-A-Fehlschläge "
        f"(glm, nonconformant: 2000/3000 statt 1000 Punkte geschrieben) → {views['write']['success']} Stufe-A-Erfolge. "
        f"Stufe-B-Fehlschläge (Stufe-A-Erfolg ohne Wert): {views['stage_b_failures_per_op']}.",
        "  Tabellen/CSVs geben beide Sichten nebeneinander (`stage_b_cells.csv`: `*_all` / `*_success`; `delta_vs_reference.csv`:",
        '  Spalten ohne / mit Suffix `_success`; `tab_stage_b_<op>.tex`: Zusatzzeile "successes only" wo die Sichten abweichen;',
        "  `tab_delta.tex` und `tab_stage_b_tests.tex`: je eine Tabelle pro Sicht).",
        "- **Keine primäre Metrik.** Throughput, Latenz p50, Δ zur Referenz, die Tests (und in den Zusatzskripten TOST, Client-Modus)",
        "  sind gleichrangige Indikatoren und werden zusammen gelesen.",
        "- Delete ist ein einzelner Request je Run und bimodal (≈ 25 vs. ≈ 75 ops/s) in zeitlichen Phasen: Zustandseffekt der",
        "  InfluxDB, nicht des Clients; Delete-Δ nur mit Vorsicht deuten.",
        "- Stufe-B-Tests: je Arm × Operation Kruskal-Wallis (L2–L5) + Spearman, Holm innerhalb Arm × Operation (m = 2), je Sicht;",
        "  zusätzlich Holm über alle 36 Spearman-Tests einer Sicht (`rho_holm_all`, `rho_success_holm_all`). Explorativ, nicht",
        "  vorab festgelegt. Harness-Batterie je Hauptmatrix-Epoche nur über Hauptmodell-Runs (`stage_b_battery_by_epoch.csv`;",
        "  E12/E13 ohne flash-lite, dort also nicht identisch mit dem Dashboard; E14–E17 poolen dort zero- und multi-shot, E8/E9",
        "  enthalten zusätzlich die 10 flash-multi-shot-L3-Runs der Epoche, die zu keinem Hauptmatrix-Arm gehören).",
        "- Prompting-Vergleich Stufe A: Fisher exakt je Level, Holm über die fünf Level einer Zeile (3 Modelle × 2 Dialekte). Stufe B:",
        "  Mann-Whitney auf dem Throughput (write/read/delete) von L3 innerhalb einer Epoche (E8/E9 flash, E14–E17 haiku/glm), beide",
        "  Sichten, unkorrigiert. Die flash-multi-shot-L3-Runs in E8/E9 (je 10) gehören zu keinem Hauptmatrix-Arm (flash ms = E12/E13).",
        "- Forschungsfragen unverändert: RQ1 (L1 ohne Guidance lauffähig?), RQ2a (Level für verlässliche Ausführung), RQ2b (Level, ab",
        "  dem die Performance an die Referenz heranreicht), RQ3 (Einfluss von LLM und Prompting).",
        "- Modellübergreifende Zusatzsichten (Token/Preis, L1-Probing): `scripts/thesis_results_models.py` → `models/`.",
        "",
        "## Abbildungen (`fig/`, PDF für LaTeX + PNG zur Ansicht)",
        "",
    ]
    for name, label, caption in figures:
        lines.append(f"- `{name}.pdf` (`{label}`): {caption}")
    lines += [
        "",
        "## Tabellen (`tables/*.csv`, Zahlen hinter jeder Abbildung) und LaTeX (`latex/*.tex`)",
        "",
        "- `dataset_epochs.csv` / `tab_dataset.tex`: Epochen-Übersicht mit Rolle (main / pipeline development (flash-lite) / pilot /",
        "  Kontrast), Datum, Modelle, Dialekt, Prompting, Regime, Runs, Kosten.",
        "- `stage_a_cells.csv` / `tab_stage_a.tex`: Hauptmatrix: Erfolge je Arm × Level, Funnel, Erstentwurf/Repariert, Fehlerklassen.",
        "- `stage_a_tests.csv`: Fisher L1 vs. Rest, Permutationstest L2–L5, paarweise Fisher vs. L2 (Holm) je Arm.",
        "- `tab_failure_classes.tex`: Fehlerklassen je Level × Arm (12 Spalten).",
        "- `measurement_status.csv`: wo jeder Hauptmatrix-Run die Pipeline verlässt.",
        "- `tab_feedback.tex`: Feedback-Effekt L4/L5 der Hauptmatrix (ohne E6/E7).",
        "- `stage_b_cells.csv` / `tab_stage_b_{write,read,delete}.tex`: Median (IQR) Throughput, p50 je Arm × Level × Operation, beide Sichten.",
        "- `stage_b_runs.csv`: jeder Hauptmatrix-Run mit allen Stufe-B-Werten, `in_view_all` / `in_view_success`.",
        "- `reference_runs.csv` / `tab_reference.tex`: Referenzverteilung je Dialekt × Operation.",
        "- `delta_vs_reference.csv` / `tab_delta.tex`: Δ Throughput / Δ p50 je Bedingung, beide Sichten (inkl. Quartile).",
        "- `stage_b_tests_by_arm.csv` / `tab_stage_b_tests.tex`: Kruskal-Wallis + Spearman je Arm × Operation, beide Sichten (Holm).",
        "- `stage_b_battery_by_epoch.csv`: Harness-Batterie je Hauptmatrix-Epoche (nur Hauptmodell-Runs).",
        "- `prompting_stage_a.csv`, `prompting_stage_b_L3.csv` / `tab_prompting.tex`: zero- vs. multi-shot (3 Modelle).",
        "- `retest_pairs.csv` / `tab_retest.tex`: identische Artefakte E6→E10, E7→E11 (flash-lite-Pipeline-Runs, nur Messrauschen).",
        "- `contamination_by_epoch.csv` / `tab_contamination.tex`: Piloten + Kontaminationsindikatoren je Epoche (E12/E13 geteilt).",
        "- `tokens_cost.csv` / `tab_tokens.tex`: Prompt-Tokens, Kosten je Run und je Erfolg (Hauptmatrix).",
        "- `pipeline_flash_lite.csv` / `tab_pipeline.tex`: Pipeline-Entwicklung (flash-lite): Stufe A, Feedback, Stufe-B-Zählungen.",
        "- `latex/figures.tex`: alle Figure-Floats mit Entwurfs-Captions; `summary.json`: alle Kennzahlen maschinenlesbar.",
        "",
        "Entfernt (ersetzt): `stage_a_model_comparison` (flash vs. flash-lite), `model_stage_{a,b}.csv`, `tab_model.tex`,",
        "`stage_b_<op>_<dialekt>` und `..._haiku_glm` (→ `stage_b_<op>_<dialekt>_{all,success}`), `stage_b_delta_vs_reference_<dialekt>`",
        "(→ `_{all,success}`), `stage_b_spearman_by_arm` (→ `_{all,success}`).",
        "",
        "## Einbau in die Thesis",
        "",
        "1. `fig/*.pdf` nach `thesis/fig/` kopieren (die Floats verweisen auf `fig/<name>.pdf`).",
        "2. Tabellen: `\\input{...}` oder Inhalt der `latex/tab_*.tex` in die Abschnittsdateien einfügen; `booktabs`, `graphicx`, `float`,",
        "   `amsmath` und `cleveref` sind in `thesis.tex` bereits geladen. `tab_failure_classes.tex` nutzt `\\rotatebox` (graphicx).",
        "3. Captions sind Entwürfe und sollen in der Stimme des Autors umgeschrieben werden.",
        "",
        "Gestaltung: Dataviz-Referenzpalette (Blau #2a78d6 = gemini-3.5-flash, Grün #1d9a6c = claude-haiku-5-5, Violett #8a5cd1 =",
        "glm-5.3; Orange #eb6834 = gemini-3.5-flash-lite nur in Pipeline-Ausgaben; Kreis = Kestrel, Dreieck = Tessera; in den",
        "Δ-Abbildungen Kreis = multi-shot, Quadrat = zero-shot; × = gemessener Run mit Stufe-A-Fehlschlag; ordinale Blau-Rampe für",
        "Fehlerklassen nach Pipeline-Stufe), Arial, Textbreite 15 cm, Vektor-PDF.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
