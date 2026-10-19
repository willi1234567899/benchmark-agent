from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Protocol, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_results as tr
from thesis_results import (
    BASELINE,
    GRID,
    INK,
    INK_2,
    LEVELS,
    MODEL_COLOR,
    MODEL_FULL,
    MODEL_LABEL,
    MODELS,
    MUTED,
    PROMPT_MARKER,
    RAMP_5,
    SURFACE,
    TEXT_WIDTH_IN,
    fmt_p,
    holm,
    latex_table,
    save,
    tex,
    write_csv,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    cast(TextIOWrapper, sys.stdout).reconfigure(encoding="utf-8")

PROMPTINGS: tuple[str, ...] = ("zero_shot", "multi_shot")
DIALECTS: tuple[str, ...] = ("Kestrel", "Tessera")
PROMPT_SHORT = {"zero_shot": "zs", "multi_shot": "ms"}
REGIME_PAIRS: tuple[tuple[str, str, int, int], ...] = (
    ("lite-zs-K-syntax", "lite-zs-K", 6, 10),
    ("lite-zs-T-syntax", "lite-zs-T", 7, 11),
)
MODEL_PAIRS: tuple[tuple[str, str], ...] = (
    ("gemini", "claude_haiku"),
    ("gemini", "glm"),
    ("claude_haiku", "glm"),
)
N_MAIN_ARMS = len(tr.ARMS)
PIPELINE = "pipeline development (flash-lite)"
LADDER_STEPS: tuple[tuple[str, str], ...] = (
    ("L2", "L2"),
    ("L3", "L3"),
    ("L4 first", "L4\nfirst"),
    ("L5 first", "L5\nfirst"),
    ("L4 final", "L4\nfinal"),
    ("L5 final", "L5\nfinal"),
)
LADDER_TABLE_STEPS: tuple[tuple[str, str], ...] = (("L1", "L1"), *LADDER_STEPS)
LADDER_Y_LO = 44.0
LADDER_Y_HI = 109.0
REPAIR_KINDS: tuple[tuple[str, str], ...] = (
    ("first_ok", "first draft accepted"),
    ("validator_only", "repaired, validator rounds only"),
    ("execution", "repaired, ≥1 execution round"),
    ("failed", "failed"),
)
REPAIR_COLOR = {
    "first_ok": GRID,
    "validator_only": RAMP_5[1],
    "execution": RAMP_5[3],
    "failed": INK_2,
}


TIME_WINDOW = re.compile(r"invalid_(since|until|from|to|window_from|window_to)|missing_since")
DECORATOR = re.compile(r"^\s*@(given|when|then|step)\(")
CALL_LINE = re.compile(r"^[A-Za-z_]\w*\(context")
CONTEXT_CONFIG = re.compile(r"^\s*context\.config\s*=", re.M)

SUBCLASSES: tuple[tuple[str, str, str], ...] = (
    (
        "json_contract",
        "JSON tool-contract violation",
        "generation\\_failed; the model output violated the JSON contract, no artefact",
    ),
    (
        "no_artefact_other",
        "No artefact, other",
        "generation\\_failed without a contract message (empty response)",
    ),
    (
        "guessed_schema",
        "Guessed request schema",
        "api\\_rejected without influx\\_* or time-window code (unknown\\_field\\_*, invalid\\_record, missing\\_*, non\\_json\\_body)",
    ),
    (
        "weak_canary",
        "Weak canary (start/limit)",
        "api\\_rejected with influx\\_field\\_start or influx\\_field\\_limit",
    ),
    (
        "time_window",
        "Time-window literal",
        "rejection code invalid\\_since\\slash\\_from\\slash\\_until\\slash\\_window\\_from or missing\\_since",
    ),
    (
        "quoted_literal",
        "Quoted literal in URL",
        "quoted\\_literal: Gherkin quotes reached the URL path",
    ),
    (
        "quote_collision",
        "Quote collision in decorator",
        "`unterminated string' or `part of the string?' compile error on a step-decorator line (nested quotes)",
    ),
    (
        "missing_def",
        "Missing \\texttt{def}",
        "compile error on a call-like line directly below a step decorator",
    ),
    (
        "escape",
        "Escape artefact",
        "`unexpected character after line continuation character' (stray backslash, literal \\textbackslash n)",
    ),
    (
        "python_other",
        "Python syntax, other",
        "other compile errors: brackets, f-strings, indentation, numeric literals",
    ),
    (
        "behave_import",
        "behave ImportError",
        "steps module fails to load: \\texttt{from behave import background}",
    ),
    (
        "regex_anchor",
        "Regex step anchors",
        "\\texttt{use\\_step\\_matcher(`re')} with \\^{}/\\$ anchors (ValueError on load)",
    ),
    ("gherkin_parse", "Gherkin parse error", "feature file could not be parsed (E001)"),
    ("undefined_step", "Undefined step", "step(s) without a matching step definition"),
    (
        "reseeding",
        "Per-scenario re-seeding",
        "executed, nonconformant write/points (more than 1000 points written)",
    ),
    (
        "nonconf_other",
        "Nonconformant, other",
        "executed, other failed workload checks (read/labels, delete/labels)",
    ),
    (
        "context_config",
        "\\texttt{context.config} trap",
        "runtime exception; the draft assigns behave's reserved \\texttt{context.config}",
    ),
    (
        "runtime_other",
        "Runtime error, other",
        "step exception or failed scenario without \\texttt{context.config}",
    ),
)
SUBCLASS_LABEL = {key: label for key, label, _ in SUBCLASSES}
PREFERRED_EXAMPLE = {
    "time_window": "L3_gemini_lite_zeroshot/run-025",
    "reseeding": "L3_glm_zeroshot/run-001",
    "weak_canary": "L1_claude_haiku_zeroshot",
    "json_contract": "L1_glm_zeroshot",
}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(sps.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(sps.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


Table2x2 = tuple[int, int, int, int]


def strat_exact(tables: Sequence[Table2x2]) -> dict[str, float]:
    distribution = np.array([1.0])
    offset = 0
    observed = 0
    mh_numerator = mh_denominator = 0.0
    for a, b, c, d in tables:
        n = a + b + c + d
        if n == 0:
            continue
        n1, m = a + b, a + c
        lo, hi = max(0, m - (n - n1)), min(n1, m)
        pmf = sps.hypergeom.pmf(np.arange(lo, hi + 1), n, m, n1)
        distribution = np.convolve(distribution, pmf)
        offset += lo
        observed += a
        mh_numerator += a * d / n
        mh_denominator += b * c / n
    p_obs = distribution[observed - offset]
    p_two = float(distribution[distribution <= p_obs * (1 + 1e-7)].sum())
    p_greater = float(distribution[observed - offset :].sum())
    p_less = float(distribution[: observed - offset + 1].sum())
    if mh_denominator > 0:
        odds = mh_numerator / mh_denominator
    elif mh_numerator > 0:
        odds = math.inf
    else:
        odds = float("nan")
    return {
        "or_mh": odds,
        "p": min(1.0, p_two),
        "p_greater": min(1.0, p_greater),
        "p_less": min(1.0, p_less),
    }


def mcnemar_exact(b: int, c: int) -> float | None:
    if b + c == 0:
        return None
    return float(sps.binomtest(b, b + c, 0.5).pvalue)


def fmt_or(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if math.isinf(x):
        return "$\\infty$"
    return f"{x:.2f}"


def fmt_or_plain(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if math.isinf(x):
        return "inf"
    return f"{x:.3f}"


def fmt_p_sci(p: float | None) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p < 0.001:
        mant, exp = f"{p:.1e}".split("e")
        return f"${mant}\\times10^{{{int(exp)}}}$"
    if p < 0.01:
        return f"{p:.4f}"
    return f"{p:.3f}"


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{100 * lo:.0f}, {100 * hi:.0f}]"


def run_dir(r: Any, root: Path) -> Path:
    path = Path(r.run_directory)
    return path if path.is_absolute() else root / path


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None


def artefact_hash(base: Path) -> str | None:
    feature, steps = base / "benchmark.feature", base / "steps.py"
    if not feature.exists() and not steps.exists():
        return None
    digest = hashlib.sha1()
    for path in (feature, steps):
        digest.update(path.read_bytes() if path.exists() else b"")
        digest.update(b"||")
    return digest.hexdigest()


def first_draft_dir(d: Path, feedback_iterations: int) -> Path | None:
    first_attempt = d / "attempts" / "attempt-00"
    if (first_attempt / "steps.py").exists() or (first_attempt / "benchmark.feature").exists():
        return first_attempt
    return d if feedback_iterations == 0 else None


def build_rows(runs: Sequence[Any], root: Path) -> list[dict[str, Any]]:
    arm_sets: list[tuple[dict[str, Any], str]] = [(arm, "main") for arm in tr.ARMS]
    arm_sets += [(arm, "pipeline") for arm in tr.PIPELINE_ARMS]
    arm_sets += [(arm, "syntax_only") for arm in tr.SYNTAX_ONLY_ARMS]
    for epoch, dialect in ((6, "Kestrel"), (7, "Tessera"), (10, "Kestrel"), (11, "Tessera")):
        arm_sets.append(
            (
                {
                    "key": f"lite-ms-{dialect[0]}-E{epoch}",
                    "model": "gemini_lite",
                    "prompting": "multi_shot",
                    "dialect": dialect,
                    "epoch": epoch,
                },
                "l3_control",
            )
        )
    rows: list[dict[str, Any]] = []
    for arm, group in arm_sets:
        for run in tr.arm_runs(runs, arm):
            if group == "l3_control" and run.guidance_level != "L3":
                continue
            directory = run_dir(run, root)
            feedback_iterations = int(run.feedback_iterations or 0)
            first_dir = first_draft_dir(directory, feedback_iterations)
            meta = read_json(directory / "metadata.json") or {}
            rounds = read_json(directory / "feedback_rounds.json") or []
            rows.append(
                {
                    "arm": arm["key"],
                    "group": group,
                    "model": arm["model"],
                    "prompting": arm["prompting"],
                    "dialect": arm["dialect"],
                    "epoch": int(arm["epoch"]),
                    "level": run.guidance_level,
                    "seed": run.seed,
                    "success": bool(run.success),
                    "fi": feedback_iterations,
                    "efr": int(run.execution_feedback_rounds or 0),
                    "first_ok": bool(run.success) and feedback_iterations == 0,
                    "fc": run.failure_class,
                    "fd": run.failure_detail or "",
                    "dir": directory,
                    "rel": f"{directory.parent.name}/{directory.name}",
                    "hash_final": artefact_hash(directory),
                    "hash_first": artefact_hash(first_dir) if first_dir else None,
                    "first_dir": first_dir,
                    "meta": meta,
                    "rounds": rounds,
                }
            )
    return rows


def cell(
    rows: Sequence[dict[str, Any]], arm: str, level: str, pred: Callable[[dict[str, Any]], bool]
) -> tuple[int, int]:
    members = [run for run in rows if run["arm"] == arm and run["level"] == level]
    return sum(1 for run in members if pred(run)), len(members)


def is_success(r: dict[str, Any]) -> bool:
    return bool(r["success"])


def is_first_ok(r: dict[str, Any]) -> bool:
    return bool(r["first_ok"])


def arm_key(model: str, prompting: str, dialect: str) -> str:
    return next(
        arm["key"]
        for arm in tr.ARMS
        if arm["model"] == model and arm["prompting"] == prompting and arm["dialect"] == dialect
    )


def ladder_value(rows: Sequence[dict[str, Any]], arm: str, step: str) -> tuple[int, int]:
    if step in LEVELS:
        return cell(rows, arm, step, is_success)
    level, kind = step.split()
    return cell(rows, arm, level, is_first_ok if kind == "first" else is_success)


def ladder_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in MODELS:
        arms = [arm for arm in tr.ARMS if arm["model"] == model]
        for step, _ in LADDER_TABLE_STEPS:
            per_arm = [ladder_value(rows, arm["key"], step) for arm in arms]
            k = sum(arm_k for arm_k, _ in per_arm)
            n = sum(arm_n for _, arm_n in per_arm)
            lo, hi = clopper_pearson(k, n)
            out.append(
                {
                    "model": model,
                    "step": step,
                    "successes": k,
                    "n": n,
                    "rate": k / n if n else None,
                    "cp_low": lo,
                    "cp_high": hi,
                    **{
                        f"arm_{arm['key']}": arm_k
                        for arm, (arm_k, _) in zip(arms, per_arm, strict=True)
                    },
                }
            )
    for step, _ in LADDER_TABLE_STEPS:
        model_rows = [row for row in out if row["step"] == step and row["model"] in MODELS]
        k, n = sum(row["successes"] for row in model_rows), sum(row["n"] for row in model_rows)
        lo, hi = clopper_pearson(k, n)
        out.append(
            {
                "model": f"all ({N_MAIN_ARMS} arms)",
                "step": step,
                "successes": k,
                "n": n,
                "rate": k / n if n else None,
                "cp_low": lo,
                "cp_high": hi,
            }
        )
    return out


def ladder_l1_note(lad: Sequence[dict[str, Any]]) -> str:
    l1 = [
        next(row for row in lad if row["model"] == model and row["step"] == "L1")
        for model in MODELS
    ]
    if all(row["successes"] == 0 for row in l1) and len({row["n"] for row in l1}) == 1:
        return f"L1: 0 of {l1[0]['n']} in every model (see L1 figure)"
    return (
        "L1: "
        + ", ".join(f"{MODEL_LABEL[row['model']]} {row['successes']} of {row['n']}" for row in l1)
        + " (see L1 figure)"
    )


def ladder_clipped(rows: Sequence[dict[str, Any]]) -> list[tuple[str, str, str, int]]:
    out = []
    for arm in tr.ARMS:
        for step, _ in LADDER_STEPS:
            value = ladder_value(rows, arm["key"], step)[0] * 10
            if value < LADDER_Y_LO:
                out.append((arm["model"], arm["key"], step, value))
    return out


def _axis_break(ax: Axes, y: float = 0.035) -> None:
    gap = 0.016
    ax.plot(
        [0, 0],
        [y - gap, y + gap],
        transform=ax.transAxes,
        color="white",
        lw=2.6,
        clip_on=False,
        zorder=9,
        solid_capstyle="butt",
    )
    for y_mark in (y - gap, y + gap):
        ax.plot(
            [0],
            [y_mark],
            transform=ax.transAxes,
            marker=[(-1, -0.55), (1, 0.55)],
            markersize=7,
            linestyle="none",
            color=INK_2,
            mec=INK_2,
            mew=0.9,
            clip_on=False,
            zorder=10,
        )


def fig_ladder(rows: Sequence[dict[str, Any]], lad: Sequence[dict[str, Any]], out: Path) -> str:
    fig, axes = plt.subplots(1, len(MODELS), figsize=(TEXT_WIDTH_IN, 2.75), sharey=True)
    xs = np.arange(len(LADDER_STEPS))
    offsets = (-0.2, -0.07, 0.07, 0.2)
    n_pre = sum(1 for step, _ in LADDER_STEPS if "final" not in step)
    y_floor = LADDER_Y_LO + 1.6
    for ax, model in zip(axes.flat, MODELS, strict=True):
        color = MODEL_COLOR[model]
        arms = [arm for arm in tr.ARMS if arm["model"] == model]
        for j, arm in enumerate(arms):
            arm_values = [ladder_value(rows, arm["key"], step)[0] * 10 for step, _ in LADDER_STEPS]
            shown = [max(value, y_floor) for value in arm_values]
            ax.scatter(
                xs + offsets[j],
                shown,
                s=11,
                marker=PROMPT_MARKER[arm["prompting"]],
                facecolor=color if arm["dialect"] == "Kestrel" else "white",
                edgecolor=color,
                linewidth=0.7,
                alpha=0.45,
                zorder=2,
                clip_on=False,
            )
            for x, value in zip(xs + offsets[j], arm_values, strict=True):
                if value < LADDER_Y_LO:
                    ax.text(
                        x + 0.09,
                        y_floor,
                        f"{value:.0f}",
                        fontsize=5.5,
                        color=INK_2,
                        ha="left",
                        va="center",
                    )
        totals = [
            next(row for row in lad if row["model"] == model and row["step"] == step)
            for step, _ in LADDER_STEPS
        ]
        rate = np.array([100 * total["rate"] for total in totals])
        lo = np.array([100 * total["cp_low"] for total in totals])
        hi = np.array([100 * total["cp_high"] for total in totals])
        for segment in (slice(0, n_pre), slice(n_pre, len(LADDER_STEPS))):
            ax.plot(xs[segment], rate[segment], color=color, lw=1.4, zorder=3)
        ax.errorbar(
            xs,
            rate,
            yerr=[rate - lo, hi - rate],
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=2,
            zorder=3,
        )
        ax.scatter(xs, rate, s=26, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=4)
        for x, total in zip(xs, totals, strict=True):
            ax.text(
                x,
                105.5,
                str(total["successes"]),
                fontsize=6.5,
                color=INK_2,
                ha="center",
                va="center",
            )
        ax.axvline(n_pre - 0.5, color=BASELINE, lw=0.8, ls=(0, (3, 2)))
        ax.set_title(f"{MODEL_FULL[model]}\n4 arms, 40 runs per step", fontsize=7.5)
        ax.set_ylim(LADDER_Y_LO, LADDER_Y_HI)
        ax.set_yticks([50, 60, 70, 80, 90, 100])
        ax.set_xlim(-0.5, len(LADDER_STEPS) - 0.5)
        ax.set_xticks(xs, [tick_label for _, tick_label in LADDER_STEPS])
        ax.tick_params(axis="x", length=0, labelsize=6.5)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(BASELINE)
        ax.spines["left"].set_bounds(LADDER_Y_LO, 100)
        _axis_break(ax)
        ax.text(
            n_pre - 0.4,
            LADDER_Y_LO + 9,
            "after\nfeedback",
            fontsize=6.0,
            color=MUTED,
            ha="left",
            va="bottom",
        )
    axes[0].set_ylabel("Stage A success (%)")
    handles = [
        Line2D(
            [],
            [],
            color=INK_2,
            marker="o",
            markersize=5,
            lw=1.4,
            label="model total, 95% Clopper-Pearson (numbers on top = successes of 40)",
        ),
        Line2D(
            [], [], color=INK_2, marker="s", markersize=4, lw=0, alpha=0.5, label="arm, zero-shot"
        ),
        Line2D(
            [], [], color=INK_2, marker="o", markersize=4, lw=0, alpha=0.5, label="arm, multi-shot"
        ),
        Line2D(
            [],
            [],
            color=INK_2,
            marker="o",
            markersize=4,
            lw=0,
            markerfacecolor="white",
            alpha=0.6,
            label="hollow = Tessera, filled = Kestrel",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.17),
        handlelength=1.6,
        columnspacing=1.6,
    )
    fig.subplots_adjust(wspace=0.08)
    note = ladder_l1_note(lad)
    if ladder_clipped(rows):
        note += "; arm values below the axis sit at the lower edge with their value"
    fig.text(0.5, -0.185, note, ha="center", va="top", fontsize=7, color=INK_2)
    return save(fig, out, "stage_a_prefeedback_ladder")


def repair_kind(r: dict[str, Any]) -> str:
    if not r["success"]:
        return "failed"
    if r["fi"] == 0:
        return "first_ok"
    return "execution" if r["efr"] > 0 else "validator_only"


def unchecked_final(r: dict[str, Any]) -> bool:
    if r["epoch"] < 10 or r["fi"] == 0:
        return False
    log = r["meta"].get("execution_feedback_log") or []
    final_attempt = r["meta"].get("attempt")
    return not any(entry.get("attempt") == final_attempt for entry in log)


RepairGroup = tuple[str, str, Callable[[dict[str, Any]], bool]]


def main_repair_groups() -> list[RepairGroup]:
    return [
        (
            MODEL_LABEL[model],
            model,
            lambda run, model=model: run["group"] == "main" and run["model"] == model,
        )
        for model in MODELS
    ]


def pipeline_repair_groups() -> list[RepairGroup]:
    exec_keys = tuple(exec_key for _, exec_key, _, _ in REGIME_PAIRS)
    return [
        (
            "flash-lite zs, E10/E11 (execution)",
            "contrast_exec",
            lambda run: run["group"] == "pipeline" and run["arm"] in exec_keys,
        ),
        (
            "flash-lite zs, E6/E7 (syntax only)",
            "contrast_syntax",
            lambda run: run["group"] == "syntax_only",
        ),
    ]


def repair_rows(
    rows: Sequence[dict[str, Any]], groups: Sequence[RepairGroup]
) -> list[dict[str, Any]]:
    out = []
    for label, key, pred in groups:
        for level in ("L4", "L5"):
            members = [run for run in rows if pred(run) and run["level"] == level]
            kinds = Counter(repair_kind(run) for run in members)
            execution_from_round_1 = sum(
                1
                for run in members
                if repair_kind(run) == "execution"
                and run["rounds"]
                and run["rounds"][0].get("kind") == "execution"
            )
            out.append(
                {
                    "role": PIPELINE if key.startswith("contrast") else tr.ROLE_MAIN,
                    "group": label,
                    "key": key,
                    "level": level,
                    "n": len(members),
                    **{kind: kinds.get(kind, 0) for kind, _ in REPAIR_KINDS},
                    "execution_from_round_1": execution_from_round_1,
                    "validator_then_execution": kinds.get("execution", 0) - execution_from_round_1,
                    "n_unchecked_final": sum(1 for run in members if unchecked_final(run)),
                    "rounds_repaired": sum(
                        run["fi"] for run in members if run["success"] and run["fi"] > 0
                    ),
                }
            )
    return out


class _CanvasWithRenderer(Protocol):
    def get_renderer(self) -> RendererBase: ...


def fig_repair_kind(
    rep: Sequence[dict[str, Any]], out: Path, name: str, header: str | None = None
) -> str:
    order = list(rep)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 0.42 * len(order) + 0.75))
    labels = []
    y_positions: list[float] = []
    y = 0.0
    prev_group = None
    for row in order:
        if prev_group is not None and row["group"] != prev_group:
            y += 0.4
        y_positions.append(y)
        labels.append(f"{row['group']} · {row['level']}")
        prev_group = row["group"]
        y += 1.0
    for y_row, row in zip(y_positions, order, strict=True):
        left = 0.0
        for key, _ in REPAIR_KINDS:
            share = row[key] / row["n"] * 100 if row["n"] else 0
            if row[key]:
                ax.barh(
                    y_row,
                    share,
                    left=left,
                    height=0.72,
                    color=REPAIR_COLOR[key],
                    edgecolor=SURFACE,
                    linewidth=1.0,
                )
                if share >= 2.4:
                    ax.text(
                        left + share / 2,
                        y_row,
                        str(row[key]),
                        ha="center",
                        va="center",
                        fontsize=6.5 if share >= 4.5 else 5.5,
                        color=INK if key == "first_ok" else "white",
                    )
                left += share
        ax.text(101.2, y_row, f"n = {row['n']}", va="center", ha="left", fontsize=6.5, color=INK_2)
    ax.set_yticks(y_positions, labels)
    ax.set_ylim(y_positions[-1] + 0.6, y_positions[0] - 0.6)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of runs (%); numbers in the bars are runs")
    ax.grid(False)
    ax.xaxis.grid(True)
    ax.tick_params(axis="y", length=0)
    handles = [
        Patch(facecolor=REPAIR_COLOR[kind], label=kind_label) for kind, kind_label in REPAIR_KINDS
    ]
    fig.canvas.draw()
    renderer = cast(_CanvasWithRenderer, fig.canvas).get_renderer()
    xlabel_box = ax.xaxis.label.get_window_extent(renderer)
    legend_y = fig.transFigure.inverted().transform((0, xlabel_box.y0))[1] - 0.02
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, legend_y),
        handlelength=1.2,
        columnspacing=1.2,
        frameon=False,
    )
    if header:
        tr._header(fig, header)
    return save(fig, out, name)


def onset(rows: Sequence[dict[str, Any]], arm: str) -> str:
    counts = [cell(rows, arm, level, is_success) for level in LEVELS]
    for i, level in enumerate(LEVELS):
        if all(k == n and n > 0 for k, n in counts[i:]):
            return level
    return "--"


def count_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    l2_vs_l3_p_values = []
    for arm in tr.ARMS:
        row: dict[str, Any] = {"arm": arm["key"], "model": arm["model"], "epoch": arm["epoch"]}
        for level in LEVELS:
            k, n = cell(rows, arm["key"], level, is_success)
            lo, hi = clopper_pearson(k, n)
            row |= {f"{level}_k": k, f"{level}_n": n, f"{level}_cp_low": lo, f"{level}_cp_high": hi}
        for level in ("L4", "L5"):
            row[f"{level}_first_k"] = cell(rows, arm["key"], level, is_first_ok)[0]
        k23 = row["L2_k"] + row["L3_k"]
        n23 = row["L2_n"] + row["L3_n"]
        row["L2L3_k"], row["L2L3_n"] = k23, n23
        row["L2L3_cp_low"], row["L2L3_cp_high"] = clopper_pearson(k23, n23)
        row["onset"] = onset(rows, arm["key"])
        row["fisher_L2_vs_L3_p"] = tr.fisher(row["L2_k"], row["L2_n"], row["L3_k"], row["L3_n"])
        l2_vs_l3_p_values.append(row["fisher_L2_vs_L3_p"])
        out.append(row)
    for row, p_holm in zip(out, holm(l2_vs_l3_p_values), strict=True):
        row["fisher_L2_vs_L3_holm"] = p_holm
    for model in MODELS:
        arm_rows = [arm_row for arm_row in out if arm_row["model"] == model]
        row = {"arm": f"model total {MODEL_LABEL[model]}", "model": model, "epoch": ""}
        for key in ("L1", "L2", "L3", "L4", "L5"):
            k = sum(arm_row[f"{key}_k"] for arm_row in arm_rows)
            n = sum(arm_row[f"{key}_n"] for arm_row in arm_rows)
            lo, hi = clopper_pearson(k, n)
            row |= {f"{key}_k": k, f"{key}_n": n, f"{key}_cp_low": lo, f"{key}_cp_high": hi}
        for level in ("L4", "L5"):
            row[f"{level}_first_k"] = sum(arm_row[f"{level}_first_k"] for arm_row in arm_rows)
        row["L2L3_k"] = sum(arm_row["L2L3_k"] for arm_row in arm_rows)
        row["L2L3_n"] = sum(arm_row["L2L3_n"] for arm_row in arm_rows)
        row["L2L3_cp_low"], row["L2L3_cp_high"] = clopper_pearson(row["L2L3_k"], row["L2L3_n"])
        row["onset"] = ""
        row["fisher_L2_vs_L3_p"] = None
        row["fisher_L2_vs_L3_holm"] = None
        out.append(row)
    return out


def tex_counts(cnt: Sequence[dict[str, Any]]) -> str:
    holm_values = [
        row["fisher_L2_vs_L3_holm"] for row in cnt if row["fisher_L2_vs_L3_holm"] is not None
    ]
    holm_min = min(holm_values)
    holm_txt = (
        f"Holm over the {N_MAIN_ARMS} arms gives $p_\\mathrm{{Holm}}=1$ for every arm."
        if holm_min >= 0.9995
        else f"Holm over the {N_MAIN_ARMS} arms: smallest $p_\\mathrm{{Holm}}={holm_min:.2f}$."
    )
    body = []
    midrule_rows = []
    for model in MODELS:
        for row in cnt:
            if row["model"] != model:
                continue
            total = str(row["arm"]).startswith("model total")
            if total:
                name = f"\\textit{{{MODEL_LABEL[model]}, 4 arms}}"
            else:
                name = tr.arm_short_tex(tr.ARM_BY_KEY[row["arm"]])
            cells = [str(row["L1_k"]), str(row["L2_k"]), str(row["L3_k"])]
            cells += [f"{row[f'{level}_k']} ({row[f'{level}_first_k']})" for level in ("L4", "L5")]
            l2_l3_cell = (
                f"{row['L2L3_k']}/{row['L2L3_n']} {fmt_ci(row['L2L3_cp_low'], row['L2L3_cp_high'])}"
            )
            body.append(
                [
                    name,
                    *cells,
                    row["onset"] or "",
                    l2_l3_cell,
                    fmt_p(row["fisher_L2_vs_L3_p"]) if not total else "",
                ]
            )
        midrule_rows.append(len(body) - 1)
    return latex_table(
        "Stage A successes per main arm and guidance level (N = 10 per cell, model rows N = 40), with the first-draft successes at L4/L5 in parentheses, "
        "the onset of reliable execution (first level from which every cell is 10/10), L2+L3 successes with a 95\\,\\% Clopper-Pearson interval in percent, "
        "and Fisher's exact test of L2 against L3 per arm.",
        "tab:stage-a-counts",
        [
            "Arm",
            "L1",
            "L2",
            "L3",
            "L4 (first)",
            "L5 (first)",
            "Onset",
            "L2+L3 [95\\,\\% CI]",
            "$p$ L2/L3",
        ],
        body,
        "lrrrrrlrr",
        note=tr.ARM_LEGEND
        + " Clopper-Pearson bounds for a single cell: 10/10 $\\rightarrow$ [69, 100]\\,\\%, 9/10 $\\rightarrow$ [55, 100]\\,\\%, 0/10 $\\rightarrow$ [0, 31]\\,\\%; for 40 runs: 40/40 $\\rightarrow$ [91, 100]\\,\\%. "
        "The onset is fragile: one failure moves it by up to two levels (haiku $\\cdot$ zs $\\cdot$ T: L2 10/10, L3 9/10, onset L4). "
        f"Fisher $p$ is uncorrected; {holm_txt} Model rows are descriptive sums over the four arms of a model (flash: four epochs, haiku and glm: two epochs each) and are not tested.",
        size=r"\scriptsize",
        midrules_after=tuple(midrule_rows[:-1]),
        tabcolsep="3pt",
    )


def test_row(
    family: str,
    comparison: str,
    tables: Sequence[Table2x2],
    strata: str,
    note: str = "",
) -> dict[str, Any]:
    result = strat_exact(tables)
    ka = sum(table[0] for table in tables)
    na = sum(table[0] + table[1] for table in tables)
    kb = sum(table[2] for table in tables)
    nb = sum(table[2] + table[3] for table in tables)
    return {
        "role": tr.ROLE_MAIN,
        "family": family,
        "comparison": comparison,
        "test": "exact stratified (hypergeometric convolution)",
        "strata": strata,
        "n_strata": len(tables),
        "a_k": ka,
        "a_n": na,
        "b_k": kb,
        "b_n": nb,
        "effect": "MH OR (A vs B)",
        "effect_value": result["or_mh"],
        "p": result["p"],
        "p_one_sided_A_greater": result["p_greater"],
        "p_holm": None,
        "note": note,
    }


def paired_row(
    family: str, comparison: str, b: int, c: int, n_pairs: int, a_k: int, b_k: int, note: str
) -> dict[str, Any]:
    p = mcnemar_exact(b, c)
    p_one = float(sps.binomtest(b, b + c, 0.5, alternative="greater").pvalue) if b + c else None
    return {
        "role": tr.ROLE_MAIN,
        "family": family,
        "comparison": comparison,
        "test": "exact McNemar (seed-paired)",
        "strata": "pairs by seed",
        "n_strata": n_pairs,
        "a_k": a_k,
        "a_n": n_pairs,
        "b_k": b_k,
        "b_n": n_pairs,
        "effect": "discordant A-only : B-only",
        "effect_value": f"{b}:{c}",
        "p": p,
        "p_one_sided_A_greater": p_one,
        "p_holm": None,
        "note": note,
    }


def stratified_tests(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def k(
        model: str,
        prompting: str,
        dialect: str,
        level: str,
        pred: Callable[[dict[str, Any]], bool] = is_success,
    ) -> tuple[int, int]:
        return cell(rows, arm_key(model, prompting, dialect), level, pred)

    family = "A: model pairs, L2+L3"
    family_rows = []
    for model_a, model_b in MODEL_PAIRS:
        tables: list[Table2x2] = []
        for prompting in PROMPTINGS:
            for dialect in DIALECTS:
                for level in ("L2", "L3"):
                    k_a, n_a = k(model_a, prompting, dialect, level)
                    k_b, n_b = k(model_b, prompting, dialect, level)
                    tables.append((k_a, n_a - k_a, k_b, n_b - k_b))
        family_rows.append(
            test_row(
                family,
                f"{MODEL_LABEL[model_a]} vs {MODEL_LABEL[model_b]}",
                tables,
                "prompting x dialect x level (8)",
                "model = model x access path; strata are different epochs",
            )
        )
    for row, p_holm in zip(family_rows, holm([row["p"] for row in family_rows]), strict=True):
        row["p_holm"] = p_holm
    out += family_rows

    family = "B: zero-shot vs multi-shot, L2+L3"
    family_rows = []
    for model in MODELS:
        tables = []
        for dialect in DIALECTS:
            for level in ("L2", "L3"):
                k_a, n_a = k(model, "zero_shot", dialect, level)
                k_b, n_b = k(model, "multi_shot", dialect, level)
                tables.append((k_a, n_a - k_a, k_b, n_b - k_b))
        epoch_note = (
            "same epoch for zs and ms"
            if model in ("claude_haiku", "glm")
            else "zs and ms in different epochs (E8/E9 vs E12/E13)"
        )
        family_rows.append(
            test_row(
                family,
                f"{MODEL_LABEL[model]}: zs vs ms",
                tables,
                "dialect x level (4)",
                epoch_note,
            )
        )
    for row, p_holm in zip(family_rows, holm([row["p"] for row in family_rows]), strict=True):
        row["p_holm"] = p_holm
    out += family_rows
    for model in MODELS:
        tables = []
        for level in ("L2", "L3"):
            k_a, n_a = k(model, "zero_shot", "Tessera", level)
            k_b, n_b = k(model, "multi_shot", "Tessera", level)
            tables.append((k_a, n_a - k_a, k_b, n_b - k_b))
        out.append(
            test_row(
                "B': zs vs ms, Tessera only (sensitivity)",
                f"{MODEL_LABEL[model]}: zs vs ms (Tessera)",
                tables,
                "level (2)",
                "Tessera prompts identical in all epochs; not in the Holm family",
            )
        )

    family = "C: L2 vs L3"
    for name, arms in (
        (f"all {N_MAIN_ARMS} arms", tr.ARMS),
        ("Kestrel arms (post hoc)", [arm for arm in tr.ARMS if arm["dialect"] == "Kestrel"]),
        ("Tessera arms (post hoc)", [arm for arm in tr.ARMS if arm["dialect"] == "Tessera"]),
    ):
        tables = []
        for arm in arms:
            k2, n2 = cell(rows, arm["key"], "L2", is_success)
            k3, n3 = cell(rows, arm["key"], "L3", is_success)
            tables.append((k2, n2 - k2, k3, n3 - k3))
        out.append(
            test_row(
                family,
                f"L2 vs L3, {name}",
                tables,
                f"arm ({len(tables)})",
                "per-arm Fisher tests: tab:stage-a-counts"
                if name.startswith("all")
                else "split chosen after seeing the data",
            )
        )

    out += material_tests(rows)
    return out


def pipeline_regime_tests(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    info: dict[str, Any] = {}
    family = "P: feedback regime, flash-lite zs L4+L5 (pipeline development)"
    tables = []
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for syntax_arm, exec_arm, _, _ in REGIME_PAIRS:
        for level in ("L4", "L5"):
            k_exec, n_exec = cell(rows, exec_arm, level, is_success)
            k_syntax, n_syntax = cell(rows, syntax_arm, level, is_success)
            tables.append((k_exec, n_exec - k_exec, k_syntax, n_syntax - k_syntax))
            exec_by_seed = {
                run["seed"]: run for run in rows if run["arm"] == exec_arm and run["level"] == level
            }
            for run in rows:
                if (
                    run["arm"] == syntax_arm
                    and run["level"] == level
                    and run["seed"] in exec_by_seed
                ):
                    pairs.append((exec_by_seed[run["seed"]], run))
    out.append(
        test_row(
            family,
            "execution (E10/E11) vs syntax only (E6/E7)",
            tables,
            "dialect x level (4)",
            "MH OR infinite: every execution cell is 10/10",
        )
    )
    exec_only = sum(
        1 for exec_run, syntax_run in pairs if exec_run["success"] and not syntax_run["success"]
    )
    syntax_only = sum(
        1 for exec_run, syntax_run in pairs if syntax_run["success"] and not exec_run["success"]
    )
    out.append(
        paired_row(
            family,
            "execution vs syntax only, all seed pairs",
            exec_only,
            syntax_only,
            len(pairs),
            sum(exec_run["success"] for exec_run, _ in pairs),
            sum(syntax_run["success"] for _, syntax_run in pairs),
            "regime effect: validator + execution loop together",
        )
    )
    identical = [
        (exec_run, syntax_run)
        for exec_run, syntax_run in pairs
        if exec_run["hash_first"] and exec_run["hash_first"] == syntax_run["hash_first"]
    ]
    exec_only_identical = sum(
        1 for exec_run, syntax_run in identical if exec_run["success"] and not syntax_run["success"]
    )
    syntax_only_identical = sum(
        1 for exec_run, syntax_run in identical if syntax_run["success"] and not exec_run["success"]
    )
    out.append(
        paired_row(
            family,
            "same, byte-identical first draft only",
            exec_only_identical,
            syntax_only_identical,
            len(identical),
            sum(exec_run["success"] for exec_run, _ in identical),
            sum(syntax_run["success"] for _, syntax_run in identical),
            "only these pairs isolate the feedback (identical starting draft)",
        )
    )
    info["regime_pairs"] = len(pairs)
    info["regime_identical_first"] = len(identical)
    info["regime_discordant_detail"] = [
        (
            exec_run["rel"],
            exec_run["epoch"],
            syntax_run["epoch"],
            exec_run["level"],
            exec_run["seed"],
            exec_run["hash_first"] == syntax_run["hash_first"],
            exec_run["fi"],
            exec_run["efr"],
            syntax_run["fc"],
        )
        for exec_run, syntax_run in pairs
        if exec_run["success"] != syntax_run["success"]
    ]
    l3_pairs = []
    for syntax_epoch, exec_epoch in ((6, 10), (7, 11)):
        for prompting in PROMPTINGS:
            exec_l3_by_seed = {
                run["seed"]: run
                for run in rows
                if run["epoch"] == exec_epoch
                and run["model"] == "gemini_lite"
                and run["prompting"] == prompting
                and run["level"] == "L3"
            }
            syntax_l3_by_seed = {
                run["seed"]: run
                for run in rows
                if run["epoch"] == syntax_epoch
                and run["model"] == "gemini_lite"
                and run["prompting"] == prompting
                and run["level"] == "L3"
            }
            l3_pairs += [
                (exec_l3_by_seed[seed], syntax_l3_by_seed[seed])
                for seed in sorted(set(exec_l3_by_seed) & set(syntax_l3_by_seed))
            ]
    exec_only_l3 = sum(
        1 for exec_run, syntax_run in l3_pairs if exec_run["success"] and not syntax_run["success"]
    )
    syntax_only_l3 = sum(
        1 for exec_run, syntax_run in l3_pairs if syntax_run["success"] and not exec_run["success"]
    )
    out.append(
        paired_row(
            family,
            "L3 control (no feedback, zs+ms): E10/E11 vs E6/E7",
            exec_only_l3,
            syntax_only_l3,
            len(l3_pairs),
            sum(exec_run["success"] for exec_run, _ in l3_pairs),
            sum(syntax_run["success"] for _, syntax_run in l3_pairs),
            "zs and ms; regeneration noise under the same seed",
        )
    )
    info["l3_control_identical"] = sum(
        1 for exec_run, syntax_run in l3_pairs if exec_run["hash_final"] == syntax_run["hash_final"]
    )
    for row in out:
        row["role"] = PIPELINE
    return out, info


def material_tests(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    family = "D: material before feedback"
    for model in (None, *MODELS):
        arms = [arm for arm in tr.ARMS if model is None or arm["model"] == model]
        tables = []
        for arm in arms:
            k2, n2 = cell(rows, arm["key"], "L2", is_success)
            k5, n5 = cell(rows, arm["key"], "L5", is_first_ok)
            tables.append((k2, n2 - k2, k5, n5 - k5))
        name = f"all {N_MAIN_ARMS} arms" if model is None else MODEL_LABEL[model]
        out.append(
            test_row(
                family,
                f"L2 vs L5 first draft, {name}",
                tables,
                f"arm ({len(tables)})",
                "" if model is None else "exploratory, per model",
            )
        )
    tables = []
    for arm in tr.ARMS:
        k4, n4 = cell(rows, arm["key"], "L4", is_first_ok)
        k5, n5 = cell(rows, arm["key"], "L5", is_first_ok)
        tables.append((k4, n4 - k4, k5, n5 - k5))
    out.append(
        test_row(
            family,
            f"L4 first vs L5 first draft, all {N_MAIN_ARMS} arms",
            tables,
            f"arm ({len(tables)})",
        )
    )
    l3_l4_pairs = []
    for arm in tr.ARMS:
        l3_by_seed = {
            run["seed"]: run for run in rows if run["arm"] == arm["key"] and run["level"] == "L3"
        }
        l3_l4_pairs += [
            (l3_by_seed[run["seed"]], run)
            for run in rows
            if run["arm"] == arm["key"] and run["level"] == "L4" and run["seed"] in l3_by_seed
        ]
    l3_only = sum(1 for l3, l4 in l3_l4_pairs if l3["success"] and not l4["first_ok"])
    l4_only = sum(1 for l3, l4 in l3_l4_pairs if l4["first_ok"] and not l3["success"])
    both = sum(1 for l3, l4 in l3_l4_pairs if l3["success"] and l4["first_ok"])
    neither = sum(1 for l3, l4 in l3_l4_pairs if not l3["success"] and not l4["first_ok"])
    out.append(
        paired_row(
            family,
            f"L3 vs L4 first draft, all {N_MAIN_ARMS} arms",
            l3_only,
            l4_only,
            len(l3_l4_pairs),
            sum(l3["success"] for l3, _ in l3_l4_pairs),
            sum(l4["first_ok"] for _, l4 in l3_l4_pairs),
            f"both {both}, neither {neither}; haiku/glm ignore the seed",
        )
    )
    out[-1]["_l3_l4_table"] = (both, l3_only, l4_only, neither)
    return out


def tex_tests(
    tests: Sequence[dict[str, Any]], caption: str, label: str, note: str, with_holm: bool = True
) -> str:
    ncol = 6 if with_holm else 5
    body: list[list[str]] = []
    midrule_rows: list[int] = []
    family = None
    for test in tests:
        if test["family"] != family:
            if body:
                midrule_rows.append(len(body) - 1)
            body.append([f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{tex(test['family'])}}}}}"])
            family = test["family"]
        if isinstance(test["effect_value"], str):
            effect = test["effect_value"]
        else:
            effect = fmt_or(test["effect_value"])
        cells = [
            "\\quad " + tex(test["comparison"]),
            f"{test['a_k']}/{test['a_n']}",
            f"{test['b_k']}/{test['b_n']}",
            effect,
            fmt_p_sci(test["p"]),
        ]
        if with_holm:
            cells.append(fmt_p_sci(test["p_holm"]) if test["p_holm"] is not None else "")
        body.append(cells)
    header = ["Comparison", "A", "B", "OR / disc.", "$p$"]
    if with_holm:
        header.append("$p_\\mathrm{Holm}$")
    return latex_table(
        caption,
        label,
        header,
        body,
        "l" + "r" * (ncol - 1),
        note=note,
        size=r"\scriptsize",
        midrules_after=tuple(midrule_rows),
        tabcolsep="3pt",
    )


def tex_main_tests(tests: Sequence[dict[str, Any]]) -> str:
    return tex_tests(
        tests,
        "Exact stratified tests of Stage A success in the main matrix (12 arms of gemini-3.5-flash, claude-haiku-5-5 and glm-5.3). "
        "Each test conditions on the margins of every stratum (convolution of hypergeometric distributions, two-sided); "
        "seed-paired comparisons use the exact McNemar test. A and B are the two groups in the order of the comparison.",
        "tab:stage-a-stratified-tests",
        "OR = Mantel-Haenszel odds ratio of A against B; disc. = discordant seed pairs A-only : B-only. "
        "Holm is applied within family A (3 model pairs) and within family B (3 models); every other $p$ is uncorrected and exploratory. "
        "A: strata prompting $\\times$ dialect $\\times$ level, i.e. different epochs per model; the factor model includes the access path. "
        "B: flash compares zero-shot (E8/E9) with multi-shot (E12/E13) epochs and the changed Kestrel descriptor; haiku and glm ran both strategies in the same epoch. "
        "D: first draft = success without any feedback round.",
    )


def tex_pipeline_tests(tests: Sequence[dict[str, Any]]) -> str:
    return tex_tests(
        tests,
        "Pipeline development (gemini-3.5-flash-lite), not part of the model comparison: Stage A success of flash-lite zero-shot at L4/L5 under execution feedback (E10/E11) "
        "against syntax-only feedback (E6/E7), same prompts and seeds, with an L3 control (no feedback in either regime).",
        "tab:pipeline-feedback-regime-tests",
        "OR = Mantel-Haenszel odds ratio of A against B; disc. = discordant seed pairs A-only : B-only (exact McNemar). "
        "All $p$ are uncorrected and exploratory. "
        + tr.PIPELINE_LEGEND
        + " These tests concern the feedback loop, not a model. E6/E7 and E10/E11 share prompts and seeds; only the byte-identical first drafts isolate the feedback, "
        "and the L3 row shows how often outcomes flip by regeneration alone.",
        with_holm=False,
    )


def python_subclass(steps: str) -> tuple[str, str]:
    try:
        compile(steps, "steps.py", "exec")
    except SyntaxError as error:
        lines = steps.splitlines()
        line_no = error.lineno or 0
        line = lines[line_no - 1] if 0 < line_no <= len(lines) else ""
        previous = next(
            (
                candidate
                for candidate in reversed(lines[: max(0, line_no - 1)])
                if candidate.strip()
            ),
            "",
        )
        message = error.msg or ""
        where = f"{type(error).__name__}: {message} (line {line_no}): {line.strip()[:70]}"
        if CALL_LINE.match(line.strip()) and DECORATOR.match(previous):
            return "missing_def", where
        if "line continuation" in message:
            return "escape", where
        quote_msg = "part of the string" in message or "unterminated string" in message
        if quote_msg and (DECORATOR.match(line) or DECORATOR.match(previous)):
            return "quote_collision", where
        return "python_other", where
    return "python_other", "compiles (error only at load time)"


def classify_final(r: dict[str, Any]) -> tuple[str, str]:
    failure_class, detail = r["fc"], r["fd"]
    steps = read_text(r["dir"] / "steps.py")
    if failure_class == "generation_failed":
        return ("json_contract" if "JSON contract" in detail else "no_artefact_other"), detail[:90]
    if failure_class == "quoted_literal":
        return "quoted_literal", detail[:90]
    if failure_class == "api_rejected":
        if "influx_field_" in detail:
            return "weak_canary", detail[:90]
        if TIME_WINDOW.search(detail):
            return "time_window", detail[:90]
        return "guessed_schema", detail[:90]
    if failure_class == "syntax_error":
        if "no matching step" in detail:
            return "undefined_step", detail[:90]
        if "ImportError" in detail and "background" in detail:
            return "behave_import", detail[:90]
        if "begin/end-markers" in detail:
            return "regex_anchor", detail[:90]
        if "Feature could not be parsed" in detail:
            return "gherkin_parse", detail[:60]
        return python_subclass(steps)
    if failure_class == "nonconformant":
        if "write/points" in detail:
            summary = read_json(r["dir"] / "evaluation" / "summary.json") or {}
            proxy_stats = summary.get("proxy_stats") or {}
            by_operation = proxy_stats.get("by_operation") or {}
            write_stats = by_operation.get("write") or {}
            return (
                "reseeding",
                f"{detail}; {write_stats.get('n_requests')} write requests, {write_stats.get('n_points')} points",
            )
        return "nonconf_other", detail[:90]
    if failure_class in ("step_error", "scenario_failed", "timeout"):
        if CONTEXT_CONFIG.search(steps):
            return "context_config", detail[:90]
        return "runtime_other", detail[:90]
    return "runtime_other", f"{failure_class}: {detail[:80]}"


def classify_first_draft(r: dict[str, Any]) -> tuple[str, str]:
    first_round = r["rounds"][0]
    steps = read_text(r["first_dir"] / "steps.py") if r["first_dir"] else ""
    if first_round.get("kind") == "execution":
        rejections = " ".join(first_round.get("rejections") or [])
        checks = first_round.get("failed_checks") or []
        if rejections:
            if TIME_WINDOW.search(rejections):
                return "time_window", rejections[:90]
            if "influx_field_" in rejections:
                return "weak_canary", rejections[:90]
            return "guessed_schema", rejections[:90]
        if first_round.get("executed_ok"):
            if "write/points" in checks:
                return "reseeding", ", ".join(checks)
            return "nonconf_other", ", ".join(checks)
        if CONTEXT_CONFIG.search(steps):
            return "context_config", "behave exit 1; " + ", ".join(checks[:3])
        return "runtime_other", "behave exit 1; " + ", ".join(checks[:3])
    codes = [issue.get("code", "") for issue in first_round.get("issues") or []]
    messages = " ".join(issue.get("message", "") for issue in first_round.get("issues") or [])
    if "UNDEFINED_STEP" in codes:
        return "undefined_step", messages[:90]
    if "E001" in codes:
        return "gherkin_parse", messages[:60]
    if "IMPORT_ERROR" in codes:
        return ("behave_import" if "background" in messages else "python_other"), messages[:90]
    if "PY_COMPILE" in codes:
        return python_subclass(steps)
    return "runtime_other", f"validator {','.join(codes)}: {messages[:70]}"


def failure_units(
    rows: Sequence[dict[str, Any]], base_group: str, extra_groups: Sequence[str] = ()
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    passes = [
        (source, group)
        for group in (base_group, *extra_groups)
        for source in ("final", "first_draft")
    ]
    role = tr.ROLE_MAIN if base_group == "main" else PIPELINE
    for source, group in passes:
        for run in rows:
            if run["group"] != group:
                continue
            if source == "final":
                if run["success"]:
                    continue
                subclass, detail = classify_final(run)
                draft_hash = run["hash_final"]
                draft = run["dir"]
            else:
                if run["level"] not in ("L4", "L5") or run["fi"] == 0 or not run["rounds"]:
                    continue
                subclass, detail = classify_first_draft(run)
                draft_hash = run["hash_first"]
                draft = run["first_dir"]
            duplicate_of = None
            if draft_hash and not (source == "final" and group == base_group):
                duplicate_of = seen.get(draft_hash)
            if draft_hash and draft_hash not in seen:
                seen[draft_hash] = f"{run['rel']} (E{run['epoch']}, {source})"
            units.append(
                {
                    "role": role,
                    "model": run["model"],
                    "arm": run["arm"],
                    "epoch": run["epoch"],
                    "level": run["level"],
                    "run_directory": run["rel"],
                    "source": source,
                    "group": group,
                    "subclass": subclass,
                    "detail": detail,
                    "duplicate_of": duplicate_of or "",
                    "counted": duplicate_of is None,
                    "base_final": source == "final" and group == base_group,
                    "has_context_config": bool(
                        draft and CONTEXT_CONFIG.search(read_text(draft / "steps.py"))
                    ),
                }
            )
    return units


def subclass_rows(
    units: Sequence[dict[str, Any]], models: Sequence[str] = MODELS
) -> list[dict[str, Any]]:
    out = []
    for key, label, rule in SUBCLASSES:
        row: dict[str, Any] = {"subclass": key, "label": label, "rule": rule}
        for model in models:
            final_units = [
                unit
                for unit in units
                if unit["subclass"] == key and unit["model"] == model and unit["base_final"]
            ]
            extra_units = [
                unit
                for unit in units
                if unit["subclass"] == key
                and unit["model"] == model
                and unit["counted"]
                and not unit["base_final"]
            ]
            row[f"{model}_final"] = len(final_units)
            row[f"{model}_extra"] = len(extra_units)
            row[f"{model}_final_by_level"] = " ".join(
                f"{level}:{n}"
                for level, n in sorted(Counter(unit["level"] for unit in final_units).items())
            )
        candidates = [unit for unit in units if unit["subclass"] == key and unit["counted"]]
        candidates.sort(
            key=lambda unit: (
                not unit["base_final"],
                unit["level"] == "L1",
                unit["level"],
                unit["run_directory"],
            )
        )
        preferred = PREFERRED_EXAMPLE.get(key)
        example = next(
            (
                unit
                for unit in candidates
                if preferred and unit["run_directory"].startswith(preferred)
            ),
            candidates[0] if candidates else None,
        )
        row["example"] = (
            f"{example['run_directory']} (E{example['epoch']}, {example['source'].replace('_', ' ')})"
            if example
            else ""
        )
        row["example_detail"] = example["detail"] if example else ""
        row["n_duplicates_dropped"] = sum(
            1 for unit in units if unit["subclass"] == key and not unit["counted"]
        )
        if any(row[f"{model}_final"] or row[f"{model}_extra"] for model in models):
            out.append(row)
    return out


def tex_subclasses(
    sub: Sequence[dict[str, Any]], units: Sequence[dict[str, Any]], pipeline: bool = False
) -> str:
    models = (tr.PIPELINE_MODEL,) if pipeline else MODELS
    body = []
    for row in sub:
        cells = []
        for model in models:
            n_final, n_extra = row[f"{model}_final"], row[f"{model}_extra"]
            text = "--" if not n_final and not n_extra else str(n_final)
            text += f"\\,+{n_extra}" if n_extra else ""
            cells.append(text)
        example_dir = row["example"].split(" (")[0]
        body.append(
            [
                f"\\raggedright {row['label']}",
                f"\\raggedright {row['rule']}; e.g.\\ \\texttt{{{tex(example_dir)}}}",
                *cells,
            ]
        )
    totals = []
    for model in models:
        n_final = sum(1 for unit in units if unit["model"] == model and unit["base_final"])
        n_extra = sum(
            1
            for unit in units
            if unit["model"] == model and unit["counted"] and not unit["base_final"]
        )
        totals.append(f"{n_final}\\,+{n_extra}" if n_extra else str(n_final))
    body.append(["\\textit{Total}", "", *totals])
    masked = sum(
        1
        for unit in units
        if unit["counted"] and unit["has_context_config"] and unit["subclass"] != "context_config"
    )
    if pipeline:
        caption = (
            "Pipeline development (gemini-3.5-flash-lite), not part of the model comparison: failure sub-classes of the flash-lite runs, one exclusive class per failing draft, "
            "assigned by the same rules as in the main table. The first number counts the final Stage A failures of the four flash-lite arms (E10--E13), "
            "the number after the plus counts further distinct failing drafts (rejected L4/L5 first drafts of E10--E13 and the final failures and rejected first drafts of the syntax-only epochs E6/E7)."
        )
        label = "tab:pipeline-failure-subclasses"
        header = ["Sub-class", "Rule and example run directory", "flash-lite"]
        align = "p{3.0cm}p{8.4cm}r"
        lead = (
            tr.PIPELINE_LEGEND
            + " Drafts byte-identical (SHA-1 of feature and steps file) to a draft already counted are dropped, e.g. the E6/E7 artefacts that reappear in E10/E11 under the same seed. "
        )
    else:
        caption = (
            "Failure sub-classes per model in the main matrix (12 arms): one exclusive class per failing draft, assigned by the first matching rule. "
            "The first number counts the final Stage A failures, the number after the plus counts further distinct failing drafts (rejected L4/L5 first drafts)."
        )
        label = "tab:failure-subclasses"
        header = ["Sub-class", "Rule and example run directory", "flash", "haiku", "glm"]
        align = "p{2.6cm}p{7.6cm}rrr"
        lead = (
            "flash = gemini-3.5-flash, haiku = claude-haiku-5-5, glm = glm-5.3. "
            "Drafts byte-identical (SHA-1 of feature and steps file) to a draft already counted are dropped. "
        )
    return latex_table(
        caption,
        label,
        header,
        body,
        align,
        note=lead
        + "First drafts are classified by the first feedback round (validator code or execution check). "
        + (
            f"{masked} further counted draft assigns \\texttt{{context.config}} but stops earlier and is counted under that earlier class. "
            if masked == 1
            else f"{masked} further counted drafts assign \\texttt{{context.config}} but stop earlier and are counted under that earlier class. "
        )
        + "Sub-class boundaries depend on the rules; the time-window, re-seeding and quote-collision counts in particular shift by one or two runs under other detectors.",
        size=r"\scriptsize",
        midrules_after=(len(body) - 2,),
        tabcolsep="3pt",
    )


def dict_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    out_rows = []
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key)
            if isinstance(value, float):
                if math.isinf(value) or math.isnan(value):
                    value = fmt_or_plain(value)
                else:
                    value = f"{value:.6g}"
            values.append(value)
        out_rows.append(values)
    write_csv(path, keys, out_rows)


def unchecked_sentence(n: int) -> str:
    if n == 0:
        return "Every repaired run ended on a draft that was execution-checked at generation time."
    runs = "run" if n == 1 else "runs"
    return (
        f"In {n} repaired {runs} the loop ended on a final draft that was not execution-checked at generation time "
        f"(all five rounds used); the evaluation decides {'its' if n == 1 else 'their'} outcome as for every other run."
    )


def float_block(figs: Sequence[tuple[str, str, str]]) -> str:
    blocks = []
    for name, label, caption in figs:
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


def tex_figure_floats(
    lad: Sequence[dict[str, Any]],
    n_unchecked: int,
    clipped: Sequence[tuple[str, str, str, int]],
) -> str:
    def k(model: str, step: str) -> tuple[int, int]:
        row = next(entry for entry in lad if entry["model"] == model and entry["step"] == step)
        return int(row["successes"]), int(row["n"])

    l1 = [k(model, "L1") for model in MODELS]
    if all(successes == 0 for successes, _ in l1) and len({n for _, n in l1}) == 1:
        l1_txt = f"L1 (0 of {l1[0][1]} runs in every model) is not plotted; it is analysed in \\cref{{fig:l1_operation_progress}}."
    else:
        l1_txt = (
            "L1 ("
            + ", ".join(
                f"{MODEL_LABEL[model]} {successes} of {n}"
                for model, (successes, n) in zip(MODELS, l1, strict=True)
            )
            + ") is not plotted; it is analysed in \\cref{fig:l1_operation_progress}."
        )
    clip_txt = ""
    if clipped:
        clip_parts = []
        for model in MODELS:
            values = sorted({value for clip_model, _, _, value in clipped if clip_model == model})
            if values:
                clip_parts.append(
                    f"{MODEL_LABEL[model]}: " + " and ".join(f"{value}\\,\\%" for value in values)
                )
        words = ("No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")
        count = words[len(clipped)] if len(clipped) < len(words) else str(len(clipped))
        clip_txt = (
            f" {count} single-arm values ({'; '.join(clip_parts)}) lie below the axis range; "
            "they sit at the lower edge and carry their value."
        )
    axis_txt = f"The y axis starts at {LADDER_Y_LO:.0f}\\,\\% (axis break at the bottom).{clip_txt}"

    repair_parts = []
    for model in MODELS:
        first = sum(k(model, step)[0] for step in ("L4 first", "L5 first"))
        final = sum(k(model, step)[0] for step in ("L4 final", "L5 final"))
        n = sum(k(model, step)[1] for step in ("L4 final", "L5 final"))
        repair_parts.append(f"{MODEL_LABEL[model]} {final - first} of {n}")
    jump = (
        "Descriptively, first-draft success changes little between L2 and the L4/L5 first drafts (tested in family D); "
        "the L4/L5 runs that reach success only through the feedback loop are "
        + ", ".join(repair_parts)
        + "."
    )
    return float_block(
        [
            (
                "stage_a_prefeedback_ladder",
                "fig:stage-a-prefeedback-ladder",
                "Stage A success per model in the main matrix (gemini-3.5-flash, claude-haiku-5-5, glm-5.3) before and after feedback: L2, L3 and the first drafts at L4/L5 (no feedback round) against the final L4/L5 result. "
                "Large points are model totals over the four arms of a model (40 runs per step; flash: four epochs, haiku and glm: two epochs each; descriptive, not tested) with 95\\,\\% Clopper-Pearson intervals; small points are the single arms (10 runs). "
                + l1_txt
                + " "
                + axis_txt
                + " "
                + jump,
            ),
            (
                "stage_a_repair_kind",
                "fig:stage-a-repair-kind",
                "How the L4/L5 runs of the main matrix reach success: first draft accepted, repaired in validator rounds only, or repaired with at least one execution-feedback round; 40 runs per model and level. "
                + unchecked_sentence(n_unchecked),
            ),
        ]
    )


def tex_pipeline_figure_floats(n_unchecked_pipeline: int) -> str:
    return float_block(
        [
            (
                "pipeline_repair_kind_flash_lite",
                "fig:pipeline-repair-kind-flash-lite",
                "Pipeline development (gemini-3.5-flash-lite), not part of the model comparison: kind of success of the flash-lite zero-shot L4/L5 runs under execution feedback (E10/E11) "
                "and syntax-only feedback (E6/E7), same prompts and seeds, 20 runs per row. flash-lite was used only to develop and test the pipeline cheaply; "
                "the contrast is evidence about the feedback loop, not about a model. "
                + unchecked_sentence(n_unchecked_pipeline),
            ),
        ]
    )


def write_readme(out: Path, headline: Sequence[str], pipeline_lines: Sequence[str]) -> None:
    text = f"""# Stage A deep dive (RQ1, RQ2a, RQ3)

Generated by `scripts/thesis_stage_a_extra.py` (run from the project root with
`uv run python scripts/thesis_stage_a_extra.py`). Same data basis and arm definitions as
`scripts/thesis_results.py`; nothing in the main package is changed. Stage A only (execution
success); there are no Stage-B figures here, so the two Stage-B views do not apply.

**Model comparison = the 12 main arms only**: gemini-3.5-flash (zs Kestrel E8, zs Tessera E9,
ms Tessera E12, ms Kestrel E13), claude-haiku-5-5 (zs+ms Kestrel E14, Tessera E15) and glm-5.3
(zs+ms Kestrel E16, Tessera E17); N = 10 per arm and level, 600 runs.
**gemini-3.5-flash-lite was used only to develop and test the pipeline cheaply.** Its runs (E10-E13
and the syntax-only-feedback epochs E6/E7) never enter a model figure, table or test; they appear
only in the files whose names start with `pipeline_` and in rows whose `role` column says
"pipeline development (flash-lite)". Those files are kept for the record only and do not go into
the thesis.

**For the thesis**: `fig/stage_a_prefeedback_ladder`, `fig/stage_a_repair_kind`, `latex/figures.tex`,
`latex/stage_a_counts.tex`, `latex/stage_a_stratified_tests.tex`, `latex/failure_subclasses.tex`
(three compared models only).

## Files

| File | Content |
|---|---|
| `fig/stage_a_prefeedback_ladder.pdf/.png` | Main matrix: success per model at L2, L3, L4/L5 first draft and L4/L5 final; model totals (40 runs, Clopper-Pearson 95 %) and the single arms. L1 (0 of 40 in every model, see the L1 figure) is left out and noted in the figure; the y axis runs from {LADDER_Y_LO:.0f} to 100 % with an axis-break mark, single-arm values below sit at the lower edge with their value. The CSV keeps L1 |
| `fig/stage_a_repair_kind.pdf/.png` | Main matrix: L4/L5 runs by kind of success (first draft / validator-only repair / >=1 execution round / failed), per model |
| `fig/pipeline_repair_kind_flash_lite.pdf/.png` | Pipeline development only (flash-lite): the same split for zero-shot E10/E11 (execution feedback) vs E6/E7 (syntax-only feedback) |
| `tables/stage_a_prefeedback_ladder.csv` | Numbers behind the ladder figure (3 models + total over the 12 arms) |
| `tables/stage_a_repair_kind.csv` | Numbers behind the main repair figure, incl. execution-from-round-1, validator-then-execution and `n_unchecked_final` |
| `tables/pipeline_repair_kind_flash_lite.csv` | Numbers behind the pipeline repair figure (flash-lite, pipeline development) |
| `tables/stage_a_counts.csv`, `latex/stage_a_counts.tex` | Successes per main arm x level with Clopper-Pearson CI, first drafts, onset, L2+L3, Fisher L2 vs L3 (+ Holm over the 12 arms) |
| `tables/stage_a_stratified_tests.csv`, `latex/stage_a_stratified_tests.tex` | Exact stratified tests on the main arms (families A-D, see below) |
| `tables/pipeline_feedback_regime_tests.csv`, `latex/pipeline_feedback_regime_tests.tex` | Pipeline development only: feedback-regime contrast E6/E7 vs E10/E11 (family P) |
| `tables/failure_subclasses.csv`, `latex/failure_subclasses.tex` | Main models x failure sub-class with rule and example run directory |
| `tables/failure_units.csv` | Every classified failing draft of the main arms (source, sub-class, detail, duplicate_of) |
| `tables/pipeline_failure_subclasses_flash_lite.csv`, `latex/pipeline_failure_subclasses_flash_lite.tex` | Pipeline development only: the same taxonomy for flash-lite (E10-E13, plus E6/E7) |
| `tables/pipeline_failure_units_flash_lite.csv` | Every classified failing flash-lite draft |
| `latex/figures.tex` | Figure floats (main matrix) with draft captions |
| `latex/pipeline_figures.tex` | Figure float of the pipeline-development figure (flash-lite), labelled as such |

## Definitions

- **First draft accepted**: `success and feedback_iterations == 0`. In E6-E9 this means validator ok
  and evaluation successful; from E10 on the generation-time execution check also passed.
- **Validator-only repair**: `feedback_iterations > 0` and `execution_feedback_rounds == 0`.
  **Execution repair**: at least one execution-feedback round (from round 1 or after validator rounds).
- **n_unchecked_final**: repaired runs whose final draft was never execution-checked at generation
  time (the loop used all five rounds); the evaluation decides their outcome like every other run.
- **Exact stratified test**: the summed group-A successes are compared with their exact conditional
  distribution given every stratum's margins (convolution of hypergeometric pmfs), two-sided by
  summing outcomes no more likely than the observed one. Effect = Mantel-Haenszel OR.
- **Families (main arms only)**: A pairs of the three main models at L2+L3 (strata prompting x
  dialect x level, Holm over 3 pairs); B zero- vs multi-shot per main model (strata dialect x level,
  Holm over 3 models; B' Tessera-only sensitivity, not in the Holm family); C L2 vs L3 (strata =
  arms; per-arm Fisher in the counts table); D material before feedback (L2 vs L5 first draft overall
  and per model, L4 vs L5 first draft, L3 vs L4 first draft seed-paired). Everything outside A and B
  is exploratory and uncorrected.
- **Family P (pipeline development, flash-lite, not model evidence)**: regime contrast E6/E7
  (syntax-only) vs E10/E11 (execution feedback), zero-shot L4+L5: stratified test, seed-paired
  McNemar, identical-first-draft subset, and an L3 control (no feedback in either regime). Uncorrected.
- **Failure sub-classes**: exclusive, first matching rule wins (order as in the table). Final failures
  are classified from `failure_class`/`failure_detail` and the final artefact; rejected first drafts
  from the first feedback round. Drafts byte-identical to one counted before are dropped. The main
  table and the flash-lite pipeline table are deduplicated separately.

## Key numbers, main matrix (12 arms, 600 runs)

{chr(10).join("- " + h for h in headline)}

## Pipeline development (gemini-3.5-flash-lite), for the record only: not part of the model comparison, not for the thesis

{chr(10).join("- " + h for h in pipeline_lines)}

Caveats: model totals pool the four arms of a model (flash: four epochs; haiku and glm: two epochs
each) and are descriptive only; the factor model is model x access path (Gemini API, Claude CLI,
OpenRouter); haiku and glm do not honour the seed, so their seed pairs are not paired in any causal
sense; sub-class counts depend on the detector rules.
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def test_line(t: dict[str, Any]) -> str:
    if isinstance(t["effect_value"], str):
        effect = t["effect_value"]
    else:
        effect = fmt_or_plain(t["effect_value"])
    holm_text = f", Holm {t['p_holm']:.2g}" if t["p_holm"] is not None else ""
    p_text = "n/a (no discordant pairs)" if t["p"] is None else f"{t['p']:.2g}"
    return (
        f"[{t['family']}] {t['comparison']}: {t['a_k']}/{t['a_n']} vs {t['b_k']}/{t['b_n']}, "
        f"{t['effect']} = {effect}, p = {p_text}{holm_text}"
    )


def subclass_lines(
    units: Sequence[dict[str, Any]], sub: Sequence[dict[str, Any]], models: Sequence[str]
) -> list[str]:
    tot_main = Counter(unit["model"] for unit in units if unit["base_final"])
    tot_extra = Counter(
        unit["model"] for unit in units if unit["counted"] and not unit["base_final"]
    )
    lines = [
        "Failing drafts (final + distinct extra): "
        + ", ".join(
            f"{MODEL_LABEL[model]} {tot_main[model]}+{tot_extra[model]}" for model in models
        )
        + f"; duplicates dropped {sum(1 for unit in units if not unit['counted'])}"
    ]
    for row in sub:
        lines.append(
            f"  {row['label']}: "
            + ", ".join(
                f"{MODEL_LABEL[model]} {row[f'{model}_final']}+{row[f'{model}_extra']}"
                for model in models
            )
            + f" | e.g. {row['example']}"
        )
    return lines


def _main_headline(
    ladder: Sequence[dict[str, Any]],
    repair: Sequence[dict[str, Any]],
    counts: Sequence[dict[str, Any]],
    tests: Sequence[dict[str, Any]],
    l3_l4: tuple[int, int, int, int],
    units: Sequence[dict[str, Any]],
    subclasses: Sequence[dict[str, Any]],
    n_unchecked: int,
) -> list[str]:
    headline: list[str] = []
    total_label = f"all ({N_MAIN_ARMS} arms)"

    def step_total(step: str) -> str:
        row = next(
            entry for entry in ladder if entry["model"] == total_label and entry["step"] == step
        )
        return f"{row['successes']}/{row['n']}"

    headline.append(
        f"Ladder ({N_MAIN_ARMS} main arms): "
        + ", ".join(f"{step} {step_total(step)}" for step, _ in LADDER_TABLE_STEPS)
    )
    for model in MODELS:
        successes = [
            next(row for row in ladder if row["model"] == model and row["step"] == step)[
                "successes"
            ]
            for step in ("L1", "L2", "L3", "L4 first", "L5 first", "L4 final", "L5 final")
        ]
        headline.append(
            f"{MODEL_LABEL[model]} L1/L2/L3/L4-first/L5-first/L4-final/L5-final of 40: "
            + "/".join(map(str, successes))
        )
    repair_totals: Counter[str] = Counter()
    for row in repair:
        for kind, _ in REPAIR_KINDS:
            repair_totals[kind] += row[kind]
        repair_totals["exec_r1"] += row["execution_from_round_1"]
        repair_totals["val_then_exec"] += row["validator_then_execution"]
    n_l45 = sum(row["n"] for row in repair)
    headline.append(
        f"L4/L5 main matrix ({n_l45} runs): first draft {repair_totals['first_ok']}, repaired {repair_totals['validator_only'] + repair_totals['execution']} "
        f"(validator-only {repair_totals['validator_only']}, >=1 execution round {repair_totals['execution']}: from round 1 {repair_totals['exec_r1']}, validator then execution {repair_totals['val_then_exec']}), failed {repair_totals['failed']}, n_unchecked_final {n_unchecked}"
    )
    for model in MODELS:
        model_rows = [row for row in repair if row["key"] == model]
        headline.append(
            f"  {MODEL_LABEL[model]}: first draft {sum(row['first_ok'] for row in model_rows)}, validator-only {sum(row['validator_only'] for row in model_rows)}, "
            f"execution {sum(row['execution'] for row in model_rows)}, failed {sum(row['failed'] for row in model_rows)}, rounds {sum(row['rounds_repaired'] for row in model_rows)}"
        )
    onsets = Counter(
        row["onset"] for row in counts if not str(row["arm"]).startswith("model total")
    )
    headline.append(
        "Onset (arms): "
        + ", ".join(f"{onset_level} {n}" for onset_level, n in sorted(onsets.items()))
    )
    smallest = min(
        (row for row in counts if row["fisher_L2_vs_L3_p"] is not None),
        key=lambda row: row["fisher_L2_vs_L3_p"],
    )
    headline.append(
        f"Smallest per-arm Fisher L2 vs L3: {smallest['arm']} {smallest['L2_k']} vs {smallest['L3_k']}, p = {smallest['fisher_L2_vs_L3_p']:.3f}, Holm({N_MAIN_ARMS}) = {smallest['fisher_L2_vs_L3_holm']:.2f}"
    )
    headline += [test_line(test) for test in tests]
    both, l3_only, l4_only, neither = l3_l4
    headline.append(
        f"L3 vs L4 first draft per seed: both {both}, L3 only {l3_only}, L4-first only {l4_only}, neither {neither}"
    )
    headline += subclass_lines(units, subclasses, MODELS)
    return headline


def _pipeline_lines(
    repair: Sequence[dict[str, Any]],
    tests: Sequence[dict[str, Any]],
    info: dict[str, Any],
    units: Sequence[dict[str, Any]],
    subclasses: Sequence[dict[str, Any]],
) -> list[str]:
    lines: list[str] = []
    for row in repair:
        lines.append(
            f"{row['group']} {row['level']}: first {row['first_ok']}, val-only {row['validator_only']}, exec {row['execution']}, failed {row['failed']} of {row['n']}"
        )
    lines += [test_line(test) for test in tests]
    lines.append(
        f"Regime contrast: {info['regime_pairs']} seed pairs, {info['regime_identical_first']} with byte-identical first draft; L3 control pairs with identical final artefact: {info['l3_control_identical']}"
    )
    lines += subclass_lines(units, subclasses, (tr.PIPELINE_MODEL,))
    return lines


def main() -> None:
    config, runs = tr.load_runs()
    root = config.paths.project_root
    out = config.paths.results_reports_dir / "thesis_results" / "extra" / "stage_a_extra"
    fig_dir, tab_dir, tex_dir = out / "fig", out / "tables", out / "latex"
    for directory in (fig_dir, tab_dir, tex_dir):
        directory.mkdir(parents=True, exist_ok=True)
    tr._style()

    rows = build_rows(runs, root)
    main_rows = [run for run in rows if run["group"] == "main"]
    assert len(main_rows) == 10 * len(LEVELS) * N_MAIN_ARMS == 600, len(main_rows)
    assert {run["model"] for run in main_rows} == set(MODELS)
    assert tr.PIPELINE_MODEL not in MODELS

    ladder = ladder_rows(main_rows)
    dict_csv(tab_dir / "stage_a_prefeedback_ladder.csv", ladder)
    fig_ladder(main_rows, ladder, fig_dir)

    repair = repair_rows(main_rows, main_repair_groups())
    dict_csv(tab_dir / "stage_a_repair_kind.csv", repair)
    fig_repair_kind(repair, fig_dir, "stage_a_repair_kind")

    counts = count_rows(main_rows)
    dict_csv(tab_dir / "stage_a_counts.csv", counts)
    (tex_dir / "stage_a_counts.tex").write_text(tex_counts(counts), encoding="utf-8")

    tests = stratified_tests(main_rows)
    l3_l4 = tests[-1].pop("_l3_l4_table")
    dict_csv(tab_dir / "stage_a_stratified_tests.csv", tests)
    (tex_dir / "stage_a_stratified_tests.tex").write_text(tex_main_tests(tests), encoding="utf-8")

    units = failure_units(rows, "main")
    subclasses = subclass_rows(units, MODELS)
    dict_csv(tab_dir / "failure_subclasses.csv", subclasses)
    dict_csv(tab_dir / "failure_units.csv", units)
    (tex_dir / "failure_subclasses.tex").write_text(
        tex_subclasses(subclasses, units), encoding="utf-8"
    )

    pipeline_repair = repair_rows(rows, pipeline_repair_groups())
    dict_csv(tab_dir / "pipeline_repair_kind_flash_lite.csv", pipeline_repair)
    fig_repair_kind(
        pipeline_repair,
        fig_dir,
        "pipeline_repair_kind_flash_lite",
        header=tr.PIPELINE_HEADER,
    )

    pipeline_tests, info = pipeline_regime_tests(rows)
    dict_csv(tab_dir / "pipeline_feedback_regime_tests.csv", pipeline_tests)
    (tex_dir / "pipeline_feedback_regime_tests.tex").write_text(
        tex_pipeline_tests(pipeline_tests), encoding="utf-8"
    )

    pipeline_units = failure_units(rows, "pipeline", ("syntax_only",))
    pipeline_subclasses = subclass_rows(pipeline_units, (tr.PIPELINE_MODEL,))
    dict_csv(tab_dir / "pipeline_failure_subclasses_flash_lite.csv", pipeline_subclasses)
    dict_csv(tab_dir / "pipeline_failure_units_flash_lite.csv", pipeline_units)
    (tex_dir / "pipeline_failure_subclasses_flash_lite.tex").write_text(
        tex_subclasses(pipeline_subclasses, pipeline_units, pipeline=True), encoding="utf-8"
    )

    n_unchecked = sum(row["n_unchecked_final"] for row in repair)
    n_unchecked_pipeline = sum(
        row["n_unchecked_final"] for row in pipeline_repair if row["key"] == "contrast_exec"
    )
    (tex_dir / "figures.tex").write_text(
        tex_figure_floats(ladder, n_unchecked, ladder_clipped(main_rows)), encoding="utf-8"
    )
    (tex_dir / "pipeline_figures.tex").write_text(
        tex_pipeline_figure_floats(n_unchecked_pipeline), encoding="utf-8"
    )

    headline = _main_headline(ladder, repair, counts, tests, l3_l4, units, subclasses, n_unchecked)
    pipeline_lines = _pipeline_lines(
        pipeline_repair, pipeline_tests, info, pipeline_units, pipeline_subclasses
    )

    write_readme(out, headline, pipeline_lines)
    print("MAIN MATRIX (12 arms)")
    print("\n".join(headline))
    print("\nPIPELINE DEVELOPMENT (flash-lite), not part of the model comparison")
    print("\n".join(pipeline_lines))
    print(
        "\nRegime discordant pairs (exec run, exec epoch, syntax epoch, level, seed, identical first draft, fi, efr, syntax-only failure class):"
    )
    for detail in info["regime_discordant_detail"]:
        print("  ", detail)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
