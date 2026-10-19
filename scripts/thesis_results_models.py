from __future__ import annotations

import itertools
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_results as tr
from thesis_results import (
    BASELINE,
    INK,
    INK_2,
    LEVELS,
    MODEL_COLOR,
    MODEL_FULL,
    MODEL_LABEL,
    OPS,
    OUTCOME_MAP,
    OUTCOMES,
    PROMPT_LABEL,
    RAMP_5,
    SURFACE,
    TEXT_WIDTH_IN,
    by_level,
    fisher,
    fmt,
    fmt_p,
    fmt_p_plain,
    holm,
    latex_table,
    save,
    sb,
    tex,
    write_csv,
)

from benchmark_agent.evaluation.cost import cost_of_usage
from benchmark_agent.evaluation.runs import SOURCE_PROXY, RunRecord
from benchmark_agent.evaluation.statistics import mann_whitney_prompting

plt = tr.plt
Line2D = tr.Line2D


MODELS: tuple[str, ...] = tr.MAIN_MODELS
DIALECTS: tuple[str, ...] = tr.DIALECTS
PROMPTINGS: tuple[str, ...] = ("zero_shot", "multi_shot")
VIEWS: tuple[str, ...] = tr.VIEWS
ACCESS = {
    "gemini": "Gemini API (AI Studio), seed sent",
    "claude_haiku": "Claude Code CLI (subscription), no seed, CLI-default effort",
    "glm": "OpenRouter, pinned to Z.AI (fp8), reasoning effort low, seed ignored",
}
VIEW_TEXT = {"all": "all measured runs", "success": "Stage-A successes only"}
MIN_RUNS_PER_LEVEL = 10
PIPELINE_NOTE = "gemini-3.5-flash-lite was used only to develop and test the pipeline cheaply and is not part of this comparison."

STALE_OUTPUTS: tuple[str, ...] = tuple(
    f"fig/{name}.{ext}"
    for name in (
        "models_stage_b_kestrel",
        "models_stage_b_tessera",
        "models_stage_b_p50_kestrel",
        "models_stage_b_p50_tessera",
        "models_stage_b_delta",
        "models_stage_b_delta_p50",
    )
    for ext in ("pdf", "png")
)


def build_arms(
    runs: Sequence[RunRecord], in_progress: list[str] | None = None
) -> list[dict[str, Any]]:
    arms = []
    for base in tr.ARMS:
        assert base["model"] in MODELS, base
        members = tr.arm_runs(runs, base)
        if not members:
            continue
        counts = [len(level_runs) for level_runs in by_level(members).values()]
        if min(counts) < MIN_RUNS_PER_LEVEL:
            if in_progress is not None:
                in_progress.append(
                    f"{base['key']} E{base['epoch']}: {len(members)} runs ({'/'.join(map(str, counts))} per level)"
                )
            continue
        arm = dict(base)
        arm["runs"] = members
        arm["short_tex"] = tr.arm_short_tex(arm)
        arm["dates"] = sorted(
            {(run.evaluated_at or "")[:10] for run in members if run.evaluated_at}
        )
        arms.append(arm)
    return arms


def _day(iso: str) -> str:
    months = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    year, month, day = iso.split("-")
    return f"{int(day)} {months[int(month) - 1]} {year}"


def measurement_days(arms: Sequence[dict[str, Any]]) -> str:
    by_day: dict[str, list[int]] = {}
    for arm in arms:
        for day in arm["dates"]:
            by_day.setdefault(day, [])
            if arm["epoch"] not in by_day[day]:
                by_day[day].append(arm["epoch"])
    return "; ".join(
        f"E{'/E'.join(map(str, sorted(epochs)))} on {_day(day)}"
        for day, epochs in sorted(by_day.items())
    )


def reference_basis(arm: dict[str, Any], reference: dict[str, Any]) -> str:
    if not arm["ref"]:
        return "none (no reference measured on that day)"
    ref_epoch = int(arm["ref"].split("E")[1])
    days = sorted({timestamp[:10] for timestamp in reference[arm["ref"]]["write"]["evaluated"]})
    kind = "same epoch" if ref_epoch == arm["epoch"] else "same day"
    return f"{kind}: gold {arm['ref']} ({', '.join(_day(day) for day in days)})"


def stage_a_rows(arms: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for arm in arms:
        for lvl, members in by_level(arm["runs"]).items():
            outcome = Counter(OUTCOME_MAP.get(run.failure_class, "other") for run in members)
            failure_details = Counter(
                f"{run.failure_class}: {run.failure_detail[:60]}"
                for run in members
                if not run.success
            )
            rows.append(
                {
                    "arm": arm["key"],
                    "model": MODEL_FULL[arm["model"]],
                    "prompting": arm["prompting"],
                    "dialect": arm["dialect"],
                    "epoch": arm["epoch"],
                    "feedback_regime": arm["regime"],
                    "level": lvl,
                    "n": len(members),
                    "success": sum(1 for run in members if run.success),
                    "syntax_ok": sum(1 for run in members if run.syntax_ok),
                    "first_draft_ok": sum(
                        1 for run in members if run.success and run.feedback_iterations == 0
                    ),
                    "repaired": sum(
                        1 for run in members if run.success and run.feedback_iterations > 0
                    ),
                    **{key: outcome.get(key, 0) for key, _ in OUTCOMES},
                    "failure_details": "; ".join(
                        f"{detail} x{count}" for detail, count in failure_details.most_common()
                    ),
                }
            )
    return rows


def cell(rows: Sequence[dict[str, Any]], arm_key: str, lvl: str) -> dict[str, Any]:
    return next(row for row in rows if row["arm"] == arm_key and row["level"] == lvl)


def stage_a_fisher(
    arms: Sequence[dict[str, Any]], a_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    out = []
    for dialect in DIALECTS:
        for prompting in PROMPTINGS:
            group = {
                arm["model"]: arm
                for arm in arms
                if arm["dialect"] == dialect and arm["prompting"] == prompting
            }
            for model_a, model_b in itertools.combinations(MODELS, 2):
                if model_a not in group or model_b not in group:
                    continue
                arm_a, arm_b = group[model_a], group[model_b]
                p_values = []
                for lvl in LEVELS:
                    cell_a = cell(a_rows, arm_a["key"], lvl)
                    cell_b = cell(a_rows, arm_b["key"], lvl)
                    p_values.append(
                        fisher(cell_a["success"], cell_a["n"], cell_b["success"], cell_b["n"])
                    )
                for lvl, p, p_holm in zip(LEVELS, p_values, holm(p_values), strict=True):
                    cell_a = cell(a_rows, arm_a["key"], lvl)
                    cell_b = cell(a_rows, arm_b["key"], lvl)
                    out.append(
                        {
                            "dialect": dialect,
                            "prompting": prompting,
                            "pair": f"{MODEL_LABEL[model_a]} vs {MODEL_LABEL[model_b]}",
                            "level": lvl,
                            "arm_a": arm_a["key"],
                            "epoch_a": arm_a["epoch"],
                            "arm_b": arm_b["key"],
                            "epoch_b": arm_b["epoch"],
                            "a_success": cell_a["success"],
                            "a_n": cell_a["n"],
                            "b_success": cell_b["success"],
                            "b_n": cell_b["n"],
                            "fisher_p": p,
                            "fisher_holm_5_levels": p_holm,
                        }
                    )
    return out


def stage_a_failed_measured(arms: Sequence[dict[str, Any]]) -> dict[str, Any]:
    failed = [
        (arm, run)
        for arm in arms
        for run in arm["runs"]
        if not run.success
        and any(run.stage_b_value(SOURCE_PROXY, op, "throughput_ops_s") is not None for op in OPS)
    ]

    def points(r: RunRecord) -> int | None:
        write_stats = r.op(SOURCE_PROXY, "write")
        return write_stats.n_points if write_stats is not None else None

    ok_points = Counter(points(run) for arm in arms for run in arm["runs"] if run.success)

    def fmt_counter(c: Counter[Any]) -> str:
        return ", ".join(f"{value} x{count}" for value, count in c.most_common())

    return {
        "n": len(failed),
        "models": fmt_counter(Counter(MODEL_FULL[arm["model"]] for arm, _ in failed)),
        "arms": fmt_counter(Counter(arm["key"] for arm, _ in failed)),
        "levels": fmt_counter(Counter(run.guidance_level for _, run in failed)),
        "classes": fmt_counter(Counter(run.failure_class for _, run in failed)),
        "write_points": fmt_counter(Counter(points(run) for _, run in failed)),
        "write_points_success": ok_points.most_common(1)[0][0] if ok_points else None,
    }


def feedback_summary(a_rows: Sequence[dict[str, Any]]) -> list[str]:
    out = []
    for model in MODELS:
        rows = [
            row
            for row in a_rows
            if row["model"] == MODEL_FULL[model] and row["level"] in ("L4", "L5")
        ]
        n = sum(row["n"] for row in rows)
        repaired = sum(row["repaired"] for row in rows)
        first = sum(row["first_draft_ok"] for row in rows)
        out.append(
            f"{MODEL_FULL[model]}: {first} of {n} L4/L5 runs accepted as first draft, {repaired} repaired after feedback"
        )
    return out


def stage_b_rows(
    arms: Sequence[dict[str, Any]],
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
    d_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    delta = {(row["arm"], row["level"], row["op"]): row for row in d_rows}
    rows = []
    for arm in arms:
        members = by_level(arm["runs"])
        for lvl in LEVELS[1:]:
            n_a_success = sum(1 for run in members[lvl] if run.success)
            for op in OPS:
                b_cell = b_cells[(arm["key"], lvl, op)]
                row: dict[str, Any] = {
                    "arm": arm["key"],
                    "model": MODEL_FULL[arm["model"]],
                    "prompting": arm["prompting"],
                    "dialect": arm["dialect"],
                    "epoch": arm["epoch"],
                    "measured_on": ", ".join(arm["dates"]),
                    "level": lvl,
                    "op": op,
                    "n_runs": len(members[lvl]),
                    "n_stage_a_success": n_a_success,
                    "n": b_cell["n"],
                    "n_success": b_cell["n_success"],
                    "n_stage_a_failed": b_cell["n_stage_a_failed"],
                    "n_stage_b_failed": n_a_success - b_cell["n_success"],
                    **{
                        key: b_cell.get(key)
                        for key in (
                            "tp_med",
                            "tp_q1",
                            "tp_q3",
                            "p50_med",
                            "p50_q1",
                            "p50_q3",
                            "p95_med",
                            "tp_s_med",
                            "tp_s_q1",
                            "tp_s_q3",
                            "p50_s_med",
                            "p50_s_q1",
                            "p50_s_q3",
                        )
                    },
                    "reference": arm["ref"] or "",
                }
                delta_row = delta.get((arm["key"], lvl, op))
                if delta_row is not None:
                    for key in DELTA_COLS:
                        row[key] = delta_row.get(key)
                rows.append(row)
    return rows


DELTA_COLS: tuple[str, ...] = (
    "ref_tp_med",
    "delta_tp_med",
    "delta_tp_q1",
    "delta_tp_q3",
    "delta_tp_med_success",
    "delta_tp_q1_success",
    "delta_tp_q3_success",
    "ref_p50_med",
    "delta_p50_med",
    "delta_p50_q1",
    "delta_p50_q3",
    "delta_p50_med_success",
    "delta_p50_q1_success",
    "delta_p50_q3_success",
    "mwu_vs_reference_p",
    "mwu_vs_reference_p_success",
)


def b_row(b_rows: Sequence[dict[str, Any]], arm_key: str, lvl: str, op: str) -> dict[str, Any]:
    return next(
        row for row in b_rows if row["arm"] == arm_key and row["level"] == lvl and row["op"] == op
    )


def _med(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def prompting_tests(arms: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model in MODELS:
        for dialect in DIALECTS:
            pair = {
                arm["prompting"]: arm
                for arm in arms
                if arm["model"] == model and arm["dialect"] == dialect
            }
            if len(pair) != 2:
                continue
            zs, ms = pair["zero_shot"], pair["multi_shot"]
            same_epoch = zs["epoch"] == ms["epoch"]
            a_zs = sum(1 for run in zs["runs"] if run.success)
            a_ms = sum(1 for run in ms["runs"] if run.success)
            for op in OPS:
                row: dict[str, Any] = {
                    "model": MODEL_FULL[model],
                    "dialect": dialect,
                    "epoch_zs": zs["epoch"],
                    "epoch_ms": ms["epoch"],
                    "op": op,
                    "stage_a_zs": f"{a_zs}/{len(zs['runs'])}",
                    "stage_a_ms": f"{a_ms}/{len(ms['runs'])}",
                    "stage_a_fisher_p": fisher(a_zs, len(zs["runs"]), a_ms, len(ms["runs"])),
                }
                for tag, arm in (("zs", zs), ("ms", ms)):
                    tp = sb(arm["runs"], op, "throughput_ops_s")
                    tp_s = sb(arm["runs"], op, "throughput_ops_s", "success")
                    row[f"{tag}_n_measured"] = len(tp)
                    row[f"{tag}_n_stage_a_failed"] = len(tp) - len(tp_s)
                    row[f"{tag}_median_tp"] = _med(tp)
                    row[f"{tag}_median_p50"] = _med(sb(arm["runs"], op, "latency_p50_ms"))
                    row[f"{tag}_median_tp_success"] = _med(tp_s)
                    row[f"{tag}_median_p50_success"] = _med(
                        sb(arm["runs"], op, "latency_p50_ms", "success")
                    )
                if same_epoch:
                    members = zs["runs"] + ms["runs"]
                    mwu = mann_whitney_prompting(members, op, SOURCE_PROXY)
                    mwu_success = mann_whitney_prompting(
                        [run for run in members if run.success], op, SOURCE_PROXY
                    )
                    row.update(
                        {
                            "stage_b_tested": "yes (same epoch, interleaved)",
                            "mwu_U": mwu.statistic,
                            "mwu_p": mwu.p_value,
                            "mwu_groups": "/".join(str(size) for size in mwu.group_sizes),
                            "mwu_U_success": mwu_success.statistic,
                            "mwu_p_success": mwu_success.p_value,
                            "mwu_groups_success": "/".join(
                                str(size) for size in mwu_success.group_sizes
                            ),
                        }
                    )
                else:
                    row.update(
                        {
                            "stage_b_tested": "no (different epochs and measurement days)",
                            "mwu_U": None,
                            "mwu_p": None,
                            "mwu_groups": "",
                            "mwu_U_success": None,
                            "mwu_p_success": None,
                            "mwu_groups_success": "",
                        }
                    )
                rows.append(row)
    return rows


def haiku_price_rates(runs: Sequence[RunRecord]) -> tuple[float, float, float] | None:
    token_counts: list[list[int]] = []
    prices: list[float] = []
    for run in runs:
        if run.model != "claude_haiku" or run.is_fixture or (run.llm_calls or 1) != 1:
            continue
        price = (run.usage or {}).get("cli_list_price_usd")
        if not price:
            continue
        usage = run.usage
        prompt = int(usage.get("prompt_token_count") or 0)
        cached = int(usage.get("cached_content_token_count") or 0)
        output = int(usage.get("candidates_token_count") or 0)
        token_counts.append([prompt - cached, cached, output])
        prices.append(float(price))
    if len(token_counts) < 5:
        return None
    coef, *_ = np.linalg.lstsq(np.array(token_counts, float), np.array(prices, float), rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def run_cost(r: RunRecord, config: Any) -> tuple[float | None, str]:
    usage = r.usage_total or r.usage
    if not usage:
        return None, "no usage"
    pricing = tr.pricing_for(r.model, config)
    if pricing is None:
        return None, "no pricing"
    basis = (
        "list price, not billed (subscription)"
        if r.model in tr.LIST_PRICE_OVERRIDES
        else "list price (models.yaml)"
    )
    return float(cost_of_usage(usage, pricing)[0]), basis


def cost_rows(arms: Sequence[dict[str, Any]], config: Any) -> list[dict[str, Any]]:
    rows = []
    for arm in arms:
        groups = {
            "L1": by_level(arm["runs"])["L1"],
            "L2-L5": [run for run in arm["runs"] if run.guidance_level != "L1"],
        }
        for name, members in groups.items():
            prompt_tokens: list[int] = []
            output_tokens: list[int] = []
            costs: list[float] = []
            calls: list[int] = []
            bases: Counter[str] = Counter()
            for run in members:
                usage = run.usage_total or run.usage
                if usage:
                    prompt_tokens.append(int(usage.get("prompt_token_count") or 0))
                    output_tokens.append(
                        int(usage.get("candidates_token_count") or 0)
                        + int(usage.get("thoughts_token_count") or 0)
                    )
                cost, basis = run_cost(run, config)
                bases[basis] += 1
                if cost is not None:
                    costs.append(cost)
                calls.append(run.llm_calls or 1)
            n_success = sum(1 for run in members if run.success)
            rows.append(
                {
                    "arm": arm["key"],
                    "model": MODEL_FULL[arm["model"]],
                    "epoch": arm["epoch"],
                    "levels": name,
                    "n": len(members),
                    "success": n_success,
                    "llm_calls_mean": float(np.mean(calls)) if calls else None,
                    "prompt_tokens_median": float(np.median(prompt_tokens))
                    if prompt_tokens
                    else None,
                    "output_tokens_median": float(np.median(output_tokens))
                    if output_tokens
                    else None,
                    "cost_usd_per_run": float(np.mean(costs)) if costs else None,
                    "cost_usd_total": float(sum(costs)) if costs else None,
                    "cost_usd_per_success": float(sum(costs)) / n_success
                    if (costs and n_success)
                    else None,
                    "cost_basis": "; ".join(f"{kind} x{count}" for kind, count in bases.items()),
                }
            )
    return rows


def l1_rows(arms: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for arm in arms:
        members = by_level(arm["runs"])["L1"]
        rejections: Counter[str] = Counter()
        for run in members:
            for rejection in run.api_rejections:
                rejections[rejection.split(" x")[0]] += 1
        outcomes = Counter(OUTCOME_MAP.get(run.failure_class, "other") for run in members)
        rows.append(
            {
                "arm": arm["key"],
                "epoch": arm["epoch"],
                "n": len(members),
                "success": sum(1 for run in members if run.success),
                "probe_calls_mean": float(np.mean([run.probe_calls or 0 for run in members]))
                if members
                else None,
                "outcomes": "; ".join(
                    f"{outcome} x{count}" for outcome, count in outcomes.most_common()
                ),
                "rejections_runs": "; ".join(
                    f"{code} x{count}" for code, count in rejections.most_common()
                ),
            }
        )
    return rows


def _epochs_label(arms: Sequence[dict[str, Any]], model: str) -> str:
    epochs = sorted({arm["epoch"] for arm in arms if arm["model"] == model})
    return "E" + ", E".join(map(str, epochs))


def fig_stage_a(arms: Sequence[dict[str, Any]], a_rows: Sequence[dict[str, Any]], out: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 3.9), sharex=True, sharey=True)
    x = np.arange(len(LEVELS))
    present = [model for model in MODELS if any(arm["model"] == model for arm in arms)]
    offsets = dict(zip(present, np.linspace(-0.12, 0.12, len(present)), strict=True))
    for i, dialect in enumerate(DIALECTS):
        for j, prompting in enumerate(PROMPTINGS):
            ax = axes[i][j]
            panel_arms = [
                arm for arm in arms if arm["dialect"] == dialect and arm["prompting"] == prompting
            ]
            for arm in panel_arms:
                ys = [cell(a_rows, arm["key"], lvl)["success"] for lvl in LEVELS]
                ax.plot(
                    x + offsets[arm["model"]],
                    ys,
                    color=MODEL_COLOR[arm["model"]],
                    lw=1.2,
                    marker="o",
                    markersize=4.2,
                    markeredgecolor=SURFACE,
                    zorder=3,
                )
            epochs = ", ".join(f"E{arm['epoch']}" for arm in panel_arms)
            ax.set_title(f"{dialect} · {PROMPT_LABEL[prompting]}  ({epochs})")
            ax.set_xticks(x, LEVELS)
            ax.set_ylim(-0.6, 10.8)
            ax.set_yticks([0, 5, 10])
            ax.tick_params(axis="x", length=0)
            ax.spines["bottom"].set_color(BASELINE)
        axes[i][0].set_ylabel("Successful runs (of 10)")
    for ax in axes[1]:
        ax.set_xlabel("Guidance level")
    handles = [
        Line2D(
            [],
            [],
            color=MODEL_COLOR[model],
            marker="o",
            markersize=4.2,
            lw=1.2,
            label=f"{MODEL_FULL[model]} ({_epochs_label(arms, model)})",
        )
        for model in present
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), handlelength=1.6
    )
    fig.subplots_adjust(wspace=0.08, hspace=0.42)
    return save(fig, out, "models_stage_a")


METRIC_NAME = {"tp": "throughput", "p50": "per-request latency p50"}
FIG_SUFFIX = {"tp": "", "p50": "_p50"}


def fig_stage_b(
    arms: Sequence[dict[str, Any]],
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
    dialect: str,
    metric: str,
    view: str,
    out: Path,
) -> str:
    d_arms = [arm for arm in arms if arm["dialect"] == dialect]
    rng = np.random.default_rng(tr.JITTER_SEED)
    height = 5.0
    fig, axes = plt.subplots(
        len(OPS), 2, figsize=(TEXT_WIDTH_IN, height), sharex=True, sharey="row"
    )
    any_failed = False
    for j, prompting in enumerate(PROMPTINGS):
        panel_arms = [
            arm
            for model in MODELS
            for arm in d_arms
            if arm["prompting"] == prompting and arm["model"] == model
        ]
        offsets = np.linspace(-0.27, 0.27, len(panel_arms)) if len(panel_arms) > 1 else [0.0]
        for i, op in enumerate(OPS):
            ax = axes[i][j]
            for arm, offset in zip(panel_arms, offsets, strict=True):
                color = MODEL_COLOR[arm["model"]]
                for k, lvl in enumerate(LEVELS[1:]):
                    b_cell = b_cells[(arm["key"], lvl, op)]
                    values, quartiles = tr.cell_view(b_cell, metric, view)
                    if not values or quartiles is None:
                        continue
                    y = np.asarray(values, dtype=float)
                    ok = (
                        np.asarray(b_cell["ok"], dtype=bool)
                        if view == "all"
                        else np.ones(len(y), dtype=bool)
                    )
                    xs = k + offset + rng.uniform(-0.05, 0.05, size=len(y))
                    ax.scatter(
                        xs[ok], y[ok], s=6, color=color, edgecolor="none", alpha=0.6, zorder=3
                    )
                    if (~ok).any():
                        any_failed = True
                        ax.scatter(
                            xs[~ok], y[~ok], s=14, marker="x", color=color, linewidth=0.8, zorder=3
                        )
                    ax.hlines(
                        quartiles[1],
                        k + offset - 0.08,
                        k + offset + 0.08,
                        color=INK,
                        lw=1.4,
                        zorder=4,
                    )
            ax.set_xticks(range(len(LEVELS) - 1), LEVELS[1:])
            ax.set_xlim(-0.55, len(LEVELS) - 1.45)
            ax.tick_params(axis="x", length=0)
            ax.spines["bottom"].set_color(BASELINE)
            if i == 0:
                ax.set_title(
                    f"{dialect} · {PROMPT_LABEL[prompting]}  ({', '.join('E' + str(arm['epoch']) for arm in panel_arms)})"
                )
            if j == 0:
                unit = "ops/s" if metric == "tp" else "p50, ms"
                ax.set_ylabel(f"{op} ({unit})")
        axes[-1][j].set_xlabel("Guidance level")
    for row in axes:
        row[0].set_ylim(bottom=0, top=row[0].get_ylim()[1])
    present = [model for model in MODELS if any(arm["model"] == model for arm in d_arms)]
    handles: list[Any] = [
        Line2D(
            [],
            [],
            marker="o",
            color=MODEL_COLOR[model],
            linestyle="",
            markersize=4.5,
            label=f"{MODEL_FULL[model]} run",
        )
        for model in present
    ]
    if any_failed:
        handles.append(
            Line2D(
                [],
                [],
                marker="x",
                color=INK_2,
                linestyle="",
                markersize=4.5,
                label="measured run that failed Stage A (kept)",
            )
        )
    handles.append(Line2D([], [], color=INK, lw=1.4, label=f"median, {VIEW_TEXT[view]}"))
    fig.subplots_adjust(wspace=0.08, hspace=0.3, top=0.9, bottom=0.5 / height)
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=1.4,
    )
    tr._header(fig, tr.figure_view_label(b_cells, d_arms, view))
    return save(fig, out, f"models_stage_b{FIG_SUFFIX[metric]}_{dialect.lower()}_{view}")


def fig_delta(
    arms: Sequence[dict[str, Any]],
    b_rows: Sequence[dict[str, Any]],
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
    metric: str,
    view: str,
    out: Path,
) -> str:
    with_ref = [arm for arm in arms if arm["ref"]]
    suffix = "" if view == "all" else "_success"
    height = 1.0 + 0.32 * len(with_ref)
    fig, axes = plt.subplots(1, len(OPS), figsize=(TEXT_WIDTH_IN, height), sharey=True)
    any_dagger = False
    for ax, op in zip(axes, OPS, strict=True):
        for i, arm in enumerate(with_ref):
            for k, lvl in enumerate(LEVELS[1:]):
                row = b_row(b_rows, arm["key"], lvl, op)
                med = row.get(f"delta_{metric}_med{suffix}")
                if med is None:
                    continue
                y = i + (k - 1.5) * 0.14
                color = RAMP_5[k + 1]
                ax.plot(
                    [row[f"delta_{metric}_q1{suffix}"], row[f"delta_{metric}_q3{suffix}"]],
                    [y, y],
                    color=color,
                    lw=0.9,
                    alpha=0.7,
                    zorder=2,
                )
                ax.scatter(
                    [med], [y], s=14, color=color, edgecolor=SURFACE, linewidth=0.5, zorder=3
                )
                if view == "all" and row["n_stage_a_failed"]:
                    any_dagger = True
                    ax.annotate(
                        "†",
                        (row[f"delta_{metric}_q3"], y),
                        xytext=(2, 0),
                        textcoords="offset points",
                        fontsize=6,
                        color=INK,
                        va="center",
                        ha="left",
                        zorder=4,
                    )
        ax.axvline(0, color=INK_2, lw=0.9, ls=(0, (4, 2)), zorder=1)
        ax.set_title(op)
        ax.grid(False)
        ax.xaxis.grid(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["bottom"].set_color(BASELINE)
    axes[1].set_xlabel(f"Δ {METRIC_NAME[metric]} vs same-day reference median (%)")
    labels = []
    for arm in with_ref:
        n_fail = sum(
            b_row(b_rows, arm["key"], lvl, OPS[0])["n_stage_a_failed"] for lvl in LEVELS[1:]
        )
        dagger = f" †{n_fail}" if (view == "all" and n_fail) else ""
        labels.append(f"{arm['label']}{dagger}")
    axes[0].set_yticks(range(len(with_ref)), labels)
    for tick_label, arm in zip(axes[0].get_yticklabels(), with_ref, strict=True):
        tick_label.set_color(MODEL_COLOR[arm["model"]])
    axes[0].invert_yaxis()
    handles: list[Any] = [
        Line2D(
            [],
            [],
            marker="o",
            color=RAMP_5[k + 1],
            linestyle="",
            markersize=4.5,
            label=f"{lvl} median (line: IQR)",
        )
        for k, lvl in enumerate(LEVELS[1:])
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
                markersize=5,
                label="cell with measured runs failing Stage A",
            )
        )
        handles.append(
            Line2D([], [], color=INK_2, lw=0, label="†k in label: k such runs in the arm (kept)")
        )
    fig.subplots_adjust(wspace=0.14, top=1 - 0.45 / height, bottom=0.5 / height)
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=1.0,
    )
    tr._header(fig, tr.figure_view_label(b_cells, with_ref, view, marker="†"))
    return save(fig, out, f"models_stage_b_delta{FIG_SUFFIX[metric]}_{view}")


def fig_cost(arms: Sequence[dict[str, Any]], c_rows: Sequence[dict[str, Any]], out: Path) -> str:
    rows = [row for row in c_rows if row["levels"] == "L2-L5"]
    arm_by_key = {arm["key"]: arm for arm in arms}
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 0.9 + 0.24 * len(rows)), sharey=True)
    for i, row in enumerate(rows):
        color = MODEL_COLOR[arm_by_key[row["arm"]]["model"]]
        if row["output_tokens_median"] is not None:
            axes[0].barh(i, row["output_tokens_median"], color=color, height=0.62)
        if row["cost_usd_per_run"] is not None:
            axes[1].barh(i, row["cost_usd_per_run"] * 100, color=color, height=0.62)
            axes[1].text(
                row["cost_usd_per_run"] * 100,
                i,
                f" {row['cost_usd_per_run'] * 100:.1f}",
                va="center",
                fontsize=6.5,
                color=INK_2,
            )
    axes[0].set_yticks(range(len(rows)), [arm_by_key[row["arm"]]["label"] for row in rows])
    for tick_label, row in zip(axes[0].get_yticklabels(), rows, strict=True):
        tick_label.set_color(MODEL_COLOR[arm_by_key[row["arm"]]["model"]])
    axes[0].invert_yaxis()
    axes[1].set_xlabel("List price per run, mean (US cents)")
    xmax = max((row["cost_usd_per_run"] or 0) * 100 for row in rows)
    axes[1].set_xlim(0, xmax * 1.18)
    for ax in axes:
        ax.grid(False)
        ax.xaxis.grid(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["bottom"].set_color(BASELINE)
    axes[0].xaxis.set_major_formatter(lambda x, _pos: f"{x / 1000:.1f}".rstrip("0").rstrip("."))
    axes[0].set_xlabel("Output tokens per run, median, thousands (incl. reasoning)")
    fig.subplots_adjust(wspace=0.18)
    return save(fig, out, "models_cost_tokens")


def _model_rules(rows_per_arm: Sequence[int], arms: Sequence[dict[str, Any]]) -> list[int]:
    rules: list[int] = []
    last_row = -1
    for i, (n_rows, arm) in enumerate(zip(rows_per_arm, arms, strict=True)):
        last_row += n_rows
        if i + 1 < len(arms) and arms[i + 1]["model"] != arm["model"]:
            rules.append(last_row)
    return rules


def tex_stage_a(
    arms: Sequence[dict[str, Any]],
    a_rows: Sequence[dict[str, Any]],
    fisher_rows: Sequence[dict[str, Any]],
) -> str:
    body = []
    for arm in arms:
        cells = []
        for lvl in LEVELS:
            stage_a_cell = cell(a_rows, arm["key"], lvl)
            significant = any(
                (fisher_row["arm_a"] == arm["key"] or fisher_row["arm_b"] == arm["key"])
                and fisher_row["level"] == lvl
                and fisher_row["fisher_holm_5_levels"] is not None
                and fisher_row["fisher_holm_5_levels"] < 0.05
                for fisher_row in fisher_rows
            )
            cells.append(
                f"{stage_a_cell['success']}/{stage_a_cell['n']}" + ("$^{*}$" if significant else "")
            )
        body.append([arm["short_tex"], *cells])
    return latex_table(
        caption="Stage A success per model, prompting strategy and dialect (main matrix: gemini-3.5-flash, claude-haiku-5-5, glm-5.3; 12 arms, N = 10 per cell).",
        label="tab:models-stage-a",
        header=["Arm", *LEVELS],
        rows=body,
        align="l" + "c" * len(LEVELS),
        note=tex(
            "Successful runs out of N per cell. * = differs from another model of the same dialect and prompting strategy (Fisher exact for each of the three model pairs, Holm over the five levels of a pair, p < 0.05; not corrected across pairs). flash zero-shot E8/E9 ran with syntax-only feedback, all other arms with execution feedback. "
            + tr.ARM_LEGEND
            + " "
            + PIPELINE_NOTE
        ),
        midrules_after=_model_rules([1] * len(arms), arms),
    )


def _sb_tex(row: dict[str, Any], view: str) -> str:
    suffix = "" if view == "all" else "_s"
    if row.get(f"tp{suffix}_med") is None:
        return "--"
    text = (
        f"{row[f'tp{suffix}_med']:.1f} ({row[f'tp{suffix}_q1']:.0f}--{row[f'tp{suffix}_q3']:.0f})"
        f"/{row[f'p50{suffix}_med']:.1f}"
    )
    if view == "all" and row["n_stage_a_failed"]:
        text += f"$^{{\\dagger {row['n_stage_a_failed']}}}$"
    return text


def tex_stage_b(
    arms: Sequence[dict[str, Any]], b_rows: Sequence[dict[str, Any]], dialect: str, days: str
) -> str:
    d_arms = [arm for arm in arms if arm["dialect"] == dialect]
    body: list[list[str]] = []
    per_arm: list[int] = []
    for arm in d_arms:
        start = len(body)
        for op in OPS:
            level_rows = [b_row(b_rows, arm["key"], lvl, op) for lvl in LEVELS[1:]]
            body.append(
                [
                    arm["short_tex"] if op == OPS[0] else "",
                    op,
                    *[_sb_tex(row, "all") for row in level_rows],
                ]
            )
            if any(row["n_stage_a_failed"] for row in level_rows):
                body.append(
                    [
                        "",
                        r"\quad succ.",
                        *[
                            _sb_tex(row, "success") if row["n_stage_a_failed"] else "$=$"
                            for row in level_rows
                        ],
                    ]
                )
        per_arm.append(len(body) - start)
    d_rows = [row for row in b_rows if row["dialect"] == dialect]
    n_fail_runs = sum(row["n_stage_a_failed"] for row in d_rows if row["op"] == OPS[0])
    n_b_failed = sum(row["n_stage_b_failed"] for row in d_rows)
    note = tex(
        f"Main rows: every measured run (view all measured runs), i.e. every run that executed and produced a value for the operation, including runs that executed but failed Stage A (DAGGER k: k such runs in the cell; {n_fail_runs} such runs in this dialect). Rows succ.: the same cell over the Stage-A successes only (view Stage-A successes only; EQ where the cell has no Stage-A failure). Stage-B failures (Stage-A successes without a value for the operation): {n_b_failed}. Throughput and latency are indicators read together; none is primary. Measurement days: {days}; absolute values from different days are not compared across models (see the deviation from the same-day reference). "
        + tr.ARM_LEGEND
    )
    note = note.replace("DAGGER k", r"$^{\dagger k}$").replace("EQ where", "$=$ where")
    return latex_table(
        caption=f"Stage B per model, {dialect}: median (interquartile range) of the proxy throughput in ops/s / median latency p50 in ms per arm, operation and guidance level, over every measured run and over the Stage-A successes only.",
        label=f"tab:models-stage-b-{dialect.lower()}",
        header=["Arm", "Op.", *LEVELS[1:]],
        rows=body,
        align="llcccc",
        size=r"\scriptsize",
        tabcolsep="3pt",
        note=note,
        midrules_after=_model_rules(per_arm, d_arms),
    )


def _delta_tex(row: dict[str, Any], view: str) -> str:
    suffix = "" if view == "all" else "_success"
    tp, p50 = row.get(f"delta_tp_med{suffix}"), row.get(f"delta_p50_med{suffix}")
    if tp is None:
        return "--"
    text = f"{tp:+.0f}/{p50:+.0f}"
    if view == "all" and row["n_stage_a_failed"]:
        text += f"$^{{\\dagger {row['n_stage_a_failed']}}}$"
    return text


def tex_delta(arms: Sequence[dict[str, Any]], b_rows: Sequence[dict[str, Any]]) -> str:
    with_ref = [arm for arm in arms if arm["ref"]]
    body: list[list[str]] = []
    per_arm: list[int] = []
    for arm in with_ref:
        start = len(body)
        for op in OPS:
            level_rows = [b_row(b_rows, arm["key"], lvl, op) for lvl in LEVELS[1:]]
            body.append(
                [
                    arm["short_tex"] if op == OPS[0] else "",
                    op,
                    *[_delta_tex(row, "all") for row in level_rows],
                ]
            )
            if any(row["n_stage_a_failed"] for row in level_rows):
                body.append(
                    [
                        "",
                        r"\quad succ.",
                        *[
                            _delta_tex(row, "success") if row["n_stage_a_failed"] else "$=$"
                            for row in level_rows
                        ],
                    ]
                )
        per_arm.append(len(body) - start)
    note = tex(
        "Δ = (median of the arm - median of the same-day reference) / median of the reference, in %, for throughput / latency p50. Reference: bddbench gold series of the same dialect measured on the same day (flash multi-shot E12/E13: gold E12/E13 on 13 Sep 2026; haiku E14/E15 and glm E16/E17: gold E16/E17 on 9 Oct 2026). flash zero-shot E8/E9 (28 Aug 2026) have no same-day reference and are left out. Main rows: every measured run (DAGGER k: k runs failing Stage A in the cell); rows succ.: Stage-A successes only (EQ where identical). Descriptive; no test across models. "
        + tr.ARM_LEGEND
    )
    note = note.replace("DAGGER k", r"$^{\dagger k}$").replace("EQ where", "$=$ where")
    return latex_table(
        caption="Deviation of throughput / latency p50 from the same-day reference median (\\%) per arm, operation and guidance level, over every measured run and over the Stage-A successes only.",
        label="tab:models-delta",
        header=["Arm", "Op.", *LEVELS[1:]],
        rows=body,
        align="llcccc",
        size=r"\scriptsize",
        tabcolsep="4pt",
        note=note,
        midrules_after=_model_rules(per_arm, with_ref),
    )


def tex_cost(arms: Sequence[dict[str, Any]], c_rows: Sequence[dict[str, Any]]) -> str:
    arm_by_key = {arm["key"]: arm for arm in arms}
    body = []
    for row in c_rows:
        body.append(
            [
                arm_by_key[row["arm"]]["short_tex"],
                row["levels"],
                fmt(row["llm_calls_mean"], 1),
                fmt(row["prompt_tokens_median"], 0),
                fmt(row["output_tokens_median"], 0),
                fmt(
                    row["cost_usd_per_run"] * 100 if row["cost_usd_per_run"] is not None else None,
                    1,
                ),
                fmt(
                    row["cost_usd_per_success"] * 100
                    if row["cost_usd_per_success"] is not None
                    else None,
                    1,
                ),
            ]
        )
    return latex_table(
        caption="Tokens and list price per run by model (L1 with probing separate from L2--L5).",
        label="tab:models-cost",
        header=["Arm", "Levels", "Calls", "Prompt tok.", "Output tok.", "ct/run", "ct/success"],
        rows=body,
        align="llrrrrr",
        size=r"\footnotesize",
        tabcolsep="4pt",
        note=tex(
            "Medians of tokens per run (all calls of a run), mean list price in US cents. Output tokens include reasoning tokens. gemini-3.5-flash and glm-5.3 priced with the list prices in config/models.yaml; claude-haiku-5-5 priced at Anthropic's list price (0.10 / 0.50 USD per MTok, cache hits 0.01; verified 9 Oct 2026) but not billed: it ran on a subscription through the Claude Code CLI. The total_cost_usd the CLI reports is not used (it corresponds to a 25 USD/MTok output rate). "
            + tr.ARM_LEGEND
        ),
        midrules_after=_model_rules([2] * len(arms), arms),
    )


def main() -> None:
    config, runs = tr.load_runs()
    out = config.paths.results_reports_dir / "thesis_results" / "models"
    fig_dir, tab_dir, tex_dir = out / "fig", out / "tables", out / "latex"
    for directory in (fig_dir, tab_dir, tex_dir):
        directory.mkdir(parents=True, exist_ok=True)
    removed = []
    for rel in STALE_OUTPUTS:
        stale_path = out / rel
        if stale_path.exists():
            stale_path.unlink()
            removed.append(rel)
    tr._style()

    in_progress: list[str] = []
    arms = build_arms(runs, in_progress)
    assert all(arm["model"] in MODELS for arm in arms)
    reference = tr.reference_stats(runs)
    b_cells = tr.stage_b_cells(runs)
    d_rows = tr.delta_rows(b_cells, reference, [arm for arm in arms if arm["ref"]])
    a_rows = stage_a_rows(arms)
    f_rows = stage_a_fisher(arms, a_rows)
    b_rows = stage_b_rows(arms, b_cells, d_rows)
    bt_rows = tr.stage_b_tests(runs, arms)
    if len(arms) == len(tr.ARMS):
        tr.add_global_holm(bt_rows)
    pt_rows = prompting_tests(arms)
    rates = haiku_price_rates(runs)
    c_rows = cost_rows(arms, config)
    l1 = l1_rows(arms)
    days = measurement_days(arms)

    write_csv(
        tab_dir / "models_stage_a.csv",
        list(a_rows[0].keys()),
        [list(row.values()) for row in a_rows],
    )
    write_csv(
        tab_dir / "models_stage_a_fisher.csv",
        list(f_rows[0].keys()),
        [list(row.values()) for row in f_rows],
    )
    b_cols = list(b_rows[0].keys()) + [col for col in DELTA_COLS if col not in b_rows[0]]
    write_csv(
        tab_dir / "models_stage_b.csv",
        b_cols,
        [[row.get(col) for col in b_cols] for row in b_rows],
    )
    bt_cols = list(dict.fromkeys(key for row in bt_rows for key in row))
    write_csv(
        tab_dir / "models_stage_b_tests.csv",
        bt_cols,
        [[row.get(col) for col in bt_cols] for row in bt_rows],
    )
    write_csv(
        tab_dir / "models_prompting_tests.csv",
        list(pt_rows[0].keys()),
        [list(row.values()) for row in pt_rows],
    )
    write_csv(
        tab_dir / "models_cost.csv", list(c_rows[0].keys()), [list(row.values()) for row in c_rows]
    )
    write_csv(tab_dir / "models_l1.csv", list(l1[0].keys()), [list(row.values()) for row in l1])
    write_csv(
        tab_dir / "models_reference.csv",
        [
            "arm",
            "epoch",
            "arm_measured_on",
            "reference",
            "reference_basis",
            "gold_runs",
            *[f"ref_{op}_tp_med" for op in OPS],
            *[f"ref_{op}_p50_med" for op in OPS],
        ],
        [
            [
                arm["key"],
                arm["epoch"],
                ", ".join(arm["dates"]),
                arm["ref"] or "",
                reference_basis(arm, reference),
                reference[arm["ref"]]["write"]["n"] if arm["ref"] else 0,
                *[reference[arm["ref"]][op]["tp_med"] if arm["ref"] else None for op in OPS],
                *[reference[arm["ref"]][op]["p50_med"] if arm["ref"] else None for op in OPS],
            ]
            for arm in arms
        ],
    )

    fig_stage_a(arms, a_rows, fig_dir)
    figs: list[tuple[str, str, str]] = [
        (
            "models_stage_a",
            "fig:models-stage-a",
            f"Stage A success per guidance level for the three models of the comparison (gemini-3.5-flash, claude-haiku-5-5, glm-5.3), by dialect and prompting strategy; N = 10 runs per point, epochs in the panel titles. flash zero-shot E8/E9 ran with syntax-only feedback, all other arms with execution feedback. {PIPELINE_NOTE}",
        )
    ]
    view_cap = {
        "all": "View: all measured runs; runs that executed and were measured but failed Stage A are kept and drawn as x in the model colour, and the median is taken over all of them.",
        "success": "View: Stage-A successes only (the runs failing Stage A are left out entirely).",
    }
    for metric in ("tp", "p50"):
        for dialect in DIALECTS:
            d_arms = [arm for arm in arms if arm["dialect"] == dialect]
            for view in VIEWS:
                name = fig_stage_b(arms, b_cells, dialect, metric, view, fig_dir)
                figs.append(
                    (
                        name,
                        f"fig:{name.replace('_', '-')}",
                        f"Stage B proxy {METRIC_NAME[metric]} per run, {dialect}, guidance levels L2–L5, the three models side by side per prompting strategy (black bar: median). {view_cap[view]} Measured on: {measurement_days(d_arms)}; absolute values from different days are not compared across models. Descriptive. {PIPELINE_NOTE}",
                    )
                )
    for metric in ("tp", "p50"):
        for view in VIEWS:
            name = fig_delta(arms, b_rows, b_cells, metric, view, fig_dir)
            dagger_note = (
                " A dagger marks cells containing measured runs that failed Stage A; †k after the arm label gives their number in the arm."
                if view == "all"
                else ""
            )
            figs.append(
                (
                    name,
                    f"fig:{name.replace('_', '-')}",
                    f"Deviation Δ of the {METRIC_NAME[metric]} from the bddbench reference median measured on the same day (gold E12/E13 on 13 Sep 2026 for flash multi-shot E12/E13; gold E16/E17 on 9 Oct 2026 for haiku E14/E15 and glm E16/E17), per arm and level: marker = median, line = IQR. {view_cap[view].split(';')[0].split(' (')[0]}.{dagger_note} flash zero-shot E8/E9 (28 Aug 2026) are left out: no reference was measured that day. Descriptive, no test across models. {PIPELINE_NOTE}",
                )
            )
    fig_cost(arms, c_rows, fig_dir)
    figs.append(
        (
            "models_cost_tokens",
            "fig:models-cost",
            f"Output tokens (median, incl. reasoning) and list price per run (mean) on L2–L5, by arm and model. claude-haiku-5-5 is priced at list price but was not billed (subscription). {PIPELINE_NOTE}",
        )
    )

    (tex_dir / "models_tables.tex").write_text(
        "\n".join(
            [
                tex_stage_a(arms, a_rows, f_rows),
                *[
                    tex_stage_b(
                        arms,
                        b_rows,
                        dialect,
                        measurement_days([arm for arm in arms if arm["dialect"] == dialect]),
                    )
                    for dialect in DIALECTS
                ],
                tex_delta(arms, b_rows),
                tex_cost(arms, c_rows),
            ]
        ),
        encoding="utf-8",
    )
    (tex_dir / "models_figures.tex").write_text(
        tr.tex_figures(
            [
                (name, label, tex(caption).replace("†", r"$\dagger$"))
                for name, label, caption in figs
            ]
        ),
        encoding="utf-8",
    )

    _readme(
        out,
        arms,
        reference,
        a_rows,
        f_rows,
        b_rows,
        b_cells,
        bt_rows,
        pt_rows,
        c_rows,
        l1,
        rates,
        figs,
        in_progress,
        days,
    )
    if removed:
        print("Removed stale outputs: " + ", ".join(removed))
    print(f"written: {out}")


def _both(
    all_value: float | None,
    success_value: float | None,
    digits: int,
    differs: bool,
    unit: str = "",
) -> str:
    text = f"{fmt(all_value, digits)}{unit}"
    return text + (f" (s {fmt(success_value, digits)}{unit})" if differs else "")


def _range(values: Sequence[float | None]) -> str:
    present = [value for value in values if value is not None]
    return f"{min(present):+.0f} to {max(present):+.0f} %" if present else "--"


def _readme_stage_b(
    arms: Sequence[dict[str, Any]],
    b_rows: Sequence[dict[str, Any]],
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
    bt_rows: Sequence[dict[str, Any]],
    pt_rows: Sequence[dict[str, Any]],
) -> list[str]:
    n_b_failed = sum(row["n_stage_b_failed"] for row in b_rows)
    lines = [
        "",
        "## Stage B: two views, no primary metric",
        "",
        "Every Stage-B figure exists twice, `<name>_all` and `<name>_success`, each with a view label inside the figure giving the counts of the runs it shows. Tables and CSVs carry both views side by side. The two views rank equally.",
        "",
        "- **View all measured runs:** every run that executed and produced a proxy value for the operation, including runs that executed but failed Stage A. Those are kept and marked: x in the model colour in the per-run figures, † in the Δ figures (†k after the arm label), †k in the tables. They enter medians, IQRs, Δ and tests of this view.",
        "- **View Stage-A successes only:** the same numbers without those runs (CSV columns `*_s_*`, `*_success`; `succ.` rows in the LaTeX tables).",
        f"- **Failure stages:** Stage-A failure = run not successful (no syntax/execution, or executed but non-conformant workload); such a run is still measured if it executed. Stage-B failure = Stage-A success without a value for the operation (NA, never 0), column `n_stage_b_failed`: {n_b_failed} over all cells and operations.",
        "- **No primary metric:** throughput, per-request latency p50, Δ to the same-day reference and the tests are indicators read together.",
        "",
        "Counts (distinct runs, L2-L5, union over the three operations; same counting as the figure headers):",
        "",
        "| Set | All measured | Failed Stage A | Stage-A successes |",
        "|---|---|---|---|",
    ]
    sets = [
        (f"{dialect}, 6 arms", [arm for arm in arms if arm["dialect"] == dialect])
        for dialect in DIALECTS
    ]
    sets += [
        (
            f"Δ figures/table, {len([arm for arm in arms if arm['ref']])} arms with same-day reference",
            [arm for arm in arms if arm["ref"]],
        ),
        ("all 12 arms", list(arms)),
    ]
    for name, set_arms in sets:
        n, n_failed = tr.view_counts(b_cells, set_arms)
        lines.append(f"| {name} | {n} | {n_failed} | {n - n_failed} |")
    lines += [
        "",
        "### Spread of Δ to the same-day reference (median per cell, L2-L5, arms with a same-day reference)",
        "",
        "| Model | Op. | Δ throughput, all measured | Δ throughput, successes only | Δ latency p50, all measured | Δ latency p50, successes only |",
        "|---|---|---|---|---|---|",
    ]
    for model in MODELS:
        keys = {arm["key"] for arm in arms if arm["model"] == model and arm["ref"]}
        for op in OPS:
            op_rows = [
                row
                for row in b_rows
                if row["arm"] in keys and row["op"] == op and "delta_tp_med" in row
            ]
            lines.append(
                f"| {MODEL_LABEL[model]} | {op} | {_range([row['delta_tp_med'] for row in op_rows])} | {_range([row.get('delta_tp_med_success') for row in op_rows])} "
                f"| {_range([row['delta_p50_med'] for row in op_rows])} | {_range([row.get('delta_p50_med_success') for row in op_rows])} |"
            )
    lines += [
        "",
        "### Per arm: n measured (†Stage-A failed), median throughput ops/s, median latency p50 ms, Δ throughput, Δ latency p50 for L2, L3, L4, L5",
        "",
        "`(s x)` = value over the Stage-A successes only, given wherever the cell contains Stage-A-failed runs.",
        "",
    ]
    for arm in arms:
        key = arm["key"]
        for op in OPS:
            level_rows = [b_row(b_rows, key, lvl, op) for lvl in LEVELS[1:]]
            has_failed = [bool(row["n_stage_a_failed"]) for row in level_rows]
            n_txt = ", ".join(
                f"{row['n']}" + (f"†{row['n_stage_a_failed']}" if row["n_stage_a_failed"] else "")
                for row in level_rows
            )
            tp_txt = ", ".join(
                _both(row.get("tp_med"), row.get("tp_s_med"), 1, differs)
                for row, differs in zip(level_rows, has_failed, strict=True)
            )
            p50_txt = ", ".join(
                _both(row.get("p50_med"), row.get("p50_s_med"), 1, differs)
                for row, differs in zip(level_rows, has_failed, strict=True)
            )
            if arm["ref"]:
                delta_tp_txt = ", ".join(
                    _both(
                        row.get("delta_tp_med"), row.get("delta_tp_med_success"), 0, differs, " %"
                    )
                    for row, differs in zip(level_rows, has_failed, strict=True)
                )
                delta_p50_txt = ", ".join(
                    _both(
                        row.get("delta_p50_med"), row.get("delta_p50_med_success"), 0, differs, " %"
                    )
                    for row, differs in zip(level_rows, has_failed, strict=True)
                )
                delta = f" | Δ tp: {delta_tp_txt} | Δ p50: {delta_p50_txt}"
            else:
                delta = " | no same-day reference"
            lines.append(
                f"- {key} (E{arm['epoch']}) {op}: n {n_txt} | tp: {tp_txt} | p50: {p50_txt}{delta}"
            )
    holm_all = "rho_holm_all" in bt_rows[0]
    lines += [
        "",
        "### Kruskal-Wallis / Spearman level vs throughput per arm (models_stage_b_tests.csv)",
        "",
        "Same function and numbers as thesis_results.py (`stage_b_tests`): L2-L5, Holm within arm x operation (m = 2)"
        + (
            ", plus Holm over all 36 Spearman tests per view (`rho_holm_all`, `rho_success_holm_all`)"
            if holm_all
            else ""
        )
        + ". All measured runs, then Stage-A successes only.",
        "",
    ]
    for test in bt_rows:
        n_removed = (test["n"] or 0) - (test["n_success"] or 0)
        holm_txt = f", Holm(36) {fmt_p_plain(test['rho_holm_all'])}" if holm_all else ""
        holm_success_txt = (
            f", Holm(36) {fmt_p_plain(test['rho_success_holm_all'])}" if holm_all else ""
        )
        success_txt = (
            f" | successes only (n = {test['n_success']}, {n_removed} removed): KW p = {fmt_p_plain(test['kw_p_success'])}, "
            f"ρ = {fmt(test['rho_success'], 2)} [{fmt(test['rho_success_ci_low'], 2)}, {fmt(test['rho_success_ci_high'], 2)}], p = {fmt_p_plain(test['rho_success_p'])}{holm_success_txt}"
            if n_removed
            else " | identical for successes only"
        )
        lines.append(
            f"- {test['arm']} {test['op']} (n = {test['n']}): KW p = {fmt_p_plain(test['kw_p'])}, ρ = {fmt(test['rho'], 2)} [{fmt(test['rho_ci_low'], 2)}, {fmt(test['rho_ci_high'], 2)}], p = {fmt_p_plain(test['rho_p'])}{holm_txt}{success_txt}"
        )
    lines += [
        "",
        "### Zero-shot vs multi-shot per model (models_prompting_tests.csv)",
        "",
        "Stage A: Fisher exact over all 50 runs per strategy (uncorrected). Stage B: Mann-Whitney only where both strategies ran interleaved in one epoch (haiku E14/E15, glm E16/E17); flash zero- and multi-shot are different epochs measured on different days (E8/E9 on 28 Aug, E12/E13 on 13 Sep 2026), so their Stage B is given descriptively only.",
        "",
    ]
    for row in pt_rows:
        n_removed = row["zs_n_stage_a_failed"] + row["ms_n_stage_a_failed"]
        epochs = f"E{row['epoch_zs']}" + (
            "" if row["epoch_zs"] == row["epoch_ms"] else f" vs E{row['epoch_ms']}"
        )
        test_txt = (
            f", MWU p = {fmt_p_plain(row['mwu_p'])}" if row["mwu_p"] is not None else ", not tested"
        )
        test_success_txt = (
            f", MWU p = {fmt_p_plain(row['mwu_p_success'])}"
            if row["mwu_p_success"] is not None
            else ""
        )
        success_txt = (
            f" | successes only: tp {fmt(row['zs_median_tp_success'])} vs {fmt(row['ms_median_tp_success'])}, "
            f"p50 {fmt(row['zs_median_p50_success'])} vs {fmt(row['ms_median_p50_success'])}{test_success_txt} ({n_removed} removed)"
            if n_removed
            else ""
        )
        lines.append(
            f"- {row['model']} {row['dialect']} {epochs} {row['op']}: Stage A {row['stage_a_zs']} vs {row['stage_a_ms']} (Fisher p = {fmt_p_plain(row['stage_a_fisher_p'])}); "
            f"all measured (n {row['zs_n_measured']}/{row['ms_n_measured']}): tp {fmt(row['zs_median_tp'])} vs {fmt(row['ms_median_tp'])}, "
            f"p50 {fmt(row['zs_median_p50'])} vs {fmt(row['ms_median_p50'])} ms{test_txt}{success_txt}"
        )
    return lines


def _readme(
    out: Path,
    arms: Sequence[dict[str, Any]],
    reference: dict[str, dict[str, dict[str, Any]]],
    a_rows: Sequence[dict[str, Any]],
    f_rows: Sequence[dict[str, Any]],
    b_rows: Sequence[dict[str, Any]],
    b_cells: dict[tuple[str, str, str], dict[str, Any]],
    bt_rows: Sequence[dict[str, Any]],
    pt_rows: Sequence[dict[str, Any]],
    c_rows: Sequence[dict[str, Any]],
    l1: Sequence[dict[str, Any]],
    rates: tuple[float, float, float] | None,
    figs: Sequence[tuple[str, str, str]],
    in_progress: Sequence[str],
    days: str,
) -> None:
    failed = stage_a_failed_measured(arms)
    lines = [
        "# Model comparison (thesis_results_models.py)",
        "",
        "Regenerate: `uv run python scripts/thesis_results_models.py` (from the project root).",
        "",
        "## Scope: one Gemini model",
        "",
        "The model comparison uses exactly three models: **gemini-3.5-flash**, **claude-haiku-5-5** and **glm-5.3**, i.e. the 12 main-matrix arms of `thesis_results.ARMS` (3 models x 2 promptings x 2 dialects, one epoch per arm, N = 10 per level, 600 runs).",
        "",
        "**gemini-3.5-flash-lite is not part of this comparison.** It was used only to develop and test the pipeline cheaply (epochs E6/E7 with syntax-only feedback, E10/E11, and its multi-shot arms in E12/E13). Its runs appear in no arm, test, figure or table of this package. They are reported only in the separately labelled pipeline-development outputs of thesis_results.py (`pipeline_*`, feedback-regime contrast E6/E7 vs E10/E11, test-retest pairs E6→E10 / E7→E11 as measurement-noise evidence).",
        "",
        f"Measurement days (evaluated_at): {days}. E8/E9 also contain 10 flash multi-shot L3 runs each that belong to no arm; they are not used here.",
        "",
        "## Arms",
        "",
        "Complete arms only (N = 10 on every level). Incomplete: "
        + ("; ".join(in_progress) if in_progress else "none")
        + ".",
        "",
        "| Arm | Model | Access | Epoch | Measured on | Feedback (L4/L5) | Runs | Same-day reference |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        lines.append(
            f"| {arm['key']} | {MODEL_FULL[arm['model']]} | {ACCESS[arm['model']]} | E{arm['epoch']} | {', '.join(arm['dates'])} | {arm['regime']} | {len(arm['runs'])} | {reference_basis(arm, reference)} |"
        )
    lines += [
        "",
        "## Stage A (successful runs of N)",
        "",
        "| Arm | " + " | ".join(LEVELS) + " |",
        "|---|" + "---|" * len(LEVELS),
    ]
    for arm in arms:
        lines.append(
            f"| {arm['key']} (E{arm['epoch']}) | "
            + " | ".join(
                f"{cell(a_rows, arm['key'], lvl)['success']}/{cell(a_rows, arm['key'], lvl)['n']}"
                for lvl in LEVELS
            )
            + " |"
        )
    significant = [
        fisher_row
        for fisher_row in f_rows
        if fisher_row["fisher_holm_5_levels"] is not None
        and fisher_row["fisher_holm_5_levels"] < 0.05
    ]
    n_pairs = len(
        {
            (fisher_row["dialect"], fisher_row["prompting"], fisher_row["pair"])
            for fisher_row in f_rows
        }
    )
    lines += [
        "",
        f"Fisher exact, three model pairs (flash vs haiku, flash vs glm, haiku vs glm) within one dialect and prompting strategy ({n_pairs} pair families x 5 levels = {len(f_rows)} tests), Holm over the five levels of a pair, not corrected across pairs: {len(significant)} with p_Holm < 0.05.",
    ]
    for fisher_row in significant:
        lines.append(
            f"- {fisher_row['dialect']} {fisher_row['prompting']} {fisher_row['level']}: {fisher_row['arm_a']} {fisher_row['a_success']}/{fisher_row['a_n']} vs {fisher_row['arm_b']} {fisher_row['b_success']}/{fisher_row['b_n']}, p_Holm = {fmt_p(fisher_row['fisher_holm_5_levels'])}"
        )
    lines += [
        "",
        "Feedback rounds (L4/L5):",
        "",
        *[f"- {summary}" for summary in feedback_summary(a_rows)],
    ]
    lines += [
        "",
        "## L1 (no API material, probing)",
        "",
        "| Arm | Success | Probe calls (mean) | Outcomes | Rejections (runs) |",
        "|---|---|---|---|---|",
    ]
    for row in l1:
        lines.append(
            f"| {row['arm']} | {row['success']}/{row['n']} | {fmt(row['probe_calls_mean'])} | {row['outcomes']} | {row['rejections_runs']} |"
        )
    lines += _readme_stage_b(arms, b_rows, b_cells, bt_rows, pt_rows)
    lines += [
        "",
        "## Tokens and price",
        "",
        "| Arm | Levels | Calls | Prompt tok. | Output tok. | $/run | $/success | Basis |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in c_rows:
        lines.append(
            f"| {row['arm']} | {row['levels']} | {fmt(row['llm_calls_mean'])} | {fmt(row['prompt_tokens_median'], 0)} | {fmt(row['output_tokens_median'], 0)} | {fmt(row['cost_usd_per_run'], 4)} | {fmt(row['cost_usd_per_success'], 4)} | {row['cost_basis']} |"
        )
    if rates:
        lines.append(
            f"\nHaiku is priced at Anthropic's list price (0.10 / 0.50 USD per MTok, cache hits 0.01; checked 2026-10-09), not billed (subscription). The total_cost_usd reported by the Claude Code CLI is NOT used: fitted to the single-call runs it corresponds to uncached input {rates[0] * 1e6:.2f}, cached input {rates[1] * 1e6:.2f}, output {rates[2] * 1e6:.2f} USD per MTok, i.e. not Haiku's price."
        )
    lines += [
        "",
        "## Caveats",
        "",
        f"- {failed['n']} runs executed and were measured but failed Stage A (arms: {failed['arms']}; levels: {failed['levels']}; failure class: {failed['classes']}; points written by the write phase: {failed['write_points']}, against {failed['write_points_success']} in the Stage-A successes). They are kept and marked in the all-measured view and left out in the success view. Most wrote a multiple of the specified volume, so they are not the same workload as the reference; read both views for those cells.",
        "- No primary metric: throughput and per-request latency both depend on how the generated client drives the SUT (concurrency, batching); they, Δ and the tests are read together.",
        "- Delete is a single request per run and bimodal (about 25 vs. 75 ops/s) in temporal phases, independent of scenario structure and time range: an InfluxDB state effect, so delete deltas are weak evidence.",
        "- Stage B across epochs = across measurement days. Absolute values are not compared between models; only Δ to the reference of the same day. flash zero-shot E8/E9 (28 Aug 2026) have no same-day reference and are not in the Δ outputs. Test-retest noise of identical artefacts is in thesis_results (pipeline-development epochs, measurement evidence only).",
        "- flash zero-shot (E8/E9) ran with syntax-only feedback, all other arms with execution feedback; flash zero- vs multi-shot is therefore also a feedback-regime and measurement-day contrast, not a pure prompting contrast.",
        "- haiku and glm ran zero- and multi-shot interleaved in one epoch; flash zero- and multi-shot are separate epochs.",
        "- claude-haiku-5-5 via Claude Code CLI: no seed (sampling not reproducible), CLI-default effort, a short context block (date, account) is appended by the CLI; tokens include extended thinking.",
        "- glm-5.3 via OpenRouter pinned to Z.AI fp8, reasoning effort low (cannot be switched off), seed ignored by the provider.",
        "- All tests are per family as stated (Holm within a pair over 5 levels, within arm x operation, and over the 36 Spearman tests); there is no correction across the whole package. Δ and the Stage-B medians across models are descriptive.",
        "",
        "## Files",
        "",
        *[f"- fig/{name}.pdf/.png" for name, _, _ in figs],
        "- tables/models_stage_a.csv: Stage A per arm and level (outcome classes, first draft vs repaired)",
        "- tables/models_stage_a_fisher.csv: Fisher exact per model pair, dialect, prompting and level (Holm over 5 levels)",
        "- tables/models_stage_b.csv: Stage B per arm, level, operation in both views, with Δ to the same-day reference",
        "- tables/models_stage_b_tests.csv: Kruskal-Wallis / Spearman per arm and operation in both views",
        "- tables/models_prompting_tests.csv: zero- vs multi-shot per model and dialect",
        "- tables/models_cost.csv, models_l1.csv, models_reference.csv",
        "- latex/models_tables.tex (tab:models-stage-a, tab:models-stage-b-kestrel, tab:models-stage-b-tessera, tab:models-delta, tab:models-cost), latex/models_figures.tex (draft captions)",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
