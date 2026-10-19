from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_results as tr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from benchmark_agent.evaluation.cost import cost_of_usage

plt = tr.plt
NAME = "cost_access"
LEVELS = tr.LEVELS
FRONTIER_LEVELS = ("L2", "L3", "L4", "L5")
LEVEL_MARKER = {"L1": "x", "L2": "o", "L3": "s", "L4": "^", "L5": "D"}
MODEL_Y_OFFSET = {"gemini": -0.2, "claude_haiku": 0.0, "glm": 0.2}
PIPELINE_LABEL = "flash-lite (pipeline testing only, not compared)"
LITE_SEED_EXTRA_ARMS: list[dict[str, Any]] = [
    {
        "key": f"lite-ms-{tr.DIALECT_OF_EPOCH[e][0]}-extra-E{e}",
        "model": "gemini_lite",
        "prompting": "multi_shot",
        "dialect": tr.DIALECT_OF_EPOCH[e],
        "epoch": e,
    }
    for e in (6, 7, 10, 11)
]
FLASH_SEED_EXTRA_ARMS: list[dict[str, Any]] = [
    {
        "key": f"flash-ms-{tr.DIALECT_OF_EPOCH[e][0]}-extra-E{e}",
        "model": "gemini",
        "prompting": "multi_shot",
        "dialect": tr.DIALECT_OF_EPOCH[e],
        "epoch": e,
    }
    for e in (8, 9)
]


def _int(value: object) -> int:
    return int(value) if isinstance(value, int | float) else 0


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError, ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


def run_rows(
    runs: Sequence[Any], config: Any, arms: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    root = Path(config.paths.project_root)
    rows: list[dict[str, Any]] = []
    for arm in arms:
        pricing = tr.pricing_for(arm["model"], config)
        assert pricing is not None, arm["model"]
        for run in tr.arm_runs(runs, arm):
            usage = run.usage_total or run.usage or {}
            final = run.usage or {}
            run_dir = Path(run.run_directory or "")
            run_dir = run_dir if run_dir.is_absolute() else root / run_dir
            meta = _read_json(run_dir / "metadata.json")
            prompt = _int(usage.get("prompt_token_count"))
            cached = min(_int(usage.get("cached_content_token_count")), prompt)
            candidates = _int(usage.get("candidates_token_count"))
            thoughts = _int(usage.get("thoughts_token_count"))
            cost, _billed, out_tokens, _cached = cost_of_usage(usage, pricing)
            gen_s = None
            try:
                started = datetime.fromisoformat(str(meta["timestamp"])).timestamp()
                gen_s = (run_dir / "raw_response.json").stat().st_mtime - started
            except KeyError, ValueError, OSError:
                pass
            raw_chars = None
            raw = _bytes(run_dir / "raw_response.json")
            if raw is not None:
                raw_chars = len(raw.decode("utf-8", errors="replace"))
            rows.append(
                {
                    "arm": arm["key"],
                    "model": arm["model"],
                    "prompting": arm["prompting"],
                    "dialect": arm["dialect"],
                    "epoch": arm["epoch"],
                    "level": run.guidance_level,
                    "seed": run.seed,
                    "condition": run.condition_name,
                    "success": bool(run.success),
                    "llm_calls": run.llm_calls or 1,
                    "feedback_iterations": run.feedback_iterations or 0,
                    "execution_rounds": run.execution_feedback_rounds or 0,
                    "prompt": prompt,
                    "cached": cached,
                    "cand": candidates,
                    "thoughts": thoughts,
                    "reasoning": _int(usage.get("reasoning_token_count")),
                    "output": out_tokens,
                    "tokens": prompt + out_tokens,
                    "total_token_count": _int(usage.get("total_token_count")),
                    "cost": cost,
                    "cost_no_cache": (
                        prompt * pricing.input_usd_per_m + out_tokens * pricing.output_usd_per_m
                    )
                    / 1e6,
                    "cost_final_turn": cost_of_usage(final, pricing)[0] if final else None,
                    "cli_cost_final_turn": final.get("cli_list_price_usd"),
                    "provider_cost_final_turn": final.get("provider_cost_usd"),
                    "llm_backend": run.llm_backend,
                    "temperature": run.temperature,
                    "prompt_sha256": meta.get("prompt_sha256"),
                    "gen_s": gen_s,
                    "raw_chars": raw_chars,
                    "run_dir": run_dir,
                }
            )
    return rows


def _med(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if len(values) else None


def _quartiles(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    if not len(values):
        return None, None, None
    return tr.quart(values)


def source_facts(root: Path) -> dict[str, Any]:
    llm = root / "src" / "benchmark_agent" / "llm"
    gemini_src = (llm / "gemini_client.py").read_text(encoding="utf-8")
    openai_src = (llm / "openai_client.py").read_text(encoding="utf-8")
    cli_src = (llm / "claude_cli_client.py").read_text(encoding="utf-8")
    cli_args = cli_src[cli_src.index("def _args") : cli_src.index("def complete")]
    system_prompt = re.search(r'_DEFAULT_SYSTEM_PROMPT = "([^"]+)"', cli_src)
    facts = {
        "gemini_seed_sent": "GenerateContentConfig(temperature=temperature, seed=seed)"
        in gemini_src,
        "gemini_thinking_config": "thinking"
        in gemini_src.lower().replace("thoughts_token_count", ""),
        "gemini_system_instruction": "system_instruction" in gemini_src,
        "openai_seed_sent": 'payload["seed"] = seed' in openai_src,
        "openai_messages_user_only": '"messages": [{"role": "user", "content": prompt}]'
        in openai_src,
        "cli_seed_sent": "seed" in cli_args,
        "cli_temperature_sent": "temperature" in cli_args,
        "cli_system_prompt": system_prompt.group(1) if system_prompt else None,
        "cli_replaces_system_prompt": '"--system-prompt"' in cli_args,
        "cli_effort_flag_only_if_set": "if self._effort:" in cli_args,
    }
    expected = {
        "gemini_seed_sent": True,
        "gemini_thinking_config": False,
        "gemini_system_instruction": False,
        "openai_seed_sent": True,
        "openai_messages_user_only": True,
        "cli_seed_sent": False,
        "cli_temperature_sent": False,
        "cli_system_prompt": "You are a helpful assistant.",
        "cli_replaces_system_prompt": True,
        "cli_effort_flag_only_if_set": True,
    }
    changed = {name: value for name, value in facts.items() if expected[name] != value}
    if changed:
        raise SystemExit(f"client sources changed, check the access table: {changed}")
    return facts


def seed_pairs(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int | None, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["llm_calls"] == 1 and row["prompt_sha256"]:
            groups[(row["condition"], row["seed"], row["prompt_sha256"])].append(row)
    out = []
    for (condition, seed, _sha), members in sorted(
        groups.items(), key=lambda item: (item[0][0], item[0][1] or 0)
    ):
        for first, second in combinations(sorted(members, key=lambda row: row["epoch"]), 2):
            if first["epoch"] == second["epoch"]:
                continue
            identical = all(
                _bytes(first["run_dir"] / name) == _bytes(second["run_dir"] / name)
                and _bytes(first["run_dir"] / name) is not None
                for name in ("benchmark.feature", "steps.py")
            )
            out.append(
                {
                    "model": first["model"],
                    "condition": condition,
                    "seed": seed,
                    "epoch_a": first["epoch"],
                    "epoch_b": second["epoch"],
                    "identical": identical,
                }
            )
    return out


def access_rows(
    rows: Sequence[dict[str, Any]],
    seeds: Sequence[dict[str, Any]],
    config: Any,
    facts: dict[str, Any],
    models: Sequence[str] = tr.MODELS,
) -> list[dict[str, Any]]:
    flash_chars_per_token = _med(
        [
            row["raw_chars"] / row["cand"]
            for row in rows
            if row["model"] == "gemini"
            and row["level"] != "L1"
            and row["llm_calls"] == 1
            and row["cand"]
            and row["raw_chars"]
        ]
    )
    out = []
    for model in models:
        model_config = config.models.models[model]
        params = model_config.parameters
        extra = params.get("extra_body") or {}
        model_rows = [row for row in rows if row["model"] == model]
        l23 = [row for row in model_rows if row["level"] in ("L2", "L3") and row["llm_calls"] == 1]
        backends = sorted({str(row["llm_backend"]) for row in model_rows})
        temps = sorted({str(row["temperature"]) for row in model_rows})
        model_seeds = [pair for pair in seeds if pair["model"] == model]
        if model in ("gemini", "gemini_lite"):
            access = "Vertex AI (google-genai SDK, global endpoint)"
            seed_sent = "yes" if facts["gemini_seed_sent"] else "no"
            seed_effect = (
                f"best effort: {sum(pair['identical'] for pair in model_seeds)}/{len(model_seeds)} same-prompt same-seed pairs across epochs byte-identical"
                if model_seeds
                else "best effort (no same-prompt same-seed pair across epochs in E6-E17)"
            )
            reasoning = "model default (no thinking config sent)"
            system_prompt = "none (whole prompt in one user turn)"
            pin = "none"
        elif model == "claude_haiku":
            access = "Claude Code CLI (claude -p, subscription login), tools/MCP/settings off"
            seed_sent = "no (recorded only)" if not facts["cli_seed_sent"] else "yes"
            seed_effect = "not applicable (no seed sent)"
            reasoning = (
                "CLI default (no --effort); thinking not reported separately, included in output"
            )
            system_prompt = f'replaced by "{facts["cli_system_prompt"]}"; CLI adds a context block (date, account) to the user turn'
            pin = "none (Anthropic via CLI)"
        else:
            access = "OpenRouter chat completions (OpenAI-compatible)"
            seed_sent = "yes" if facts["openai_seed_sent"] else "no"
            seed_effect = "ignored: seed not supported by the Z.AI endpoint (models.yaml)"
            reasoning_config = extra.get("reasoning", {})
            reasoning = f"reasoning effort {reasoning_config.get('effort')} (cannot be disabled), max_tokens {extra.get('max_tokens')}"
            system_prompt = "none (whole prompt in one user turn)"
            provider = extra.get("provider", {})
            pin = f"provider order {'/'.join(provider.get('order', []))} (fp8), allow_fallbacks {str(provider.get('allow_fallbacks')).lower()}"
        if model == "gemini":
            reasoning_med = _med([row["thoughts"] for row in l23])
            reasoning_basis = "thoughts_token_count"
        elif model == "gemini_lite":
            reasoning_med = _med([row["thoughts"] for row in l23])
            reasoning_basis = (
                "thoughts_token_count (0 in all runs)"
                if all(row["thoughts"] == 0 for row in model_rows)
                else "thoughts_token_count"
            )
        elif model == "glm":
            reasoning_med = _med([row["reasoning"] for row in l23])
            reasoning_basis = "reasoning_token_count (part of output)"
        else:
            reasoning_med = None
            reasoning_basis = "not reported"
        visible_est = None
        if model == "claude_haiku" and flash_chars_per_token:
            visible_est = _med(
                [row["raw_chars"] / flash_chars_per_token for row in l23 if row["raw_chars"]]
            )
        cached_runs = sum(1 for row in model_rows if row["cached"] > 0)
        cached_share = _med([row["cached"] / row["prompt"] for row in model_rows if row["prompt"]])
        out.append(
            {
                "model": model,
                "role": PIPELINE_LABEL if model == tr.PIPELINE_MODEL else tr.ROLE_MAIN,
                "model_id": model_config.model_name,
                "provider": model_config.provider,
                "client_adapter": model_config.client_adapter,
                "access_path": access,
                "llm_backend_observed": ";".join(backends),
                "temperature_config": params.get("temperature"),
                "temperature_recorded": ";".join(temps),
                "seed_sent": seed_sent,
                "seed_effect": seed_effect,
                "reasoning_setting": reasoning,
                "reasoning_tokens_median_L2L3": reasoning_med,
                "reasoning_basis": reasoning_basis,
                "output_tokens_median_L2L3": _med([row["output"] for row in l23]),
                "visible_tokens_median_L2L3": _med([row["cand"] for row in l23])
                if model != "claude_haiku"
                else None,
                "visible_tokens_est_L2L3": visible_est,
                "n_L2L3_single_call": len(l23),
                "prompt_tokens_median_L2_Kestrel_zs": _med(
                    [
                        row["prompt"]
                        for row in model_rows
                        if row["level"] == "L2"
                        and row["dialect"] == "Kestrel"
                        and row["prompting"] == "zero_shot"
                    ]
                ),
                "system_prompt": system_prompt,
                "provider_pin": pin,
                "runs_with_cached_tokens": cached_runs,
                "runs": len(model_rows),
                "epochs": ";".join(
                    f"E{epoch}" for epoch in sorted({row["epoch"] for row in model_rows})
                ),
                "cached_share_of_prompt_median": cached_share,
                "cached_share_of_prompt_tokens_sum": sum(row["cached"] for row in model_rows)
                / sum(row["prompt"] for row in model_rows),
            }
        )
    return out


def feedback_overhead(rows: Sequence[dict[str, Any]], model: str) -> dict[str, Any]:
    l45 = [row for row in rows if row["model"] == model and row["level"] in ("L4", "L5")]
    multi = [row for row in l45 if row["llm_calls"] > 1]
    single = [row for row in l45 if row["llm_calls"] == 1]
    total = sum(row["cost"] for row in l45)
    extra = (
        sum(row["cost"] for row in multi)
        - len(multi) * float(np.mean([row["cost"] for row in single]))
        if multi
        else 0.0
    )
    extra_cell = 0.0
    for row in multi:
        cell_costs = [
            other["cost"]
            for other in single
            if other["arm"] == row["arm"] and other["level"] == row["level"]
        ]
        extra_cell += row["cost"] - float(np.mean(cell_costs))
    return {
        "l45_runs": len(l45),
        "l45_multi_call_runs": len(multi),
        "l45_cost_usd": total,
        "feedback_extra_usd": extra,
        "feedback_share": extra / total if total else None,
        "feedback_extra_usd_cell_baseline": extra_cell if multi else 0.0,
        "feedback_share_cell_baseline": (extra_cell / total) if (multi and total) else 0.0,
    }


def cost_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    mwu_p_values = []
    for model in tr.MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        assert len(model_rows) == 200, (model, len(model_rows))
        l1 = [row for row in model_rows if row["level"] == "L1"]
        l2 = [row for row in model_rows if row["level"] == "L2"]
        l25 = [row for row in model_rows if row["level"] != "L1"]
        k25 = sum(row["success"] for row in l25)
        exec_l45 = [
            row for row in model_rows if row["level"] in ("L4", "L5") and row["epoch"] >= 10
        ]
        first = [row["tokens"] for row in exec_l45 if row["feedback_iterations"] == 0]
        repaired = [row["tokens"] for row in exec_l45 if row["feedback_iterations"] > 0]
        _u, mwu_p = tr.mwu(first, repaired)
        mwu_p_values.append(mwu_p)
        gen = [row["gen_s"] for row in l25 if row["llm_calls"] == 1 and row["gen_s"] is not None]
        gen_quartiles = _quartiles(gen)
        overhead = feedback_overhead(rows, model)
        summary = {
            "model": model,
            "runs": len(model_rows),
            "cost_per_run_L1": float(np.mean([row["cost"] for row in l1])),
            "cost_per_run_L2_L5": float(np.mean([row["cost"] for row in l25])),
            "successes_L2_L5": k25,
            "runs_L2_L5": len(l25),
            "cost_per_success_L2_L5": sum(row["cost"] for row in l25) / k25 if k25 else None,
            "l1_to_l2_cost_ratio": float(np.mean([row["cost"] for row in l1]))
            / float(np.mean([row["cost"] for row in l2])),
            "l1_share_of_cost": sum(row["cost"] for row in l1)
            / sum(row["cost"] for row in model_rows),
            "cost_total_usd": sum(row["cost"] for row in model_rows),
            "cost_total_no_cache_usd": sum(row["cost_no_cache"] for row in model_rows),
            "exec_regime_l45_firstpass_n": len(first),
            "exec_regime_l45_repaired_n": len(repaired),
            "tokens_firstpass_median": _med(first),
            "tokens_repaired_median": _med(repaired),
            "repaired_to_firstpass_ratio": (_med(repaired) / _med(first))
            if (repaired and first)
            else None,
            "mwu_p_firstpass_vs_repaired": mwu_p,
            **overhead,
            "gen_time_s_median": gen_quartiles[1],
            "gen_time_s_q1": gen_quartiles[0],
            "gen_time_s_q3": gen_quartiles[2],
            "gen_time_n": len(gen),
            "gen_time_basis": "end-to-end: metadata.timestamp -> raw_response.json mtime, single-call L2-L5 runs (incl. CLI start / routing)",
        }
        out.append(summary)
    for summary, p_holm in zip(out, tr.holm(mwu_p_values), strict=True):
        summary["mwu_p_holm"] = p_holm
    return out


def pipeline_cost_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    assert all(row["model"] == tr.PIPELINE_MODEL for row in rows)
    out = []
    for epoch in sorted({row["epoch"] for row in rows}):
        epoch_rows = [row for row in rows if row["epoch"] == epoch]
        out.append(
            {
                "role": PIPELINE_LABEL,
                "epoch": f"E{epoch}",
                "epoch_role": tr.epoch_role(epoch, tr.PIPELINE_MODEL),
                "parts": ";".join(sorted({row["arm"] for row in epoch_rows})),
                "runs": len(epoch_rows),
                "successes": sum(row["success"] for row in epoch_rows),
                "cost_total_usd": sum(row["cost"] for row in epoch_rows),
                "cost_total_no_cache_usd": sum(row["cost_no_cache"] for row in epoch_rows),
                "cost_per_run_usd": sum(row["cost"] for row in epoch_rows) / len(epoch_rows),
            }
        )
    out.append(
        {
            "role": PIPELINE_LABEL,
            "epoch": "all",
            "epoch_role": "total pipeline development (flash-lite)",
            "parts": "",
            "runs": len(rows),
            "successes": sum(row["success"] for row in rows),
            "cost_total_usd": sum(row["cost"] for row in rows),
            "cost_total_no_cache_usd": sum(row["cost_no_cache"] for row in rows),
            "cost_per_run_usd": sum(row["cost"] for row in rows) / len(rows),
        }
    )
    return out


def frontier_cells(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    cells = []
    for arm in tr.ARMS:
        for lvl in LEVELS:
            cell_rows = [row for row in rows if row["arm"] == arm["key"] and row["level"] == lvl]
            k = sum(row["success"] for row in cell_rows)
            costs = [row["cost"] for row in cell_rows]
            cells.append(
                {
                    "arm": arm["key"],
                    "model": arm["model"],
                    "epoch": arm["epoch"],
                    "level": lvl,
                    "n": len(cell_rows),
                    "success": k,
                    "cost_per_run": float(np.mean(costs)),
                    "cost_per_success": sum(costs) / k if k else None,
                    "mean_llm_calls": float(np.mean([row["llm_calls"] for row in cell_rows])),
                }
            )
    return cells


def fig_frontier(
    cells: Sequence[dict[str, Any]], cost: Sequence[dict[str, Any]], out: Path
) -> tuple[str, dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(tr.TEXT_WIDTH_IN, 3.3))
    for cell in cells:
        y = cell["success"] + MODEL_Y_OFFSET[cell["model"]]
        if cell["level"] == "L1":
            ax.scatter(
                cell["cost_per_run"],
                y,
                marker="x",
                s=16,
                linewidths=0.9,
                color=tr.MODEL_COLOR[cell["model"]],
                alpha=0.85,
                zorder=3,
            )
        else:
            ax.scatter(
                cell["cost_per_run"],
                y,
                marker=LEVEL_MARKER[cell["level"]],
                s=20,
                color=tr.MODEL_COLOR[cell["model"]],
                edgecolors=tr.SURFACE,
                linewidths=0.4,
                alpha=0.9,
                zorder=3,
            )
    for row in cost:
        model = row["model"]
        mean_k = row["successes_L2_L5"] / (row["runs_L2_L5"] / 10)
        ax.scatter(
            row["cost_per_run_L2_L5"],
            mean_k,
            marker="o",
            s=95,
            facecolors="white",
            edgecolors=tr.MODEL_COLOR[model],
            linewidths=1.8,
            zorder=4,
        )
        ax.annotate(
            tr.MODEL_LABEL[model],
            (row["cost_per_run_L2_L5"], mean_k),
            xytext=(0, 9 if model in ("claude_haiku", "gemini") else -9),
            textcoords="offset points",
            ha="center",
            va="bottom" if model in ("claude_haiku", "gemini") else "top",
            fontsize=7,
            color=tr.MODEL_COLOR[model],
            fontweight="bold",
            zorder=5,
        )
    full = [cell for cell in cells if cell["success"] == 10]
    best = min(full, key=lambda cell: cell["cost_per_run"])
    arm = tr.ARM_BY_KEY[best["arm"]]
    ax.annotate(
        f"cheapest 10/10: {tr.MODEL_LABEL[arm['model']]} {tr.PROMPT_LABEL[arm['prompting']]} {arm['dialect']} {best['level']} (E{arm['epoch']}), {best['cost_per_run']:.4f} USD/run",
        (best["cost_per_run"], 10 + MODEL_Y_OFFSET[best["model"]]),
        xytext=(-8, 27),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=7,
        color=tr.INK_2,
        arrowprops={
            "arrowstyle": "-",
            "color": tr.MUTED,
            "lw": 0.7,
            "shrinkA": 0,
            "shrinkB": 3,
            "relpos": (0.0, 0.0),
        },
    )
    ax.set_xscale("log")
    ax.set_ylim(-0.8, 12.4)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_ylabel("Successful runs (of 10)")
    ax.set_xlabel("Mean list-price cost per run (USD, log scale)")
    ax.grid(True, axis="x", which="major", color=tr.GRID, lw=0.6)
    ax.spines["bottom"].set_color(tr.BASELINE)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    model_handles = [
        Patch(color=tr.MODEL_COLOR[model], label=tr.MODEL_FULL[model]) for model in tr.MODELS
    ]
    level_handles = [
        Line2D(
            [],
            [],
            ls="",
            marker=LEVEL_MARKER[lvl],
            color=tr.INK_2,
            markersize=4.5 if lvl != "L1" else 5,
            label=lvl + (" (all 0/10)" if lvl == "L1" else ""),
        )
        for lvl in LEVELS
    ]
    level_handles.append(
        Line2D(
            [],
            [],
            ls="",
            marker="o",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=tr.INK_2,
            markeredgewidth=1.5,
            label="model mean L2-L5",
        )
    )
    model_legend = ax.legend(
        handles=model_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.17),
        handlelength=1.2,
    )
    ax.legend(
        handles=level_handles,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.26),
        handlelength=1.0,
        columnspacing=1.2,
    )
    ax.add_artist(model_legend)
    return tr.save(fig, out, "stage_a_cost_frontier"), best


def fig_tokens(rows: Sequence[dict[str, Any]], access: Sequence[dict[str, Any]], out: Path) -> str:
    fig, ax = plt.subplots(figsize=(tr.TEXT_WIDTH_IN, 3.0))
    width = 0.24
    rng = np.random.default_rng(tr.JITTER_SEED)
    shares = {}
    for model in tr.MODELS:
        l25 = [
            row
            for row in rows
            if row["model"] == model
            and row["level"] != "L1"
            and row["llm_calls"] == 1
            and row["output"]
        ]
        if model == "gemini":
            shares[model] = (
                f"thoughts {100 * float(np.median([row['thoughts'] / row['output'] for row in l25])):.0f} % of output"
            )
        elif model == "glm":
            shares[model] = (
                f"reasoning {100 * float(np.median([row['reasoning'] / row['output'] for row in l25])):.1f} % of output"
            )
        else:
            model_access = next(entry for entry in access if entry["model"] == model)
            visible = model_access["visible_tokens_est_L2L3"]
            total = model_access["output_tokens_median_L2L3"]
            shares[model] = (
                f"thinking not reported (~{100 * (1 - visible / total):.0f} % est.)"
                if visible and total
                else "thinking not reported"
            )
    for i, lvl in enumerate(LEVELS):
        for j, model in enumerate(tr.MODELS):
            outputs = [
                row["output"]
                for row in rows
                if row["model"] == model and row["level"] == lvl and row["output"] > 0
            ]
            if not outputs:
                continue
            x = i + (j - (len(tr.MODELS) - 1) / 2) * (width + 0.03)
            ax.boxplot(
                [outputs],
                positions=[x],
                widths=width,
                whis=(0, 100),
                patch_artist=True,
                showfliers=False,
                medianprops={"color": tr.INK, "lw": 1.0},
                boxprops={"facecolor": "white", "edgecolor": tr.MODEL_COLOR[model], "lw": 0.9},
                whiskerprops={"color": tr.MODEL_COLOR[model], "lw": 0.8},
                capprops={"color": tr.MODEL_COLOR[model], "lw": 0.8},
            )
            jitter = rng.uniform(-width * 0.32, width * 0.32, len(outputs))
            ax.scatter(
                np.full(len(outputs), x) + jitter,
                outputs,
                s=3.5,
                color=tr.MODEL_COLOR[model],
                alpha=0.55,
                lw=0,
                zorder=3,
            )
    ax.set_yscale("log")
    ax.set_xticks(range(len(LEVELS)), LEVELS)
    ax.set_xlim(-0.55, len(LEVELS) - 0.45)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Output tokens per run\n(all turns, incl. reasoning)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    ax.spines["bottom"].set_color(tr.BASELINE)
    handles = [
        Patch(
            facecolor="white",
            edgecolor=tr.MODEL_COLOR[model],
            lw=1.2,
            label=f"{tr.MODEL_FULL[model]}: {shares[model]}",
        )
        for model in tr.MODELS
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        ncol=1,
        bbox_to_anchor=(0.5, -0.1),
        handlelength=1.2,
        columnspacing=1.5,
    )
    return tr.save(fig, out, "tokens_by_model")


def _tex_count(value: float | None) -> str:
    return "--" if value is None else f"{value:,.0f}".replace(",", "\\,")


def tex_access(
    access: Sequence[dict[str, Any]],
    seed_counts: dict[str, tuple[int, int, str]],
) -> str:
    by_model = {entry["model"]: entry for entry in access}
    order = tr.MODELS

    def row(label: str, f: Any) -> list[str]:
        return [label] + [f(by_model[model]) for model in order]

    def cache(a: dict[str, Any]) -> str:
        return f"{a['runs_with_cached_tokens']}/{a['runs']} runs; run median {100 * a['cached_share_of_prompt_median']:.1f}\\,\\%, overall {100 * a['cached_share_of_prompt_tokens_sum']:.1f}\\,\\% of prompt tokens"

    def reasoning_tok(a: dict[str, Any]) -> str:
        if a["model"] == "claude_haiku":
            visible = a["visible_tokens_est_L2L3"]
            total = a["output_tokens_median_L2L3"]
            return (
                f"not reported (est.\\ $\\approx${100 * (1 - visible / total):.0f}\\,\\% of output)"
            )
        return _tex_count(a["reasoning_tokens_median_L2L3"])

    short_access = {
        "gemini": "Vertex AI (google-genai SDK)",
        "claude_haiku": "Claude Code CLI, subscription; tools, MCP, settings off",
        "glm": "OpenRouter (OpenAI-compatible API)",
    }
    short_reason = {
        "gemini": "model default (no thinking config)",
        "claude_haiku": "CLI default (no effort flag)",
        "glm": "effort low (cannot be disabled)",
    }
    short_seed = {
        "gemini": "sent; best effort",
        "claude_haiku": "not sent (recorded only)",
        "glm": "sent; not supported by the endpoint",
    }
    short_sys = {
        "gemini": "none (one user turn)",
        "claude_haiku": "``You are a helpful assistant.'' + CLI context block",
        "glm": "none (one user turn)",
    }
    pin = {
        "gemini": "--",
        "claude_haiku": "--",
        "glm": "Z.AI only (fp8), no fallbacks",
    }

    def seed_cell(a: dict[str, Any]) -> str:
        text = short_seed[a["model"]]
        if a["model"] == "gemini":
            n_identical, n_pairs, _ = seed_counts.get("gemini", (0, 0, ""))
            text += (
                f" ({n_identical}/{n_pairs} repeat pairs identical)"
                if n_pairs
                else " (no repeat pair)"
            )
        return text

    rows = [
        row("Model id", lambda entry: tr.tex(entry["model_id"])),
        row("Access path", lambda entry: short_access[entry["model"]]),
        row(
            "Temperature",
            lambda entry: (
                tr.fmt(entry["temperature_config"], 1)
                if entry["temperature_config"] is not None
                else "CLI default (not settable)"
            ),
        ),
        row("Seed", seed_cell),
        row("Reasoning", lambda entry: short_reason[entry["model"]]),
        row("Reasoning tokens (median L2/L3)", reasoning_tok),
        row(
            "Output tokens (median L2/L3)",
            lambda entry: _tex_count(entry["output_tokens_median_L2L3"]),
        ),
        row("System prompt", lambda entry: short_sys[entry["model"]]),
        row("Provider pin", lambda entry: pin[entry["model"]]),
        row("Implicit caching", cache),
    ]
    col = r">{\raggedright\arraybackslash}p{3.75cm}"
    flash_identical, flash_n_pairs, flash_pairs = seed_counts.get("gemini", (0, 0, ""))
    flash_seed = (
        f"flash: {flash_identical}/{flash_n_pairs} pairs identical ({flash_pairs}; the multi-shot L3 runs of E8/E9 belong to no arm and serve only this check, the E8 runs found no same-prompt partner); "
        if flash_n_pairs
        else "flash: no repeat pair; "
    )
    return tr.latex_table(
        "Access conditions of the three compared models. The factor ``model'' is the model together with its access configuration (draft).",
        "tab:model_access_conditions",
        [""] + [f"\\textbf{{{tr.MODEL_LABEL[model]}}}" for model in order],
        rows,
        r">{\raggedright\arraybackslash}p{2.35cm}" + col * len(order),
        note="Sources: \\texttt{config/models.yaml}, the client adapters in \\texttt{src/benchmark\\_agent/llm/} and the run metadata of the main matrix (200 runs per model, four arms each). "
        "Seed: same-condition, same-seed, same-prompt single-call pairs across epochs with byte-identical feature and steps file; "
        + flash_seed
        + "haiku and glm ran every prompt--seed pair once. "
        "Token medians over single-call L2/L3 runs; output includes reasoning (counted as output tokens). haiku's thinking share is "
        "estimated from response length at flash's characters per visible token. Implicit caching: runs with cached prompt tokens; cached share of the prompt as median over runs and over all prompt tokens of the model (multi-turn L1/L4/L5 runs are cached less); "
        "the glm and haiku caches are provider-side (no explicit cache was set up).",
        size=r"\scriptsize",
        tabcolsep="3pt",
    )


def tex_cost(cost: Sequence[dict[str, Any]]) -> str:
    rows = []
    for model_cost in cost:
        tokens = (
            f"{model_cost['tokens_firstpass_median'] / 1000:.1f} / {model_cost['tokens_repaired_median'] / 1000:.1f}"
            if model_cost["tokens_repaired_median"] is not None
            else f"{model_cost['tokens_firstpass_median'] / 1000:.1f} / --"
        )
        rows.append(
            [
                tr.MODEL_LABEL[model_cost["model"]],
                f"{model_cost['cost_per_run_L1']:.4f}",
                f"{model_cost['cost_per_run_L2_L5']:.4f}",
                f"{model_cost['cost_per_success_L2_L5']:.4f}",
                f"{model_cost['successes_L2_L5']}/{model_cost['runs_L2_L5']}",
                f"{model_cost['cost_total_usd']:.2f}",
                tokens
                + f" ({model_cost['exec_regime_l45_firstpass_n']}/{model_cost['exec_regime_l45_repaired_n']})",
                f"{100 * model_cost['feedback_share']:.0f}\\,\\%"
                if model_cost["l45_multi_call_runs"]
                else "--",
                f"{model_cost['gen_time_s_median']:.1f}",
            ]
        )
    total = sum(model_cost["cost_total_usd"] for model_cost in cost)
    rows.append(["main matrix", "", "", "", "", f"{total:.2f}", "", "", ""])
    return tr.latex_table(
        "List-price cost, tokens and generation time of the three compared models in the main matrix (12 arms, 600 runs; draft).",
        "tab:cost_by_model",
        [
            "Model",
            "\\$/run L1",
            "\\$/run L2--5",
            "\\$/success",
            "Succ.",
            "Total \\$",
            "Tokens first / repaired (k, n)",
            "Feedback",
            "Time (s)",
        ],
        rows,
        "lrrrrrlrr",
        note="USD at list price from the recorded tokens (cached prompt tokens at each provider's cache price); haiku was not billed (subscription). "
        "\\$/success and Succ.: L2--L5 (160 runs per model). Tokens: median prompt + output tokens of a run over all turns, L4/L5 runs of the "
        "execution-feedback epochs (E12--E17; flash only in its multi-shot arms, its zero-shot arms E8/E9 had syntax-only feedback) without vs.\\ with feedback rounds. Feedback: extra cost of the multi-call L4/L5 runs over the model's mean "
        "single-call L4/L5 run, as share of all L4/L5 cost (estimate; per-turn usage is not recorded). Time: median end-to-end generation time of single-call "
        "L2--L5 runs (metadata timestamp to the write of the response file), including CLI start-up (haiku) and OpenRouter routing (glm). "
        "Tokens are not comparable across providers (different tokenizers; haiku's prompt carries the CLI context block).",
        size=r"\scriptsize",
        midrules_after=(len(rows) - 2,),
        tabcolsep="3pt",
    )


def tex_figures(best: dict[str, Any]) -> str:
    arm = tr.ARM_BY_KEY[best["arm"]]
    floats = [
        (
            "stage_a_cost_frontier",
            "Stage-A reliability against list-price cost for the three compared models (12 arms, 60 cells). Each small marker is one arm $\\times$ level cell (N = 10; marker = level, colour = model); "
            "markers are offset vertically by model to reduce overlap. L1 (crosses) failed in every cell. Open circles: mean over the model's L2--L5 "
            f"cells (four arms, descriptive). Cheapest cell with 10/10: {tr.tex(tr.MODEL_LABEL[arm['model']])} {tr.PROMPT_LABEL[arm['prompting']]} {arm['dialect']} "
            f"{best['level']} (E{arm['epoch']}), {best['cost_per_run']:.4f} USD per run. haiku priced at list price, not billed (draft).",
            "fig:stage_a_cost_frontier",
        ),
        (
            "tokens_by_model",
            "Output tokens per run (all model turns, reasoning included, log scale) by model and level, three compared models, 200 runs each (main matrix); boxes span the quartiles, whiskers the range. "
            "flash reports its thinking separately, glm its reasoning; haiku's thinking is part of the output count and its share is a length-based estimate. "
            "Token counts come from different tokenizers and are not directly comparable across providers (draft).",
            "fig:tokens_by_model",
        ),
    ]
    parts = []
    for name, caption, label in floats:
        parts.append(
            "\n".join(
                [
                    r"\begin{figure}[H]",
                    r"    \centering",
                    f"    \\includegraphics[width=\\textwidth]{{fig/{name}.pdf}}",
                    f"    \\caption{{{caption}}}",
                    f"    \\label{{{label}}}",
                    r"\end{figure}",
                    "",
                ]
            )
        )
    return "\n".join(parts)


def main() -> None:
    tr._style()
    config, runs = tr.load_runs()
    root = Path(config.paths.project_root)
    out = root / "results" / "reports" / "thesis_results" / "extra" / NAME
    dirs = {name: out / name for name in ("fig", "tables", "latex")}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    facts = source_facts(root)
    main_rows = run_rows(runs, config, tr.ARMS)
    flash_extra_rows = run_rows(runs, config, FLASH_SEED_EXTRA_ARMS)
    lite_rows = (
        run_rows(runs, config, tr.SYNTAX_ONLY_ARMS)
        + run_rows(runs, config, tr.PIPELINE_ARMS)
        + run_rows(runs, config, LITE_SEED_EXTRA_ARMS)
    )
    seeds = seed_pairs(main_rows + flash_extra_rows)
    lite_seeds = seed_pairs(lite_rows)
    assert {pair["model"] for pair in seeds} <= set(tr.MODELS)
    assert {pair["model"] for pair in lite_seeds} <= {tr.PIPELINE_MODEL}
    access = access_rows(main_rows, seeds, config, facts)
    lite_access = access_rows(lite_rows, lite_seeds, config, facts, (tr.PIPELINE_MODEL,))[0]
    cost = cost_rows(main_rows)
    lite_cost = pipeline_cost_rows(lite_rows)
    lite_total = lite_cost[-1]
    cells = frontier_cells(main_rows)
    flash_pairs = sorted({(pair["epoch_a"], pair["epoch_b"], pair["condition"]) for pair in seeds})
    seed_counts = {
        "gemini": (
            sum(pair["identical"] for pair in seeds),
            len(seeds),
            ", ".join(
                f"{condition.split('_')[0]} {'multi-shot' if 'multishot' in condition else 'zero-shot'} E{epoch_a}$\\to$E{epoch_b}"
                for epoch_a, epoch_b, condition in flash_pairs
            ),
        ),
        tr.PIPELINE_MODEL: (
            sum(pair["identical"] for pair in lite_seeds),
            len(lite_seeds),
            "",
        ),
    }

    assert len(main_rows) == 600, len(main_rows)
    assert {row["model"] for row in main_rows} == set(tr.MODELS)
    assert len(cells) == 60, len(cells)
    assert len(lite_rows) == 340, len(lite_rows)
    assert len({str(row["run_dir"]) for row in lite_rows}) == len(lite_rows)
    assert not {str(row["run_dir"]) for row in lite_rows} & {
        str(row["run_dir"]) for row in main_rows
    }
    mismatch = sum(
        1
        for row in main_rows
        if row["total_token_count"] and row["total_token_count"] != row["tokens"]
    )
    glm_single = [
        row
        for row in main_rows
        if row["model"] == "glm" and row["llm_calls"] == 1 and row["provider_cost_final_turn"]
    ]
    glm_ratio = [row["provider_cost_final_turn"] / row["cost_final_turn"] for row in glm_single]
    haiku_single = [
        row
        for row in main_rows
        if row["model"] == "claude_haiku" and row["llm_calls"] == 1 and row["cli_cost_final_turn"]
    ]
    haiku_ratio = [row["cli_cost_final_turn"] / row["cost_final_turn"] for row in haiku_single]

    access_keys = list(access[0].keys())
    tr.write_csv(
        dirs["tables"] / "model_access_conditions.csv",
        access_keys,
        [[entry[key] for key in access_keys] for entry in access],
    )
    cost_keys = list(cost[0].keys())
    tr.write_csv(
        dirs["tables"] / "cost_by_model.csv",
        cost_keys,
        [[model_cost[key] for key in cost_keys] for model_cost in cost],
    )
    cell_keys = list(cells[0].keys())
    tr.write_csv(
        dirs["tables"] / "cost_frontier_cells.csv",
        cell_keys,
        [[cell[key] for key in cell_keys] for cell in cells],
    )
    seed_keys = ["model", "condition", "seed", "epoch_a", "epoch_b", "identical"]
    tr.write_csv(
        dirs["tables"] / "seed_reproducibility.csv",
        seed_keys,
        [[pair[key] for key in seed_keys] for pair in seeds],
    )
    tr.write_csv(
        dirs["tables"] / "pipeline_flash_lite_access.csv",
        access_keys,
        [[lite_access[key] for key in access_keys]],
    )
    lite_cost_keys = list(lite_cost[0].keys())
    tr.write_csv(
        dirs["tables"] / "pipeline_flash_lite_cost.csv",
        lite_cost_keys,
        [[epoch_cost[key] for key in lite_cost_keys] for epoch_cost in lite_cost],
    )
    tr.write_csv(
        dirs["tables"] / "pipeline_flash_lite_seed_reproducibility.csv",
        ["role", *seed_keys],
        [[PIPELINE_LABEL, *[pair[key] for key in seed_keys]] for pair in lite_seeds],
    )
    (dirs["latex"] / "model_access_conditions.tex").write_text(
        tex_access(access, seed_counts), encoding="utf-8"
    )
    (dirs["latex"] / "cost_by_model.tex").write_text(tex_cost(cost), encoding="utf-8")

    frontier_fig, best = fig_frontier(cells, cost, dirs["fig"])
    tokens_fig = fig_tokens(main_rows, access, dirs["fig"])
    (dirs["latex"] / "figures.tex").write_text(tex_figures(best), encoding="utf-8")

    print("== access conditions (main models)")
    for entry in access:
        print(
            f"  {entry['model']:13s} backend={entry['llm_backend_observed']} T={entry['temperature_recorded']} seed={entry['seed_sent']} | "
            f"reasoning med L2/L3={entry['reasoning_tokens_median_L2L3']} output med={entry['output_tokens_median_L2L3']} visible med={entry['visible_tokens_median_L2L3']} "
            f"haiku visible est={entry['visible_tokens_est_L2L3']} | prompt L2 K zs med={entry['prompt_tokens_median_L2_Kestrel_zs']} | "
            f"cached runs {entry['runs_with_cached_tokens']}/{entry['runs']} share med={entry['cached_share_of_prompt_median']:.4f} sum={entry['cached_share_of_prompt_tokens_sum']:.4f}"
        )
    for label, pairs in (("flash", seeds), ("flash-lite (pipeline)", lite_seeds)):
        print(
            f"  seed pairs {label} (single-call, same prompt): {sum(pair['identical'] for pair in pairs)}/{len(pairs)} identical"
        )
        by_condition: dict[tuple[str, int, int], list[bool]] = defaultdict(list)
        for pair in pairs:
            by_condition[(pair["condition"], pair["epoch_a"], pair["epoch_b"])].append(
                pair["identical"]
            )
        for (condition, epoch_a, epoch_b), identical_flags in sorted(by_condition.items()):
            print(
                f"    {condition} E{epoch_a}->E{epoch_b}: {sum(identical_flags)}/{len(identical_flags)}"
            )
    print(f"  total_token_count != prompt+output in {mismatch} of {len(main_rows)} runs")
    print(
        f"  glm provider_cost / list (single call): n={len(glm_ratio)} min={min(glm_ratio):.5f} max={max(glm_ratio):.5f}"
    )
    print(
        f"  haiku CLI cost / list (single call): n={len(haiku_ratio)} median={float(np.median(haiku_ratio)):.1f}x"
    )
    print("== cost by model (main matrix)")
    for model_cost in cost:
        print(
            f"  {model_cost['model']:13s} total={model_cost['cost_total_usd']:.3f} (no cache {model_cost['cost_total_no_cache_usd']:.2f}) $/run L1={model_cost['cost_per_run_L1']:.4f} "
            f"L2-5={model_cost['cost_per_run_L2_L5']:.4f} $/succ={model_cost['cost_per_success_L2_L5']:.4f} ({model_cost['successes_L2_L5']}/{model_cost['runs_L2_L5']}) L1/L2={model_cost['l1_to_l2_cost_ratio']:.2f} "
            f"L1 share={100 * model_cost['l1_share_of_cost']:.1f}%"
        )
        print(
            f"      tokens first/repaired={model_cost['tokens_firstpass_median']}/{model_cost['tokens_repaired_median']} (n {model_cost['exec_regime_l45_firstpass_n']}/{model_cost['exec_regime_l45_repaired_n']}) "
            f"MWU p={model_cost['mwu_p_firstpass_vs_repaired'] or float('nan'):.2g} Holm={model_cost['mwu_p_holm'] or float('nan'):.2g} | feedback {model_cost['feedback_extra_usd']:.3f}/{model_cost['l45_cost_usd']:.3f} "
            f"= {100 * (model_cost['feedback_share'] or 0):.1f}% (cell baseline {100 * model_cost['feedback_share_cell_baseline']:.1f}%) multi={model_cost['l45_multi_call_runs']} | "
            f"gen time median {model_cost['gen_time_s_median']:.1f} s (IQR {model_cost['gen_time_s_q1']:.1f}-{model_cost['gen_time_s_q3']:.1f}, n={model_cost['gen_time_n']})"
        )
    main_total = sum(model_cost["cost_total_usd"] for model_cost in cost)
    print(f"  main matrix total (3 models, {len(main_rows)} runs) = {main_total:.2f} USD")
    print(
        f"  cheapest 10/10 cell: {best['arm']} {best['level']} {best['cost_per_run']:.4f} USD/run"
    )
    print("== pipeline development (flash-lite, not compared)")
    for epoch_cost in lite_cost:
        print(
            f"  {epoch_cost['epoch']:4s} runs={epoch_cost['runs']} succ={epoch_cost['successes']} cost={epoch_cost['cost_total_usd']:.3f} USD (no cache {epoch_cost['cost_total_no_cache_usd']:.3f})"
        )
    print(
        f"  flash-lite access: {lite_access['runs']} runs, cached {lite_access['runs_with_cached_tokens']}, "
        f"output med L2/L3={lite_access['output_tokens_median_L2L3']}"
    )
    tested = [
        tr.MODEL_LABEL[model_cost["model"]]
        for model_cost in cost
        if model_cost["mwu_p_firstpass_vs_repaired"] is not None
        and not np.isnan(model_cost["mwu_p_firstpass_vs_repaired"])
    ]
    flash_pairs_txt = ", ".join(
        f"{condition} E{epoch_a}->E{epoch_b}" for epoch_a, epoch_b, condition in flash_pairs
    )
    model_totals = ", ".join(
        f"{tr.MODEL_LABEL[model_cost['model']]} {model_cost['cost_total_usd']:.2f}"
        for model_cost in cost
    )

    readme = f"""# Cost and model access conditions (thesis 3.6, RQ3)

Generated by `scripts/thesis_cost_access.py` (uv run python scripts/thesis_cost_access.py). Read-only on
run metadata, `raw_response.json` mtimes, `config/models.yaml` and `src/benchmark_agent/llm/*.py`.
All values are descriptive except the exploratory first-pass-vs-repaired MWU test (Holm-corrected).

**Model comparison = three models only:** gemini-3.5-flash (flash), claude-haiku-5-5 (haiku) and
glm-5.3 (glm); main matrix = 12 arms (3 models x zero-/multi-shot x Kestrel/Tessera) x 5 levels x
N = 10 = {len(main_rows)} runs (flash E8/E9/E12/E13, haiku E14/E15, glm E16/E17).
**gemini-3.5-flash-lite (flash-lite) was used only to develop and test the pipeline cheaply.** It is in
no model figure, model table or model test here; its access conditions, cost and seed pairs are in the
separate `pipeline_flash_lite_*` files (record only), labelled "{PIPELINE_LABEL}"; no LaTeX table or figure
names it. Costs are not part of the research: the cost files below are kept for the record only.

Totals at list price:
- main matrix (3 models, {len(main_rows)} runs): {main_total:.2f} USD ({model_totals}; haiku not billed, subscription)
- pipeline development, flash-lite ({lite_total["runs"]} runs in E6, E7, E10-E13): {lite_total["cost_total_usd"]:.2f} USD (not compared)
- in neither total: the 20 flash multi-shot L3 runs of E8/E9 that belong to no arm (used only for the flash seed check) and the pilots E1-E5.

| File | Content |
|---|---|
| fig/{frontier_fig}.pdf/.png | Stage-A successes of 10 vs mean list-price USD/run (log), one marker per arm x level (60 cells, 3 models), model means L2-L5, cheapest 10/10 cell annotated |
| fig/{tokens_fig}.pdf/.png | Output tokens per run (all turns, incl. reasoning) by model (3 models) and level, log y; thinking share in the legend |
| tables/model_access_conditions.csv, latex/model_access_conditions.tex | For the thesis. 3 models: access path, model id, temperature, seed, reasoning setting/tokens, output tokens, system prompt, provider pin, implicit caching. No prices (costs are not part of the research) and no flash-lite |
| tables/cost_by_model.csv, latex/cost_by_model.tex | Record only (costs are not part of the research). 3 models: $/run L1 and L2-L5, $/success, totals (with and without cache), first-pass vs repaired tokens (MWU where both groups have >= 2 runs: {", ".join(tested) or "none"}; Holm over these), feedback overhead share, end-to-end generation time |
| tables/cost_frontier_cells.csv | The 60 arm x level cells behind the frontier figure |
| tables/seed_reproducibility.csv | flash: same condition + seed + prompt hash in two epochs (single-call; found: {flash_pairs_txt}, {seed_counts["gemini"][0]}/{seed_counts["gemini"][1]} identical; the E8/E9 multi-shot L3 runs belong to no arm): byte-identical artefacts? |
| tables/pipeline_flash_lite_access.csv | Record only, pipeline development (flash-lite, not compared): same columns as model_access_conditions |
| tables/pipeline_flash_lite_cost.csv | Pipeline development only (flash-lite, not compared): runs, Stage-A successes and list-price cost per epoch + total |
| tables/pipeline_flash_lite_seed_reproducibility.csv | Pipeline development only (flash-lite): same-seed pairs across E6/E7/E10-E13 (pairs share runs, not independent) |
| latex/figures.tex | Figure floats with draft captions |

Definitions
- Stage-A success = syntax + executed + workload conformant (the 16 nonconformant glm runs count as failures here; this folder has no Stage-B view).
- Cost: list price from tokens (`evaluation/cost.py:cost_of_usage`, pricing via `thesis_results.pricing_for`); haiku 0.10/0.50, cache 0.01 USD/MTok, not billed. The CLI's `total_cost_usd` is not used (median {float(np.median(haiku_ratio)):.0f}x list price). glm list price equals OpenRouter's billed `provider_cost_usd` in {len(glm_ratio)}/{len(glm_ratio)} single-call runs.
- Feedback overhead: sum over L4/L5 multi-call runs of (run cost - the model's mean single-call L4/L5 run cost), divided by all L4/L5 cost. Per-turn usage is not recorded, so this is an estimate; `feedback_share_cell_baseline` uses the arm x level mean instead.
- First-pass vs repaired tokens: prompt + output over all turns, L4/L5 of the execution-feedback epochs (E12-E17; flash's zero-shot arms E8/E9 had syntax-only feedback and are left out), feedback_iterations == 0 vs > 0. MWU exploratory, only where both groups have at least 2 runs, Holm over those models.
- Generation time: end-to-end, metadata.timestamp -> raw_response.json mtime, single-call L2-L5 runs; includes CLI start (haiku) and OpenRouter routing (glm).
- Gemini access path is Vertex AI (`llm_backend = vertex:...@global` in every run); prices are the Gemini API list prices from models.yaml.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
