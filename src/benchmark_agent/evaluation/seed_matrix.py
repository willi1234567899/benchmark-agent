from __future__ import annotations

from dataclasses import dataclass
from math import floor, sin
from typing import Literal

LEVELS = ("L1", "L2", "L3", "L4", "L5")
MODELS = ("Gemini", "Modell B", "Modell C")
PROMPTS = ("zero-shot", "multi-shot")
ROW_LABELS = (
    "Gemini ZS",
    "Gemini MS",
    "Modell B ZS",
    "Modell B MS",
    "Modell C ZS",
    "Modell C MS",
)
REF_OPS = 156.0
NChoice = Literal[1, 3, 8]
MetricName = Literal["ratio", "success"]


@dataclass(frozen=True)
class SeedRep:
    syntax_ok: bool
    executed_ok: bool
    n_requests: int
    ops_per_sec: float | None
    ratio: float | None
    in_stage_b: bool


@dataclass(frozen=True)
class CellSummary:
    reps: tuple[SeedRep, ...]
    success_count: int
    success_rate: float
    median_ratio: float | None
    naive_mean_ratio: float
    iqr_ratio: float | None


SEED_EXAMPLE_CELL: tuple[SeedRep, ...] = (
    SeedRep(True, True, 1, 12.4, 0.08, True),
    SeedRep(False, False, 0, None, None, False),
    SeedRep(True, False, 0, None, None, False),
    SeedRep(True, True, 100, 142.0, 0.91, True),
    SeedRep(True, True, 100, 61.0, 0.39, True),
    SeedRep(False, False, 0, None, None, False),
    SeedRep(True, True, 100, 55.0, 0.35, True),
    SeedRep(True, True, 20, 28.0, 0.18, True),
)
L2_GEMINI_ZS = SEED_EXAMPLE_CELL


def unit(seed: float) -> float:
    value = sin(seed * 127.1 + 311.7) * 43758.5453
    return value - floor(value)


def clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def true_ratio(guidance: int, model: int, prompting: int) -> float:
    base = (0.12, 0.4, 0.6, 0.68, 0.9)[guidance]
    model_gap = (0.0, -0.08, -0.05)[model]
    prompt_boost = 0.05 if prompting == 1 else 0.0
    return clamp(base + model_gap + prompt_boost, 0.04, 0.98)


def success_prob(guidance: int, model: int, prompting: int) -> float:
    base = (0.18, 0.5, 0.74, 0.86, 0.96)[guidance]
    model_gap = (0.0, -0.1, -0.06)[model]
    prompt_boost = 0.06 if prompting == 1 else 0.0
    return clamp(base + model_gap + prompt_boost, 0.05, 0.99)


def generate_reps(guidance: int, model: int, prompting: int) -> tuple[SeedRep, ...]:
    if guidance == 1 and model == 0 and prompting == 0:
        return SEED_EXAMPLE_CELL
    reps: list[SeedRep] = []
    for repetition in range(8):
        seed = guidance * 100 + model * 20 + prompting * 8 + repetition
        if unit(seed) >= success_prob(guidance, model, prompting):
            syntax_fail = unit(seed + 3) < (0.7 if guidance == 0 else 0.45)
            reps.append(
                SeedRep(
                    syntax_ok=not syntax_fail,
                    executed_ok=False,
                    n_requests=0,
                    ops_per_sec=None,
                    ratio=None,
                    in_stage_b=False,
                )
            )
            continue
        ratio = clamp(
            true_ratio(guidance, model, prompting) + (unit(seed + 9) - 0.5) * 0.22,
            0.05,
            0.99,
        )
        partial = unit(seed + 11) < (0.55 if guidance == 0 else 0.12)
        if not partial:
            n_requests = 100
        elif unit(seed + 13) < 0.5:
            n_requests = 1
        else:
            n_requests = 20
        ops = round(REF_OPS * ratio * (n_requests / 100) * 10) / 10
        if n_requests == 100:
            observed = ratio
        else:
            observed = clamp(ratio * (n_requests / 100) * 2.2, 0.05, 0.5)
        reps.append(
            SeedRep(
                syntax_ok=True,
                executed_ok=True,
                n_requests=n_requests,
                ops_per_sec=ops,
                ratio=observed,
                in_stage_b=True,
            )
        )
    return tuple(reps)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def iqr(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    ordered = sorted(values)
    q1 = _percentile(ordered, 0.25)
    q3 = _percentile(ordered, 0.75)
    return q3 - q1


def _percentile(ordered: list[float], p: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    rank = p * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(guidance: int, model: int, prompting: int, n: int) -> CellSummary:
    reps = generate_reps(guidance, model, prompting)[:n]
    ratios = [rep.ratio for rep in reps if rep.in_stage_b and rep.ratio is not None]
    naive_ratios = [rep.ratio if rep.ratio is not None else 0.0 for rep in reps]
    success_count = sum(1 for rep in reps if rep.in_stage_b)
    return CellSummary(
        reps=tuple(reps),
        success_count=success_count,
        success_rate=success_count / n,
        median_ratio=median(ratios),
        naive_mean_ratio=sum(naive_ratios) / len(naive_ratios),
        iqr_ratio=iqr(ratios),
    )


def cell_value(
    guidance: int,
    model: int,
    prompting: int,
    n: int,
    metric: MetricName,
) -> float | None:
    summary = summarize(guidance, model, prompting, n)
    if metric == "ratio":
        return summary.median_ratio
    return summary.success_rate


def row_to_model_prompt(row: int) -> tuple[int, int]:
    return divmod(row, 2)


def latency_p95_ratio(throughput_ratio: float) -> float:
    return 1.0 / (0.35 + 0.65 * throughput_ratio)


def server_ops(proxy_ops: float, seed: float) -> float:
    return round(proxy_ops * (1.03 + 0.018 * unit(seed + 21)), 1)
