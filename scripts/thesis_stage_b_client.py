from __future__ import annotations

import datetime
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from functools import cache
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import numpy as np
import thesis_results as tr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as sps

from benchmark_agent.evaluation.runs import SOURCE_PROXY, RunRecord

NAME = "stage_b_client"
PIN = {"write": 10, "read": 20}
OPS_TR = ("write", "read")
MARGIN = (0.8, 1.2)
CONF = 0.90
BOOT_B = 10_000
MODES: tuple[str, ...] = ("session", "fresh", "fresh+timeout")
MODE_LABEL = {
    "session": "persistent session",
    "fresh": "new connection per request",
    "fresh+timeout": "new connection + timeout=",
}
MODE_COLOR = {"session": tr.BASELINE, "fresh": tr.MUTED, "fresh+timeout": tr.INK}
METRICS: dict[str, tuple[str, bool]] = {
    "lat2n": ("latency, requests 2..n (ms)", True),
    "tp_pinned": ("throughput, first 10 writes / 20 reads (ops/s)", False),
    "tp_harness": ("throughput, harness (ops/s)", False),
}

METRIC_SHORT = {
    "lat2n": "latency 2..$n$",
    "tp_pinned": "tp pinned (10/20)",
    "tp_harness": "tp harness (all req.)",
}

RE_COMMENT = re.compile(r"#[^\n]*")
RE_REQUESTS = re.compile(
    r"^\s*(?:import requests\b|from requests\b)|__import__\(\s*['\"]requests['\"]", re.M
)
RE_HTTPX = re.compile(r"^\s*(?:import httpx\b|from httpx\b)", re.M)
RE_HTTPCLIENT = re.compile(r"http\.client|HTTPS?Connection\(")
RE_URLLIB = re.compile(r"urllib\.request|from urllib import[^\n]*\brequest\b|urlopen\(")
RE_SESSION = re.compile(
    r"requests\.Session\(|\bSession\(\)|httpx\.Client\(|context\.\w+\s*=\s*(?:http\.client\.)?HTTPS?Connection\("
)
RE_TIMEOUT = re.compile(r"\btimeout\s*=(?!=)")


def classify_client(steps: str) -> tuple[str, str, bool, bool]:
    code = RE_COMMENT.sub("", steps)
    if RE_HTTPX.search(code):
        lib = "httpx"
    elif RE_REQUESTS.search(code):
        lib = "requests"
    elif RE_HTTPCLIENT.search(code):
        lib = "http.client"
    elif RE_URLLIB.search(code):
        lib = "urllib"
    else:
        lib = "none"
    session = bool(RE_SESSION.search(code))
    timeout = bool(RE_TIMEOUT.search(code))
    if session:
        mode = "session"
    elif timeout:
        mode = "fresh+timeout"
    else:
        mode = "fresh"
    return mode, lib, session, timeout


def _span(reqs: Sequence[dict[str, Any]]) -> float:
    return max(req["ended_at"] for req in reqs) - min(req["started_at"] for req in reqs)


def op_features(reqs: Sequence[dict[str, Any]], op: str) -> dict[str, Any]:
    n = len(reqs)
    out: dict[str, Any] = {"n": n}
    if n == 0:
        return out
    latencies = [float(req["latency_ms"]) for req in reqs]
    gaps = [(reqs[i + 1]["started_at"] - reqs[i]["ended_at"]) * 1000 for i in range(n - 1)]
    out["lat_first"] = latencies[0]
    out["lat_med"] = float(np.median(latencies))
    out["lat2n"] = float(np.median(latencies[1:])) if n >= 2 else None
    out["gap_med"] = float(np.median(gaps)) if gaps else None
    span = _span(reqs)
    out["tp_all"] = n / span if span > 0 else None
    out["lat_share"] = sum(latencies) / 1000 / span if span > 0 else None
    n_pinned = PIN.get(op)
    if n_pinned is not None and n >= n_pinned:
        first = reqs[:n_pinned]
        out["tp_pinned"] = n_pinned / _span(first)
    else:
        out["tp_pinned"] = None
    return out


def build_rows(config: Any, runs: Sequence[RunRecord]) -> list[dict[str, Any]]:
    root = config.paths.project_root
    plan: list[tuple[RunRecord, str, dict[str, Any] | None, str]] = []
    for arm in tr.ARMS:
        for run in tr.arm_runs(runs, arm):
            plan.append((run, arm["key"], arm, run.guidance_level or ""))
    for dialect, epoch in tr.REFERENCE_EPOCHS:
        for run in runs:
            if (
                run.epoch == epoch
                and run.is_fixture
                and (run.condition_name or "").startswith("_gold_")
            ):
                plan.append((run, f"gold {dialect} E{epoch}", None, "gold"))
    rows = []
    for run, key, arm, level in plan:
        run_path = Path(run.run_directory) if run.run_directory else None
        if run_path is not None and not run_path.is_absolute():
            run_path = root / run_path
        steps_path = run_path / "steps.py" if run_path else None
        steps = (
            steps_path.read_text(encoding="utf-8", errors="replace")
            if steps_path and steps_path.exists()
            else ""
        )
        mode, lib, session, timeout = classify_client(steps) if steps else (None, None, None, None)
        reqs: list[dict[str, Any]] = []
        summary_path = run_path / "evaluation" / "summary.json" if run_path else None
        if summary_path is not None and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            reqs = (summary.get("proxy_stats") or {}).get("requests") or []
        gold = arm is None
        row: dict[str, Any] = {
            "run_id": run.run_id,
            "arm": key,
            "model": arm["model"] if arm else "gold",
            "prompting": arm["prompting"] if arm else "",
            "dialect": arm["dialect"] if arm else key.split()[1],
            "epoch": run.epoch,
            "ref": arm["ref"] if arm else key.removeprefix("gold "),
            "level": level,
            "success": bool(run.success),
            "executed_ok": bool(run.executed_ok),
            "stage_a_failed": bool(not gold and not run.success),
            "failure_class": "" if gold else run.failure_class,
            "client_mode": mode,
            "http_lib": lib,
            "session": session,
            "timeout_kw": timeout,
            "run_directory": run.run_directory,
            "eval_date": (run.evaluated_at or "")[:10],
        }
        for op in tr.OPS:
            op_reqs = sorted(
                (req for req in reqs if req.get("operation") == op),
                key=lambda req: req["started_at"],
            )
            features = op_features(op_reqs, op)
            for name, value in features.items():
                row[f"{op}_{name}"] = value
            row[f"{op}_tp_harness"] = run.stage_b_value(SOURCE_PROXY, op, "throughput_ops_s")
            row[f"{op}_p50_harness"] = run.stage_b_value(SOURCE_PROXY, op, "latency_p50_ms")
            row[f"{op}_measured"] = row[f"{op}_tp_harness"] is not None
        row["stage_b_failed"] = bool(
            not gold and run.success and not all(row[f"{op}_measured"] for op in tr.OPS)
        )
        rows.append(row)
    return rows


DETAIL_COLS = [
    "run_id",
    "arm",
    "model",
    "prompting",
    "dialect",
    "epoch",
    "ref",
    "level",
    "success",
    "executed_ok",
    "stage_a_failed",
    "failure_class",
    "write_measured",
    "read_measured",
    "delete_measured",
    "stage_b_failed",
    "client_mode",
    "http_lib",
    "session",
    "timeout_kw",
    *[
        f"{op}_{metric}"
        for op in ("write", "read")
        for metric in (
            "n",
            "gap_med",
            "lat_first",
            "lat2n",
            "lat_share",
            "tp_pinned",
            "tp_all",
            "tp_harness",
            "p50_harness",
        )
    ],
    "delete_n",
    "delete_lat_first",
    "delete_lat_med",
    "delete_tp_harness",
    "eval_date",
    "run_directory",
]


def _rounded(value: Any, digits: int = 4) -> Any:
    return round(value, digits) if isinstance(value, float) else value


@cache
def _u_null_counts(m: int, n: int) -> tuple[int, ...]:
    if m == 0 or n == 0:
        return (1,)
    counts_without_last_x = _u_null_counts(m - 1, n)
    counts_without_last_y = _u_null_counts(m, n - 1)
    size = m * n + 1
    counts = [0] * size
    for u, count in enumerate(counts_without_last_x):
        counts[u + n] += count
    for u, count in enumerate(counts_without_last_y):
        counts[u] += count
    return tuple(counts)


def hl_ci(x: Sequence[float], y: Sequence[float], conf: float = CONF) -> tuple[float, float, float]:
    log_x, log_y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    diffs = np.sort((log_x[:, None] - log_y[None, :]).ravel())
    m, n = len(log_x), len(log_y)
    counts = np.asarray(_u_null_counts(m, n), float)
    cdf = np.cumsum(counts) / counts.sum()
    alpha = (1 - conf) / 2
    rank = int(np.searchsorted(cdf, alpha, side="right"))
    rank = max(rank, 1)
    lo, hi = diffs[rank - 1], diffs[len(diffs) - rank]
    return math.exp(float(np.median(diffs))), math.exp(float(lo)), math.exp(float(hi))


def hl_ci_normal(x: Sequence[float], y: Sequence[float], conf: float = CONF) -> tuple[float, float]:
    log_x, log_y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    diffs = np.sort((log_x[:, None] - log_y[None, :]).ravel())
    m, n = len(log_x), len(log_y)
    z = sps.norm.isf((1 - conf) / 2)
    rank = max(int(math.floor(m * n / 2 - z * math.sqrt(m * n * (m + n + 1) / 12))), 0)
    return math.exp(float(diffs[rank])), math.exp(float(diffs[len(diffs) - 1 - rank]))


def tost_p(x: Sequence[float], y: Sequence[float]) -> float:
    log_x, log_y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    p_low = sps.mannwhitneyu(log_x - math.log(MARGIN[0]), log_y, alternative="greater").pvalue
    p_high = sps.mannwhitneyu(log_x - math.log(MARGIN[1]), log_y, alternative="less").pvalue
    return float(max(p_low, p_high))


def boot_band(values: Sequence[float], rng: np.random.Generator) -> tuple[float, float]:
    sample = np.asarray(values, float)
    median = float(np.median(sample))
    boot_medians = np.median(rng.choice(sample, (BOOT_B, len(sample))), axis=1)
    return (
        float(np.percentile(boot_medians, 2.5) / median),
        float(np.percentile(boot_medians, 97.5) / median),
    )


def signed_z(a: Sequence[float], b: Sequence[float]) -> float:
    p = float(sps.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    z = float(sps.norm.isf(min(max(p, 1e-300), 1.0) / 2))
    return z if float(np.median(a)) > float(np.median(b)) else -z


SUBSETS: tuple[str, ...] = ("measured", "success")
SUBSET_TEX = {"measured": "all meas.", "success": "succ.\\ only"}


def _column_op(col: str) -> str:
    return col.split("_", 1)[0]


def is_measured(row: dict[str, Any], op: str) -> bool:
    return row["model"] == "gold" or row.get(f"{op}_tp_harness") is not None


def member(row: dict[str, Any], subset: str, op: str) -> bool:
    if row["model"] == "gold":
        return True
    if not is_measured(row, op):
        return False
    if subset == "measured":
        return True
    if subset == "success":
        return bool(row["success"])
    if subset == "stage_a_failed":
        return not row["success"]
    raise ValueError(subset)


def measured_any(row: dict[str, Any]) -> bool:
    return any(is_measured(row, op) for op in OPS_TR)


def pick(
    rows: Sequence[dict[str, Any]],
    col: str,
    pred: Callable[[dict[str, Any]], bool],
    subset: str = "measured",
) -> list[dict[str, Any]]:
    op = _column_op(col)
    return [
        row for row in rows if pred(row) and member(row, subset, op) and row.get(col) is not None
    ]


def values(
    rows: Sequence[dict[str, Any]],
    col: str,
    pred: Callable[[dict[str, Any]], bool],
    subset: str = "measured",
) -> list[float]:
    return [float(row[col]) for row in pick(rows, col, pred, subset)]


def gold_rows(rows: Sequence[dict[str, Any]], ref: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["model"] == "gold" and row["ref"] == ref]


def _med(sample: Sequence[float]) -> float | None:
    return float(np.median(sample)) if len(sample) else None


HL_KEYS = (
    "n",
    "median",
    "delta_median_pct",
    "hl_ratio",
    "ci90_lo",
    "ci90_hi",
    "ci90_lo_normal",
    "ci90_hi_normal",
    "tost_p",
    "tost_eq",
    "tost_eq_normal",
    "in_gold_band",
    "share_fresh_timeout",
)


def _hl_stats(
    cell_rows: Sequence[dict[str, Any]], col: str, y: Sequence[float], band: tuple[float, float]
) -> dict[str, Any]:
    out: dict[str, Any] = dict.fromkeys(HL_KEYS)
    x = [float(row[col]) for row in cell_rows]
    out["n"] = len(x)
    if len(x) < 2:
        return out
    hl, lo, hi = hl_ci(x, y)
    normal_lo, normal_hi = hl_ci_normal(x, y)
    ratio_med = float(np.median(x)) / float(np.median(y))
    out.update(
        median=float(np.median(x)),
        delta_median_pct=(ratio_med - 1) * 100,
        hl_ratio=hl,
        ci90_lo=lo,
        ci90_hi=hi,
        ci90_lo_normal=normal_lo,
        ci90_hi_normal=normal_hi,
        tost_p=tost_p(x, y),
        tost_eq=lo > MARGIN[0] and hi < MARGIN[1],
        tost_eq_normal=normal_lo > MARGIN[0] and normal_hi < MARGIN[1],
        in_gold_band=band[0] <= ratio_med <= band[1],
        share_fresh_timeout=sum(row["client_mode"] == "fresh+timeout" for row in cell_rows)
        / len(x),
    )
    return out


def delta_cells(rows: Sequence[dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    bands: dict[tuple[str, str], tuple[float, float]] = {}
    for arm in tr.ARMS:
        if arm["ref"] is None:
            continue
        gold = gold_rows(rows, arm["ref"])
        for level in tr.LEVELS:

            def in_cell(r: dict[str, Any], a: str = arm["key"], lv: str = level) -> bool:
                return r["arm"] == a and r["level"] == lv

            for op in OPS_TR:
                for metric in METRICS:
                    col = f"{op}_{metric}"
                    measured = pick(rows, col, in_cell, "measured")
                    if len(measured) < 2:
                        continue
                    gold_values = values(gold, col, lambda row: True)
                    if (arm["ref"], col) not in bands:
                        bands[(arm["ref"], col)] = boot_band(gold_values, rng)
                    band = bands[(arm["ref"], col)]
                    failed = [row for row in measured if not row["success"]]
                    modes = Counter(row["client_mode"] for row in measured)
                    cell: dict[str, Any] = {
                        "arm": arm["key"],
                        "model": arm["model"],
                        "prompting": arm["prompting"],
                        "dialect": arm["dialect"],
                        "epoch": arm["epoch"],
                        "reference": arm["ref"],
                        "level": level,
                        "op": op,
                        "metric": metric,
                        "n_measured": len(measured),
                        "n_success": len(measured) - len(failed),
                        "n_stage_a_failed": len(failed),
                        "n_stage_b_failed": sum(
                            1
                            for row in rows
                            if in_cell(row) and row["success"] and not is_measured(row, op)
                        ),
                        "n_metric_na": sum(
                            1
                            for row in rows
                            if in_cell(row) and is_measured(row, op) and row.get(col) is None
                        ),
                        "n_ref": len(gold_values),
                        "ref_median": float(np.median(gold_values)),
                        "gold_band_lo": band[0],
                        "gold_band_hi": band[1],
                        "n_session": modes.get("session", 0),
                        "n_fresh": modes.get("fresh", 0),
                        "n_fresh_timeout": modes.get("fresh+timeout", 0),
                        "_failed_vals": [float(row[col]) for row in failed],
                        "_members": [(row["run_id"], bool(row["success"])) for row in measured],
                    }
                    for subset, subset_rows in (
                        ("measured", measured),
                        ("success", [row for row in measured if row["success"]]),
                    ):
                        for stat, value in _hl_stats(subset_rows, col, gold_values, band).items():
                            cell[f"{stat}_{subset}"] = value
                    cells.append(cell)
    for subset in SUBSETS:
        families: dict[tuple[str, str], list[int]] = {}
        for i, cell in enumerate(cells):
            if cell[f"tost_p_{subset}"] is not None:
                families.setdefault((cell["metric"], cell["op"]), []).append(i)
        for cell in cells:
            cell[f"tost_p_holm_{subset}"] = None
            cell[f"tost_eq_holm_{subset}"] = None
        for indices in families.values():
            adjusted = tr.holm([cells[i][f"tost_p_{subset}"] for i in indices])
            for i, p in zip(indices, adjusted, strict=True):
                cells[i][f"tost_p_holm_{subset}"] = p
                cells[i][f"tost_eq_holm_{subset}"] = p is not None and p < 0.05
    return cells


def tost_summary(cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for metric in METRICS:
        for op in OPS_TR:
            metric_cells = [cell for cell in cells if cell["metric"] == metric and cell["op"] == op]
            if not metric_cells:
                continue
            record: dict[str, Any] = {
                "metric": metric,
                "op": op,
                "cells_with_stage_a_failed": sum(
                    cell["n_stage_a_failed"] > 0 for cell in metric_cells
                ),
                "stage_a_failed_runs": sum(cell["n_stage_a_failed"] for cell in metric_cells),
                "stage_b_failed_runs": sum(cell["n_stage_b_failed"] for cell in metric_cells),
            }
            for subset in SUBSETS:
                computed = [cell for cell in metric_cells if cell[f"hl_ratio_{subset}"] is not None]
                equivalent = [cell for cell in computed if cell[f"tost_eq_{subset}"]]
                record[f"cells_{subset}"] = len(computed)
                record[f"tost_eq_exact_{subset}"] = len(equivalent)
                record[f"tost_eq_normal_approx_{subset}"] = sum(
                    bool(cell[f"tost_eq_normal_{subset}"]) for cell in computed
                )
                record[f"tost_eq_holm_{subset}"] = sum(
                    bool(cell[f"tost_eq_holm_{subset}"]) for cell in computed
                )
                record[f"in_gold_band_{subset}"] = sum(
                    bool(cell[f"in_gold_band_{subset}"]) for cell in computed
                )
                record[f"median_abs_delta_pct_{subset}"] = _med(
                    [abs(cell[f"delta_median_pct_{subset}"]) for cell in computed]
                )
                record[f"equivalent_cells_{subset}"] = "; ".join(
                    f"{cell['arm']} {cell['level']}" for cell in equivalent
                )
                record[f"eq_cells_median_slow_share_{subset}"] = _med(
                    [cell[f"share_fresh_timeout_{subset}"] for cell in equivalent]
                )
            out.append(record)
    return out


def mode_deltas(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    groups: list[tuple[str, dict[str, Any] | None]] = [(arm["key"], arm) for arm in tr.ARMS]
    groups += [(f"gold {dialect} E{epoch}", None) for dialect, epoch in tr.REFERENCE_EPOCHS]
    for key, arm in groups:
        ref = arm["ref"] if arm else key.removeprefix("gold ")
        gold = gold_rows(rows, ref) if ref else []
        for mode in (*MODES, "all"):

            def pred(r: dict[str, Any], k: str = key, m: str = mode) -> bool:
                return r["arm"] == k and (m == "all" or r["client_mode"] == m)

            measured = [row for row in rows if pred(row) and measured_any(row)]
            if not measured:
                continue
            n_failed = sum(1 for row in measured if row["model"] != "gold" and not row["success"])
            record: dict[str, Any] = {
                "arm": key,
                "model": arm["model"] if arm else "gold",
                "prompting": arm["prompting"] if arm else "",
                "epoch": arm["epoch"] if arm else int(key.split("E")[-1]),
                "reference": ref,
                "client_mode": mode,
                "n_measured": len(measured),
                "n_success": len(measured) - n_failed,
                "n_stage_a_failed": n_failed,
            }
            for subset in SUBSETS:
                for op in OPS_TR:
                    for metric in ("tp_pinned", "lat2n", "gap_med"):
                        median = _med(values(rows, f"{op}_{metric}", pred, subset))
                        record[f"{op}_{metric}_med_{subset}"] = median
                        if metric == "gap_med":
                            continue
                        gold_values = (
                            values(gold, f"{op}_{metric}", lambda row: True) if gold else []
                        )
                        record[f"{op}_{metric}_delta_pct_{subset}"] = (
                            (median / float(np.median(gold_values)) - 1) * 100
                            if (median is not None and gold_values)
                            else None
                        )
            out.append(record)
    return out


STRATA_METRICS = ("tp_harness", "tp_pinned", "gap_med", "lat2n")


def strata_tests(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons = {
        "vs_all_other": lambda row: row["client_mode"] in ("session", "fresh"),
        "vs_fresh_no_session": lambda row: row["client_mode"] == "fresh",
    }
    detail: list[dict[str, Any]] = []
    long: list[dict[str, Any]] = []
    for comparison, is_other in comparisons.items():
        for subset in SUBSETS:
            for op in OPS_TR:
                for metric in STRATA_METRICS:
                    col = f"{op}_{metric}"
                    z_scores: list[float] = []
                    slower = 0
                    n_strata = 0
                    n_failed = 0
                    for arm in tr.ARMS:
                        for level in tr.LEVELS:

                            def in_cell(
                                r: dict[str, Any], a: str = arm["key"], lv: str = level
                            ) -> bool:
                                return r["arm"] == a and r["level"] == lv

                            timeout_rows = pick(
                                rows,
                                col,
                                lambda row, in_stratum=in_cell: (
                                    in_stratum(row) and row["client_mode"] == "fresh+timeout"
                                ),
                                subset,
                            )
                            other_rows = pick(
                                rows,
                                col,
                                lambda row, in_stratum=in_cell, other=is_other: (
                                    in_stratum(row) and other(row)
                                ),
                                subset,
                            )
                            if not timeout_rows or not other_rows:
                                continue
                            timeout_values = [float(row[col]) for row in timeout_rows]
                            other_values = [float(row[col]) for row in other_rows]
                            diff = float(np.median(timeout_values)) - float(np.median(other_values))
                            is_slower = diff < 0 if metric.startswith("tp") else diff > 0
                            z = signed_z(timeout_values, other_values)
                            n_failed_stratum = sum(
                                not row["success"] for row in (*timeout_rows, *other_rows)
                            )
                            n_strata += 1
                            slower += int(is_slower)
                            n_failed += n_failed_stratum
                            z_scores.append(z)
                            detail.append(
                                {
                                    "comparison": comparison,
                                    "subset": subset,
                                    "op": op,
                                    "metric": metric,
                                    "arm": arm["key"],
                                    "level": level,
                                    "n_timeout": len(timeout_values),
                                    "n_other": len(other_values),
                                    "n_stage_a_failed": n_failed_stratum,
                                    "median_timeout": float(np.median(timeout_values)),
                                    "median_other": float(np.median(other_values)),
                                    "timeout_slower": is_slower,
                                    "z": z,
                                }
                            )
                    if not n_strata:
                        continue
                    z_combined = sum(z_scores) / math.sqrt(n_strata)
                    long.append(
                        {
                            "comparison": comparison,
                            "subset": subset,
                            "op": op,
                            "metric": metric,
                            "strata": n_strata,
                            "stage_a_failed_runs": n_failed,
                            "timeout_slower": slower,
                            "sign_test_p": float(sps.binomtest(slower, n_strata).pvalue),
                            "stouffer_z": z_combined,
                            "stouffer_p": float(2 * sps.norm.sf(abs(z_combined))),
                        }
                    )
    families: dict[tuple[str, str], list[int]] = {}
    for i, test in enumerate(long):
        families.setdefault((test["comparison"], test["subset"]), []).append(i)
    for indices in families.values():
        adjusted = tr.holm([long[i]["sign_test_p"] for i in indices])
        for i, p in zip(indices, adjusted, strict=True):
            long[i]["sign_test_p_holm"] = p
    wide: dict[tuple[str, str, str], dict[str, Any]] = {}
    for test in long:
        record = wide.setdefault(
            (test["comparison"], test["op"], test["metric"]),
            {"comparison": test["comparison"], "op": test["op"], "metric": test["metric"]},
        )
        for column in (
            "strata",
            "stage_a_failed_runs",
            "timeout_slower",
            "sign_test_p",
            "sign_test_p_holm",
            "stouffer_z",
            "stouffer_p",
        ):
            record[f"{column}_{test['subset']}"] = test[column]
    return detail, list(wide.values())


def gap_by_style(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, bool, bool], list[dict[str, Any]]] = {}
    for row in rows:
        if row["model"] == "gold" or row["http_lib"] is None or not measured_any(row):
            continue
        style = (row["http_lib"], bool(row["session"]), bool(row["timeout_kw"]))
        groups.setdefault(style, []).append(row)
    gold = [row for row in rows if row["model"] == "gold"]
    style_metrics = ("gap_med", "tp_harness", "tp_pinned", "lat2n")
    out = []
    for (lib, session, timeout), members in sorted(groups.items()):
        n_failed = sum(not row["success"] for row in members)
        record: dict[str, Any] = {
            "group": "generated",
            "http_lib": lib,
            "session": session,
            "timeout_kw": timeout,
            "n_measured": len(members),
            "n_success": len(members) - n_failed,
            "n_stage_a_failed": n_failed,
        }
        for subset in SUBSETS:
            for op in OPS_TR:
                for metric in style_metrics:
                    record[f"{op}_{metric}_{subset}"] = _med(
                        values(members, f"{op}_{metric}", lambda row: True, subset)
                    )
        out.append(record)
    record = {"group": "gold", "http_lib": "urllib", "session": False, "timeout_kw": True}
    record.update(n_measured=len(gold), n_success=len(gold), n_stage_a_failed=0)
    for subset in SUBSETS:
        for op in OPS_TR:
            for metric in style_metrics:
                record[f"{op}_{metric}_{subset}"] = _med(
                    values(gold, f"{op}_{metric}", lambda row: True)
                )
    out.append(record)
    return out


PCT_TICKS = (-50, 0, 100)
PCT_MINOR = (-25, 25, 50, 200)


def _view_label(n_runs: int, n_failed: int, view: str) -> str:
    if view == "all":
        failed = (
            f"{n_failed} of them failed Stage A (\u00d7)"
            if n_failed
            else "none of them failed Stage A"
        )
        return f"All measured runs: {n_runs} runs, {failed}"
    if view == "success":
        return f"Stage-A successes only: {n_runs} runs"
    raise ValueError(view)


def _view_header(n_runs: int, n_failed: int, view: str, n_gold: int, gold_note: str) -> str:
    return f"{_view_label(n_runs, n_failed, view)}, plus {n_gold} reference runs ({gold_note})"


def _delta_cell(
    cells: Sequence[dict[str, Any]], arm: str, level: str, op: str, metric: str
) -> dict[str, Any] | None:
    return next(
        (
            cell
            for cell in cells
            if cell["arm"] == arm
            and cell["level"] == level
            and cell["op"] == op
            and cell["metric"] == metric
        ),
        None,
    )


DELTA_PANELS: tuple[tuple[str, str], ...] = tuple(
    (metric, op) for metric in ("lat2n", "tp_pinned") for op in OPS_TR
)


def latency_delta_counts(
    cells: Sequence[dict[str, Any]], dialect: str, view: str
) -> tuple[int, int, list[str]]:
    subset = tr.VIEW_SUBSET[view]
    success_by_run: dict[Any, bool] = {}
    refs: set[str] = set()
    for cell in cells:
        if cell["dialect"] != dialect or (cell["metric"], cell["op"]) not in DELTA_PANELS:
            continue
        if cell[f"hl_ratio_{subset}"] is None:
            continue
        refs.add(cell["reference"])
        for run_id, succeeded in cell["_members"]:
            if view == "all" or succeeded:
                success_by_run[run_id] = succeeded
    return (
        len(success_by_run),
        sum(not succeeded for succeeded in success_by_run.values()),
        sorted(refs),
    )


def fig_latency_delta(
    cells: Sequence[dict[str, Any]],
    dialect: str,
    view: str,
    out: Path,
    gold_n: dict[str, int],
) -> str:
    subset = tr.VIEW_SUBSET[view]
    arms = [arm for arm in tr.ARMS if arm["dialect"] == dialect and arm["ref"] is not None]
    levels = [
        level
        for level in tr.LEVELS
        if any(cell["level"] == level and cell["dialect"] == dialect for cell in cells)
    ]
    step = 0.8 / len(arms)
    offsets = np.linspace(-0.4 + step / 2, 0.4 - step / 2, len(arms))
    fig, axes = plt.subplots(1, 4, figsize=(tr.TEXT_WIDTH_IN, 4.4), sharey=True)
    lo_lim, hi_lim = math.log(0.4), math.log(3.2)

    def clip(v: float) -> float:
        return min(max(math.log(v), lo_lim), hi_lim)

    any_failed = False
    for ax, (metric, op) in zip(axes, DELTA_PANELS, strict=True):
        ax.axvspan(math.log(MARGIN[0]), math.log(MARGIN[1]), color="#ecebe4", zorder=0, lw=0)
        for margin in MARGIN:
            ax.axvline(math.log(margin), color=tr.BASELINE, lw=0.7, ls=(0, (2, 2)), zorder=1)
        for level_index, level in enumerate(levels):
            if level_index:
                ax.axhline(level_index - 0.5, color=tr.GRID, lw=0.6, zorder=0)
            for arm, offset in zip(arms, offsets, strict=True):
                cell = _delta_cell(cells, arm["key"], level, op, metric)
                if cell is None or cell[f"hl_ratio_{subset}"] is None:
                    continue
                y = level_index + offset
                ax.scatter(
                    [math.log(cell["gold_band_lo"]), math.log(cell["gold_band_hi"])],
                    [y, y],
                    marker="|",
                    s=14,
                    color=tr.INK_2,
                    linewidth=0.8,
                    zorder=2,
                )
                color = tr.MODEL_COLOR[arm["model"]]
                equivalent = bool(cell[f"tost_eq_{subset}"])
                ax.plot(
                    [clip(cell[f"ci90_lo_{subset}"]), clip(cell[f"ci90_hi_{subset}"])],
                    [y, y],
                    color=color,
                    lw=1.9 if equivalent else 0.8,
                    alpha=0.95 if equivalent else 0.6,
                    solid_capstyle="butt",
                    zorder=2,
                )
                for end, limit, arrow in (("lo", lo_lim, "<"), ("hi", hi_lim, ">")):
                    log_end = math.log(cell[f"ci90_{end}_{subset}"])
                    if (end == "lo" and log_end < limit) or (end == "hi" and log_end > limit):
                        ax.scatter(
                            [limit],
                            [y],
                            marker=arrow,
                            s=12,
                            color=color,
                            linewidth=0,
                            zorder=3,
                            clip_on=False,
                        )
                ax.scatter(
                    [clip(cell[f"hl_ratio_{subset}"])],
                    [y],
                    marker=tr.PROMPT_MARKER[arm["prompting"]],
                    s=15,
                    color=color,
                    edgecolor=tr.SURFACE,
                    linewidth=0.4,
                    zorder=4,
                )
                if view == "all" and cell["n_stage_a_failed"]:
                    any_failed = True
                    ax.scatter(
                        [clip(value / cell["ref_median"]) for value in cell["_failed_vals"]],
                        [y] * len(cell["_failed_vals"]),
                        marker="x",
                        s=11,
                        color=color,
                        linewidth=0.7,
                        zorder=3,
                    )
        ax.axvline(0, color=tr.INK_2, lw=0.9, ls=(0, (4, 2)), zorder=1)
        ax.set_xlim(lo_lim, hi_lim)
        ax.set_xticks(
            [math.log(1 + pct / 100) for pct in PCT_TICKS],
            [f"{pct:+d}" if pct else "0" for pct in PCT_TICKS],
        )
        ax.set_xticks([math.log(1 + pct / 100) for pct in PCT_MINOR], minor=True)
        ax.tick_params(axis="x", which="minor", length=2)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.set_yticks(range(len(levels)), levels)
        ax.grid(False)
        ax.xaxis.grid(True, which="both", color=tr.GRID, lw=0.4)
        ax.tick_params(axis="y", length=0)
        ax.spines["bottom"].set_color(tr.BASELINE)
        ax.set_title(f"{'latency 2..n' if metric == 'lat2n' else 'pinned tp'} · {op}")
        ax.set_xlabel("HL ratio vs ref. (%)", fontsize=7)
    axes[0].set_ylim(len(levels) - 0.5, -0.5)
    handles: list[Any] = [
        Line2D(
            [],
            [],
            marker=tr.PROMPT_MARKER[arm["prompting"]],
            color=tr.MODEL_COLOR[arm["model"]],
            markerfacecolor=tr.MODEL_COLOR[arm["model"]],
            linestyle="-",
            lw=0.8,
            markersize=4.5,
            label=f"{arm['short']} (E{arm['epoch']}, ref. {arm['ref'].split()[1]})",
        )
        for arm in arms
    ]
    handles.append(Line2D([], [], color=tr.INK_2, lw=1.9, label="thick CI = TOST-equivalent"))
    if any_failed:
        handles.append(
            Line2D(
                [],
                [],
                marker="x",
                color=tr.INK_2,
                linestyle="",
                markersize=4.5,
                label="run failing Stage A (value / reference median)",
            )
        )
    handles += [
        Patch(
            facecolor="#ecebe4",
            edgecolor=tr.BASELINE,
            lw=0.5,
            ls=(0, (2, 2)),
            label="±20 % TOST margin",
        ),
        Line2D(
            [],
            [],
            marker="|",
            color=tr.INK_2,
            linestyle="",
            markersize=6,
            label="95 % bootstrap band of the reference median",
        ),
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0), handlelength=1.8
    )
    fig.subplots_adjust(wspace=0.1)
    n, n_failed, refs = latency_delta_counts(cells, dialect, view)
    tr._header(
        fig,
        _view_header(
            n,
            n_failed,
            view,
            sum(gold_n[ref] for ref in refs),
            ", ".join(f"E{ref.split('E')[-1]}" for ref in refs),
        ),
    )
    return tr.save(fig, out, f"stage_b_latency_delta_{dialect.lower()}_{view}")


def client_mode_counts(rows: Sequence[dict[str, Any]], view: str) -> tuple[int, int]:
    measured = [
        row for row in rows if row["model"] != "gold" and measured_any(row) and row["client_mode"]
    ]
    failed = sum(not row["success"] for row in measured)
    return (len(measured) if view == "all" else len(measured) - failed), failed


def fig_client_mode(rows: Sequence[dict[str, Any]], view: str, out: Path) -> str:
    subset = tr.VIEW_SUBSET[view]
    groups: list[tuple[str, str, str]] = []
    for arm in tr.ARMS:
        groups.append(
            (
                arm["key"],
                f"{arm['short']} · {arm['dialect'][0]} (E{arm['epoch']})",
                tr.MODEL_COLOR[arm["model"]],
            )
        )
    for dialect, epoch in tr.REFERENCE_EPOCHS:
        groups.append((f"gold {dialect} E{epoch}", f"reference · {dialect[0]} (E{epoch})", tr.INK))
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(tr.TEXT_WIDTH_IN, 4.5),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )
    y_positions = []
    y = 0.0
    for i, (key, _, _) in enumerate(groups):
        if i and key.startswith("gold") and not groups[i - 1][0].startswith("gold"):
            y += 0.6
        y_positions.append(y)
        y += 1
    any_failed = False
    for (key, _, color), y_row in zip(groups, y_positions, strict=True):
        members = [
            row for row in rows if row["arm"] == key and measured_any(row) and row["client_mode"]
        ]
        failed = [row for row in members if row["model"] != "gold" and not row["success"]]
        if view == "success":
            members = [row for row in members if row not in failed]
            failed = []
        any_failed = any_failed or bool(failed)
        left = 0.0
        for mode in MODES:
            n_ok = sum(row["client_mode"] == mode and row not in failed for row in members)
            n_bad = sum(row["client_mode"] == mode for row in failed)
            for n_part, hatched in ((n_ok, False), (n_bad, True)):
                share = n_part / len(members) if members else 0
                if not share:
                    continue
                ax1.barh(
                    y_row,
                    share,
                    left=left,
                    height=0.72,
                    color=MODE_COLOR[mode],
                    edgecolor=color if hatched else MODE_COLOR[mode],
                    hatch="xxxx" if hatched else None,
                    lw=0,
                )
                left += share
        label = f"{len(members)}" + (f" ({len(failed)}†)" if failed else "")
        ax1.text(1.02, y_row, label, va="center", ha="left", fontsize=6.5, color=tr.INK_2)
        for op, marker, dy in (("write", "o", -0.17), ("read", "^", 0.17)):
            gap_col = f"{op}_gap_med"
            for mode in MODES:

                def in_mode(r: dict[str, Any], m: str = mode, k: str = key) -> bool:
                    return r["arm"] == k and r["client_mode"] == m

                gaps = values(rows, gap_col, in_mode, subset)
                if not gaps:
                    continue
                ax2.scatter(
                    [float(np.median(gaps))],
                    [y_row + dy],
                    marker=marker,
                    s=12 + 2 * min(len(gaps), 40) ** 0.5 * 3,
                    color=MODE_COLOR[mode],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
                if view != "all" or key.startswith("gold"):
                    continue
                failed_gaps = values(rows, gap_col, in_mode, "stage_a_failed")
                if failed_gaps:
                    ax2.scatter(
                        failed_gaps,
                        [y_row + dy] * len(failed_gaps),
                        marker="x",
                        s=10,
                        color=color,
                        linewidth=0.7,
                        zorder=4,
                    )
    ax1.set_yticks(y_positions, [group[1] for group in groups])
    for tick_label, (_, _, color) in zip(ax1.get_yticklabels(), groups, strict=True):
        tick_label.set_color(color)
    ax1.set_xlim(0, 1)
    ax1.set_xticks([0, 0.5, 1], ["0", "50", "100"])
    ax1.set_xlabel(
        "share of measured runs (%)\n[n at right, † = failing Stage A]"
        if view == "all"
        else "share of Stage-A successes (%)\n[n at right]"
    )
    ax1.grid(False)
    ax1.tick_params(axis="y", length=0)
    ax1.invert_yaxis()
    ax2.set_xscale("log")
    ax2.set_xlim(1, 80)
    ax2.set_xticks([1, 2, 5, 10, 20, 50], ["1", "2", "5", "10", "20", "50"])
    ax2.grid(False)
    ax2.xaxis.grid(True, color=tr.GRID, lw=0.5)
    for y_row in y_positions:
        ax2.axhline(y_row, color=tr.GRID, lw=0.3, zorder=0)
    ax2.set_xlabel("median client gap between\nrequests (ms, log)")
    ax2.tick_params(axis="y", length=0)
    ax1.set_title("client mode")
    ax2.set_title("client gap per mode")
    handles: list[Any] = [
        Patch(facecolor=MODE_COLOR[mode], label=MODE_LABEL[mode]) for mode in MODES
    ]
    handles += [
        Line2D(
            [],
            [],
            marker="o",
            color=tr.INK_2,
            markerfacecolor=tr.INK_2,
            linestyle="",
            markersize=4,
            label="write gap (median)",
        ),
        Line2D(
            [],
            [],
            marker="^",
            color=tr.INK_2,
            markerfacecolor=tr.INK_2,
            linestyle="",
            markersize=4,
            label="read gap (median)",
        ),
    ]
    if any_failed:
        handles += [
            Patch(
                facecolor=tr.MUTED,
                edgecolor=tr.INK_2,
                hatch="xxxx",
                lw=0,
                label="hatched: runs failing Stage A (kept)",
            ),
            Line2D(
                [],
                [],
                marker="x",
                color=tr.INK_2,
                linestyle="",
                markersize=4,
                label="gap of a run failing Stage A",
            ),
        ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(wspace=0.22)
    n, n_failed = client_mode_counts(rows, view)
    n_gold = sum(1 for row in rows if row["model"] == "gold" and measured_any(row))
    tr._header(fig, _view_header(n, n_failed, view, n_gold, f"{len(tr.REFERENCE_EPOCHS)} series"))
    return tr.save(fig, out, f"stage_b_client_mode_{view}")


def _arm_cell(key: str) -> str:
    if key.startswith("gold"):
        return f"gold {key.split()[1][0]} (E{key.split('E')[-1]})"
    arm = tr.ARM_BY_KEY[key]
    prompting = "zs" if arm["prompting"] == "zero_shot" else "ms"
    return f"{tr.MODEL_LABEL[arm['model']]} {prompting} {arm['dialect'][0]} (E{arm['epoch']})"


def _pct(value: float | None) -> str:
    if value is None:
        return "--"
    text = f"{value:+.0f}"
    return "+0" if text == "-0" else text


def _mode_cols(mode_row: dict[str, Any], subset: str) -> list[str]:
    return [
        tr.fmt(mode_row[f"write_tp_pinned_med_{subset}"]),
        _pct(mode_row[f"write_tp_pinned_delta_pct_{subset}"]),
        _pct(mode_row[f"write_lat2n_delta_pct_{subset}"]),
        tr.fmt(mode_row[f"write_gap_med_med_{subset}"]),
        tr.fmt(mode_row[f"read_tp_pinned_med_{subset}"]),
        _pct(mode_row[f"read_tp_pinned_delta_pct_{subset}"]),
        _pct(mode_row[f"read_lat2n_delta_pct_{subset}"]),
        tr.fmt(mode_row[f"read_gap_med_med_{subset}"]),
    ]


def tex_mode_deltas(mrows: Sequence[dict[str, Any]]) -> str:
    table: list[list[str]] = []
    keys = [arm["key"] for arm in tr.ARMS if arm["ref"] is not None]
    keys += [f"gold {dialect} E{epoch}" for dialect, epoch in tr.REFERENCE_EPOCHS]
    starts: list[int] = []
    for key in keys:
        arm_mode_rows = [
            mode_row
            for mode_row in mrows
            if mode_row["arm"] == key and mode_row["client_mode"] != "all"
        ]
        if not arm_mode_rows:
            continue
        starts.append(len(table))
        for i, mode_row in enumerate(arm_mode_rows):
            table.append(
                [
                    _arm_cell(key) if i == 0 else "",
                    tr.tex(mode_row["client_mode"]),
                    str(mode_row["n_measured"]),
                    str(mode_row["n_stage_a_failed"]) if mode_row["n_stage_a_failed"] else "",
                    *_mode_cols(mode_row, "measured"),
                ]
            )
            if mode_row["n_stage_a_failed"]:
                table.append(
                    [
                        "",
                        r"\quad succ.\ only",
                        str(mode_row["n_success"]),
                        "",
                        *[f"\\textit{{{text}}}" for text in _mode_cols(mode_row, "success")],
                    ]
                )
    header = [
        "Arm",
        "Mode",
        "$n$",
        r"$\dagger$",
        "tp$_w$",
        r"$\Delta$tp$_w$",
        r"$\Delta$lat$_w$",
        "gap$_w$",
        "tp$_r$",
        r"$\Delta$tp$_r$",
        r"$\Delta$lat$_r$",
        "gap$_r$",
    ]
    return tr.latex_table(
        "Stage B by HTTP client mode (draft). Main-matrix arms with a same-day reference (flash multi-shot E12/E13, haiku E14/E15, glm E16/E17; flash zero-shot E8/E9 has none) and the four gold series. Per arm and client mode: all measured runs $n$ (levels pooled within the arm's epoch), of which $\\dagger$ failed Stage A after executing; median throughput over the first 10 writes / 20 reads (tp, ops/s), its deviation $\\Delta$tp from the same-day gold median, the deviation $\\Delta$lat of the median per-request latency of requests 2..$n$ (positive = slower), and the median client gap between requests (ms). Where $\\dagger > 0$, an italic row repeats the values for the Stage-A successes only.",
        "tab:stage-b-client-mode",
        header,
        table,
        "llrrrrrrrrrr",
        note="session = persistent \\texttt{requests.Session}/\\texttt{httpx.Client}; fresh = new connection per request without a socket timeout; fresh+timeout = new connection with \\texttt{timeout=}. The gold reference itself is fresh+timeout (urllib, \\texttt{timeout=30}), so its rows show 0 by construction. $\\Delta$ in percent; gold rows give the gold medians. Pinned throughput and per-request latency are two separate indicators; neither is ranked above the other. Descriptive (no test). "
        + tr.ARM_LEGEND,
        size=r"\scriptsize",
        midrules_after=tuple(start - 1 for start in starts[1:]),
        tabcolsep="3pt",
    )


def tex_tost_summary(summary: Sequence[dict[str, Any]]) -> str:
    table: list[list[str]] = []
    starts = []
    for row in summary:
        starts.append(len(table))
        for i, subset in enumerate(SUBSETS):
            table.append(
                [
                    METRIC_SHORT[row["metric"]] if i == 0 else "",
                    row["op"] if i == 0 else "",
                    SUBSET_TEX[subset],
                    str(row[f"cells_{subset}"]),
                    str(row["stage_a_failed_runs"]) if subset == "measured" else "--",
                    str(row["stage_b_failed_runs"]),
                    tr.fmt(row[f"median_abs_delta_pct_{subset}"]),
                    str(row[f"in_gold_band_{subset}"]),
                    str(row[f"tost_eq_exact_{subset}"]),
                    str(row[f"tost_eq_holm_{subset}"]),
                ]
            )
    return tr.latex_table(
        "Equivalence with the same-day reference per arm $\\times$ level cell (draft), main-matrix arms with a same-day reference only (flash multi-shot E12/E13 vs gold of 13 Sep 2026; haiku E14/E15 and glm E16/E17 vs gold E16/E17 of 9 Oct 2026). TOST at $\\pm$20\\,\\% on the ratio of the cell to the gold series: Hodges-Lehmann ratio with exact distribution-free 90\\,\\% CI, cells with $n \\geq 2$. Each indicator is given for all measured runs and for the Stage-A successes only.",
        "tab:stage-b-tost",
        [
            "Indicator",
            "Op.",
            "Runs",
            "Cells",
            r"$\dagger$",
            "S-B fail",
            r"med.\ $|\Delta|$ (\%)",
            r"in ref.\ band",
            "TOST eq.",
            "Holm",
        ],
        table,
        "lllrrrrrrr",
        note="all meas.\\ = every run with a proxy value for the operation; succ.\\ only = Stage-A successes. $\\dagger$ = measured runs that failed Stage A (kept in all meas.). S-B fail = Stage-A successes without a value for the operation (Stage-B failures). Ref.\\ band = 95\\,\\% percentile bootstrap band of the reference median. TOST eq.\\ = 90\\,\\% CI inside [0.8, 1.2]; Holm = TOST $p$ (max of two shifted one-sided Mann-Whitney tests) Holm-adjusted over the cells of one indicator, operation and run set. tp harness counts every request the run sent (some glm runs send more than 20 reads); tp pinned uses the first 10 writes / 20 reads; latency 2..$n$ is the median per-request latency without the cold first request.",
        size=r"\scriptsize",
        midrules_after=tuple(start - 1 for start in starts[1:]),
        tabcolsep="2.5pt",
    )


def tex_strata(summary: Sequence[dict[str, Any]]) -> str:
    table: list[list[str]] = []
    starts = []
    for row in summary:
        if row["comparison"] != "vs_all_other":
            continue
        starts.append(len(table))
        for i, subset in enumerate(SUBSETS):
            if f"strata_{subset}" not in row:
                continue
            table.append(
                [
                    row["op"] if i == 0 else "",
                    tr.tex(row["metric"]).replace("\\_", " ") if i == 0 else "",
                    SUBSET_TEX[subset],
                    str(row[f"stage_a_failed_runs_{subset}"]) if subset == "measured" else "--",
                    f"{row[f'timeout_slower_{subset}']}/{row[f'strata_{subset}']}",
                    tr.fmt_p(row[f"sign_test_p_{subset}"]),
                    tr.fmt_p(row[f"sign_test_p_holm_{subset}"]),
                    f"{row[f'stouffer_z_{subset}']:+.2f}",
                    tr.fmt_p(row[f"stouffer_p_{subset}"]),
                ]
            )
    return tr.latex_table(
        "Client mode inside arm $\\times$ level strata of the 12 main-matrix arms (draft): runs with a new connection and \\texttt{timeout=} against all other runs of the same stratum, for all measured runs and for the Stage-A successes only. Slower = lower throughput or larger gap/latency.",
        "tab:stage-b-timeout-strata",
        [
            "Op.",
            "Indicator",
            "Runs",
            r"$\dagger$",
            "slower",
            "sign test $p$",
            "Holm",
            "Stouffer $Z$",
            "$p$",
        ],
        table,
        "lllrrrrrr",
        note="Strata with at least one run in each group. $\\dagger$ = runs failing Stage A inside the compared strata (kept in all meas.). Sign test two-sided over strata, Holm over the 8 rows of one run set; Stouffer combines the signed two-sided Mann-Whitney $z$ per stratum with equal weights (exploratory, uncorrected). tp harness = all requests, tp pinned = first 10 writes / 20 reads, gap = median client gap, lat2n = median latency of requests 2..$n$.",
        size=r"\scriptsize",
        midrules_after=tuple(start - 1 for start in starts[1:]),
        tabcolsep="2.5pt",
    )


def _csv(path: Path, records: Sequence[dict[str, Any]], digits: int = 4) -> None:
    all_keys = dict.fromkeys(key for record in records for key in record)
    cols = [key for key in all_keys if not key.startswith("_")]
    tr.write_csv(
        path, cols, [[_rounded(record.get(col), digits) for col in cols] for record in records]
    )


def main() -> None:
    tr._style()
    config, runs = tr.load_runs()
    base = config.paths.project_root / "results" / "reports" / "thesis_results" / "extra" / NAME
    fig_dir, tab_dir, tex_dir = base / "fig", base / "tables", base / "latex"
    for directory in (fig_dir, tab_dir, tex_dir):
        directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(tr.JITTER_SEED)

    rows = build_rows(config, runs)
    tr.write_csv(
        tab_dir / "stage_b_runs_detail.csv",
        DETAIL_COLS,
        [[_rounded(row.get(col)) for col in DETAIL_COLS] for row in rows],
    )

    n_mismatch = sum(
        1
        for row in rows
        for op in tr.OPS
        if row.get(f"{op}_tp_harness")
        and row.get(f"{op}_tp_all")
        and abs(row[f"{op}_tp_harness"] / row[f"{op}_tp_all"] - 1) > 1e-6
    )
    n_checked = sum(
        1
        for row in rows
        for op in tr.OPS
        if row.get(f"{op}_tp_harness") and row.get(f"{op}_tp_all")
    )

    cells = delta_cells(rows, rng)
    _csv(tab_dir / "delta_latency_vs_reference.csv", cells)
    tost_rows = tost_summary(cells)
    _csv(tab_dir / "tost_summary.csv", tost_rows)
    mode_rows = mode_deltas(rows)
    _csv(tab_dir / "stage_b_delta_by_client_mode.csv", mode_rows)
    strata_detail, strata_summary = strata_tests(rows)
    _csv(tab_dir / "timeout_strata_detail.csv", strata_detail)
    _csv(tab_dir / "timeout_strata_tests.csv", strata_summary, 6)
    styles = gap_by_style(rows)
    _csv(tab_dir / "client_gap_by_style.csv", styles)

    gold_n = Counter(row["ref"] for row in rows if row["model"] == "gold")
    for stale in STALE_FIGS:
        for ext in ("pdf", "png"):
            (fig_dir / f"{stale}.{ext}").unlink(missing_ok=True)
    figs: dict[tuple[str, str], str] = {}
    for view in tr.VIEWS:
        for dialect in tr.DIALECTS:
            figs[(dialect, view)] = fig_latency_delta(cells, dialect, view, fig_dir, gold_n)
        figs[("client", view)] = fig_client_mode(rows, view, fig_dir)

    (tex_dir / "stage_b_delta_by_client_mode.tex").write_text(
        tex_mode_deltas(mode_rows), encoding="utf-8"
    )
    (tex_dir / "stage_b_tost.tex").write_text(tex_tost_summary(tost_rows), encoding="utf-8")
    (tex_dir / "stage_b_timeout_strata.tex").write_text(
        tex_strata(strata_summary), encoding="utf-8"
    )
    (tex_dir / "figures.tex").write_text(
        tr.tex_figures(figure_captions(rows, cells, figs)).replace("{fig/", f"{{extra/{NAME}/fig/"),
        encoding="utf-8",
    )

    key_lines = print_headlines(
        rows, cells, tost_rows, mode_rows, strata_summary, styles, n_mismatch, n_checked
    )
    write_readme(
        base,
        figs,
        key_lines,
        readme_facts(rows, cells, tost_rows, strata_summary, mode_rows),
    )


def _fmt_fixed(value: float | None, digits: int = 1) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def print_headlines(
    rows: Sequence[dict[str, Any]],
    cells: Sequence[dict[str, Any]],
    tsum: Sequence[dict[str, Any]],
    mrows: Sequence[dict[str, Any]],
    ssum: Sequence[dict[str, Any]],
    styles: Sequence[dict[str, Any]],
    mism: int,
    checked: int,
) -> list[str]:
    lines: list[str] = []
    main_rows = [row for row in rows if row["model"] != "gold"]
    measured = [row for row in main_rows if measured_any(row)]
    succeeded = [row for row in measured if row["success"]]
    failed = [row for row in measured if not row["success"]]
    n_stage_b_failed = sum(1 for row in main_rows if row["success"] and row["stage_b_failed"])
    lines.append(
        f"runs: {len(main_rows)} main-matrix + {len(rows) - len(main_rows)} gold; measured main (write or read value): {len(measured)} = {len(succeeded)} Stage-A successes + {len(failed)} Stage-A failures (kept, marked); Stage-B failures (success without a value for an op): {n_stage_b_failed}; harness tp == n/span in {checked - mism}/{checked} run x op rows"
    )
    failed_cells = Counter((row["arm"], row["level"]) for row in failed)
    failure_classes = Counter(row["failure_class"] for row in failed)
    lines.append(
        "Stage-A-failed measured runs per cell: "
        + ", ".join(f"{arm} {level} {n}" for (arm, level), n in sorted(failed_cells.items()))
        + "; failure classes: "
        + ", ".join(f"{failure_class} {n}" for failure_class, n in failure_classes.most_common())
    )
    for label, group in (
        ("measured", measured),
        ("success", succeeded),
        ("stage_a_failed", failed),
    ):
        modes = Counter(row["client_mode"] for row in group)
        libs = Counter(row["http_lib"] for row in group)
        lines.append(
            f"client mode ({label} main runs): "
            + ", ".join(f"{mode} {n}" for mode, n in modes.most_common())
            + "; http_lib: "
            + ", ".join(f"{lib} {n}" for lib, n in libs.most_common())
        )
    gold_modes = Counter(
        (row["ref"], row["client_mode"], row["http_lib"]) for row in rows if row["model"] == "gold"
    )
    lines.append(
        "gold client: "
        + ", ".join(
            f"{ref} {mode}/{lib} x{n}" for (ref, mode, lib), n in sorted(gold_modes.items())
        )
    )
    for style in styles:
        if (
            style["group"] == "generated"
            and style["http_lib"] == "requests"
            and not style["session"]
        ):
            lines.append(
                f"requests, no session, timeout={style['timeout_kw']}: n={style['n_measured']} (success {style['n_success']}, Stage-A failed {style['n_stage_a_failed']}) write gap {_fmt_fixed(style['write_gap_med_measured'])} / {_fmt_fixed(style['write_gap_med_success'])} ms, harness write tp {_fmt_fixed(style['write_tp_harness_measured'])} / {_fmt_fixed(style['write_tp_harness_success'])}, pinned {_fmt_fixed(style['write_tp_pinned_measured'])} / {_fmt_fixed(style['write_tp_pinned_success'])}; read gap {_fmt_fixed(style['read_gap_med_measured'])} / {_fmt_fixed(style['read_gap_med_success'])} ms  [measured / success]"
            )
        if style["group"] == "gold":
            lines.append(
                f"gold (urllib+timeout): write gap {style['write_gap_med_measured']:.1f} ms, lat2n {style['write_lat2n_measured']:.2f} ms; read gap {style['read_gap_med_measured']:.1f} ms"
            )
    for tost_row in tsum:
        parts = []
        for subset in SUBSETS:
            parts.append(
                f"{subset}: {tost_row[f'tost_eq_exact_{subset}']}/{tost_row[f'cells_{subset}']} (normal {tost_row[f'tost_eq_normal_approx_{subset}']}, Holm {tost_row[f'tost_eq_holm_{subset}']}), band {tost_row[f'in_gold_band_{subset}']}, med|D| {_fmt_fixed(tost_row[f'median_abs_delta_pct_{subset}'])}% [{tost_row[f'equivalent_cells_{subset}']}]"
            )
        lines.append(
            f"TOST +-20% {tost_row['metric']:<10} {tost_row['op']:<5} (Stage-A failed runs {tost_row['stage_a_failed_runs']} in {tost_row['cells_with_stage_a_failed']} cells, Stage-B failed {tost_row['stage_b_failed_runs']}): "
            + " | ".join(parts)
        )
    changed = [
        cell
        for cell in cells
        if cell["n_stage_a_failed"]
        and (
            cell["tost_eq_measured"] != cell["tost_eq_success"]
            or cell["in_gold_band_measured"] != cell["in_gold_band_success"]
        )
    ]
    lines.append(
        f"cells where the TOST or gold-band verdict changes between the sets: {len(changed)}"
        + "".join(
            f"; {cell['arm']} {cell['level']} {cell['op']} {cell['metric']}: tost {cell['tost_eq_measured']}->{cell['tost_eq_success']}, band {cell['in_gold_band_measured']}->{cell['in_gold_band_success']}"
            for cell in changed
        )
    )
    for cell in cells:
        if not cell["n_stage_a_failed"] or cell["metric"] == "tp_harness":
            continue
        lines.append(
            f"cell {cell['arm']:<9} {cell['level']} {cell['op']:<5} {cell['metric']:<9} n={cell['n_measured']} (dagger {cell['n_stage_a_failed']}): HL {_pct((cell['hl_ratio_measured'] - 1) * 100)}% [{_pct((cell['ci90_lo_measured'] - 1) * 100)}, {_pct((cell['ci90_hi_measured'] - 1) * 100)}] vs success-only n={cell['n_success']} HL "
            + (
                f"{_pct((cell['hl_ratio_success'] - 1) * 100)}% [{_pct((cell['ci90_lo_success'] - 1) * 100)}, {_pct((cell['ci90_hi_success'] - 1) * 100)}]"
                if cell["hl_ratio_success"] is not None
                else "-- (n < 2)"
            )
        )
    for strata_row in ssum:
        parts = []
        for subset in SUBSETS:
            if f"strata_{subset}" not in strata_row:
                continue
            parts.append(
                f"{subset}: slower {strata_row[f'timeout_slower_{subset}']}/{strata_row[f'strata_{subset}']} sign p={strata_row[f'sign_test_p_{subset}']:.2g} (Holm {strata_row[f'sign_test_p_holm_{subset}']:.2g}), Z={strata_row[f'stouffer_z_{subset}']:+.2f} p={strata_row[f'stouffer_p_{subset}']:.2g}"
            )
        lines.append(
            f"strata {strata_row['comparison']:<20} {strata_row['op']:<5} {strata_row['metric']:<10} (dagger {strata_row.get('stage_a_failed_runs_measured', 0)}): "
            + " | ".join(parts)
        )
    for mode_row in mrows:
        if mode_row["reference"] is None or mode_row["model"] == "gold":
            continue
        line = f"mode D {mode_row['arm']:<11} {mode_row['client_mode']:<14} n={mode_row['n_measured']:>2} (dagger {mode_row['n_stage_a_failed']}) write: Dtp_pinned {_pct(mode_row['write_tp_pinned_delta_pct_measured'])}%, Dlat2n {_pct(mode_row['write_lat2n_delta_pct_measured'])}%; read: Dtp_pinned {_pct(mode_row['read_tp_pinned_delta_pct_measured'])}%, Dlat2n {_pct(mode_row['read_lat2n_delta_pct_measured'])}%"
        if mode_row["n_stage_a_failed"]:
            line += f"  || success-only n={mode_row['n_success']}: write Dtp_pinned {_pct(mode_row['write_tp_pinned_delta_pct_success'])}%, Dlat2n {_pct(mode_row['write_lat2n_delta_pct_success'])}%; read Dtp_pinned {_pct(mode_row['read_tp_pinned_delta_pct_success'])}%, Dlat2n {_pct(mode_row['read_lat2n_delta_pct_success'])}%"
        lines.append(line)
    for line in lines:
        print(line)
    return lines


STALE_FIGS: tuple[str, ...] = (
    "stage_b_latency_delta_kestrel",
    "stage_b_latency_delta_tessera",
    "stage_b_client_mode",
)
VIEW_TEX = {"all": "all measured runs", "success": "Stage-A successes only"}


def _date(iso: str) -> str:
    date = datetime.date.fromisoformat(iso)
    return f"{date.day} {date.strftime('%b %Y')}"


def _ref_dates(rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    dates_by_ref: dict[str, set[str]] = {}
    for row in rows:
        if row["model"] == "gold" and row["eval_date"]:
            dates_by_ref.setdefault(row["ref"], set()).add(row["eval_date"])
    return {
        ref: ", ".join(_date(date) for date in sorted(dates)) for ref, dates in dates_by_ref.items()
    }


def _pairs_text(rows: Sequence[dict[str, Any]], dialect: str) -> str:
    dates = _ref_dates(rows)
    parts = []
    for ref_dialect, epoch in tr.REFERENCE_EPOCHS:
        if ref_dialect != dialect:
            continue
        ref = f"{ref_dialect} E{epoch}"
        arms = [arm for arm in tr.ARMS if arm["ref"] == ref]
        if not arms:
            continue
        by_model: dict[str, list[dict[str, Any]]] = {}
        for arm in arms:
            by_model.setdefault(arm["model"], []).append(arm)
        who = " and ".join(
            f"{tr.MODEL_LABEL[model]} "
            + "/".join(tr.PROMPT_LABEL[arm["prompting"]] for arm in model_arms)
            + f" E{model_arms[0]['epoch']}"
            for model, model_arms in by_model.items()
        )
        parts.append(f"{who} vs gold E{epoch} ({dates.get(ref, '?')})")
    return "; ".join(parts)


def _missing_levels_text(cells: Sequence[dict[str, Any]], dialect: str) -> str:
    missing = [
        level
        for level in tr.LEVELS
        if not any(cell["level"] == level and cell["dialect"] == dialect for cell in cells)
    ]
    if not missing:
        return ""
    return f" {', '.join(missing)} not shown: no cell of this level has at least 2 measured runs."


def figure_captions(
    rows: Sequence[dict[str, Any]],
    cells: Sequence[dict[str, Any]],
    figs: dict[tuple[str, str], str],
) -> list[tuple[str, str, str]]:
    captions: list[tuple[str, str, str]] = []
    zs_epoch = {arm["dialect"]: arm["epoch"] for arm in tr.ARMS if arm["ref"] is None}
    for dialect in tr.DIALECTS:
        for view in tr.VIEWS:
            n, n_failed, _ = latency_delta_counts(cells, dialect, view)
            other = "success" if view == "all" else "all"
            if view == "all":
                shown = f"all measured runs ({n} generated runs, {n_failed} of which executed but failed Stage A and are additionally drawn as $\\times$ = run value over the gold median)"
            else:
                shown = f"Stage-A successes only ({n} generated runs)"
            captions.append(
                (
                    figs[(dialect, view)],
                    f"fig:stage-b-latency-delta-{dialect.lower()}-{view}",
                    f"{dialect}, view: {shown} (draft). Deviation of each main-matrix arm $\\times$ level cell from its same-day gold reference: {_pairs_text(rows, dialect)}. flash zero-shot (E{zs_epoch[dialect]}) has no same-day reference and is not shown.{_missing_levels_text(cells, dialect)} Hodges-Lehmann ratio with exact distribution-free 90\\,\\% CI, cells with $\\geq 2$ runs, for two indicators shown side by side: the median latency of requests 2..$n$ and the throughput over the first 10 writes / 20 reads. Shaded: $\\pm$20\\,\\% TOST margin; vertical bars: 95\\,\\% bootstrap band of the gold median; thick CI = TOST-equivalent per cell (uncorrected; Holm-adjusted counts in \\cref{{tab:stage-b-tost}}); a CI reaching beyond the axis range ends in an arrowhead. Colour = model, marker = prompting (square zero-shot, circle multi-shot). Positive values mean slower for latency and faster for throughput. The other view: \\cref{{fig:stage-b-latency-delta-{dialect.lower()}-{other}}}.",
                )
            )
    for view in tr.VIEWS:
        n, n_failed = client_mode_counts(rows, view)
        other = "success" if view == "all" else "all"
        n_gold = sum(1 for row in rows if row["model"] == "gold")
        if view == "all":
            shown = f"all measured runs ({n} generated runs of the 12 main-matrix arms, {n_failed} of which executed but failed Stage A: hatched bar parts, $\\dagger$ counts, $\\times$ = gap of a single such run)"
        else:
            shown = f"Stage-A successes only ({n} generated runs of the 12 main-matrix arms)"
        captions.append(
            (
                figs[("client", view)],
                f"fig:stage-b-client-mode-{view}",
                f"HTTP client mode of the generated steps per arm and of the {n_gold} gold reference runs (draft), view: {shown}. Left: share of runs per mode. Right: median client gap between consecutive requests per mode (write: circles, read: triangles; marker area grows with the number of runs). Descriptive. The other view: \\cref{{fig:stage-b-client-mode-{other}}}.",
            )
        )
    return captions


def readme_facts(
    rows: Sequence[dict[str, Any]],
    cells: Sequence[dict[str, Any]],
    tsum: Sequence[dict[str, Any]],
    ssum: Sequence[dict[str, Any]],
    mrows: Sequence[dict[str, Any]],
) -> dict[str, str]:
    facts: dict[str, str] = {}
    main_rows = [row for row in rows if row["model"] != "gold"]
    measured = [row for row in main_rows if measured_any(row)]
    failed = [row for row in measured if not row["success"]]
    facts["n_main"] = str(len(main_rows))
    facts["n_gold"] = str(len(rows) - len(main_rows))
    facts["n_meas"] = str(len(measured))
    facts["n_succ"] = str(len(measured) - len(failed))
    facts["n_failed"] = str(len(failed))
    facts["n_unmeasured"] = str(len(main_rows) - len(measured))
    facts["n_sbf"] = str(sum(1 for row in main_rows if row["stage_b_failed"]))
    facts["failed_models"] = (
        ", ".join(sorted({tr.MODEL_LABEL[row["model"]] for row in failed})) or "none"
    )
    facts["failed_classes"] = ", ".join(
        f"{failure_class} {n}"
        for failure_class, n in Counter(row["failure_class"] for row in failed).most_common()
    )
    facts["failed_cells"] = ", ".join(
        f"{arm} {level} {n}"
        for (arm, level), n in sorted(Counter((row["arm"], row["level"]) for row in failed).items())
    )
    facts["failed_modes"] = ", ".join(
        f"{mode} {n}" for mode, n in Counter(row["client_mode"] for row in failed).most_common()
    )
    facts["failed_libs"] = ", ".join(
        f"{lib} {n}" for lib, n in Counter(row["http_lib"] for row in failed).most_common()
    )
    facts["arms_with_ref"] = str(sum(1 for arm in tr.ARMS if arm["ref"] is not None))
    for subset in SUBSETS:
        facts[f"cells_{subset}"] = "/".join(
            sorted(
                {
                    str(tost_row[f"cells_{subset}"])
                    for tost_row in tsum
                    if tost_row["metric"] != "tp_harness"
                }
            )
        )
    lines = []
    for tost_row in tsum:
        lines.append(
            f"  - {METRIC_SHORT[tost_row['metric']].replace('$', '')} {tost_row['op']}: TOST-equivalent "
            + ", ".join(
                f"{tost_row[f'tost_eq_exact_{subset}']}/{tost_row[f'cells_{subset}']} ({'all' if subset == 'measured' else 'success'}; Holm {tost_row[f'tost_eq_holm_{subset}']})"
                for subset in SUBSETS
            )
            + "; inside the gold band "
            + " / ".join(str(tost_row[f"in_gold_band_{subset}"]) for subset in SUBSETS)
            + "; median |Delta| "
            + " / ".join(
                f"{_fmt_fixed(tost_row[f'median_abs_delta_pct_{subset}'])} %" for subset in SUBSETS
            )
        )
    facts["tost_lines"] = "\n".join(lines)
    for subset in SUBSETS:
        survivors = [
            f"{cell['arm']} {cell['level']} {METRIC_SHORT[cell['metric']].replace('$', '')} {cell['op']} (n = {cell[f'n_{subset}']}"
            + (
                f", {cell['n_stage_a_failed']} of them failed Stage A"
                if subset == "measured" and cell["n_stage_a_failed"]
                else ""
            )
            + ")"
            for cell in cells
            if cell[f"tost_eq_holm_{subset}"]
        ]
        facts[f"holm_{subset}"] = "; ".join(survivors) if survivors else "none"
    changed = [
        cell
        for cell in cells
        if cell["tost_eq_measured"] != cell["tost_eq_success"]
        or cell["in_gold_band_measured"] != cell["in_gold_band_success"]
    ]
    facts["n_diff"] = str(len(changed))
    facts["diff_arms"] = (
        ", ".join(sorted({f"{cell['arm']} {cell['level']}" for cell in changed})) or "none"
    )
    facts["n_failed_cells"] = str(
        len({(cell["arm"], cell["level"]) for cell in cells if cell["n_stage_a_failed"]})
    )
    tost_flips = [
        f"{cell['arm']} {cell['level']} {cell['op']} {METRIC_SHORT[cell['metric']].replace('$', '')}: "
        + (
            "equivalent only in the all-runs view"
            if cell["tost_eq_measured"]
            else "equivalent only in the success view"
        )
        for cell in changed
        if cell["tost_eq_measured"] != cell["tost_eq_success"]
    ]
    facts["tost_flips"] = "; ".join(tost_flips) or "none"
    for subset in SUBSETS:
        equivalent_tp_shares = [
            cell[f"share_fresh_timeout_{subset}"]
            for cell in cells
            if cell["metric"] == "tp_pinned" and cell[f"tost_eq_{subset}"]
        ]
        facts[f"eq_tp_share_{subset}"] = (
            f"{len(equivalent_tp_shares)} cells, median fresh+timeout share {100 * float(np.median(equivalent_tp_shares)):.0f} %, min {100 * min(equivalent_tp_shares):.0f} %"
            if equivalent_tp_shares
            else "0 cells"
        )
    for subset in SUBSETS:
        strata_rows = [
            strata_row
            for strata_row in ssum
            if strata_row["comparison"] == "vs_all_other" and f"strata_{subset}" in strata_row
        ]
        facts[f"strata_{subset}"] = ", ".join(
            f"{strata_row['op']} {strata_row['metric']} {strata_row[f'timeout_slower_{subset}']}/{strata_row[f'strata_{subset}']}"
            for strata_row in strata_rows
        )
    glm_kestrel = []
    for mode_row in mrows:
        if (
            mode_row["client_mode"] == "all"
            and mode_row["model"] == "glm"
            and mode_row["arm"].endswith("-K")
        ):
            glm_kestrel.append(
                f"{mode_row['arm']} write {_pct(mode_row['write_lat2n_delta_pct_measured'])} % (all) / {_pct(mode_row['write_lat2n_delta_pct_success'])} % (success)"
            )
    facts["glm_k_lat"] = "; ".join(glm_kestrel)
    gold_bimodal = []
    for ref_dialect, epoch in tr.REFERENCE_EPOCHS:
        gaps = [
            row["read_gap_med"]
            for row in rows
            if row["model"] == "gold" and row["ref"] == f"{ref_dialect} E{epoch}"
        ]
        gaps = [float(gap) for gap in gaps if gap is not None]
        fast = [gap for gap in gaps if gap < 5]
        slow = [gap for gap in gaps if gap >= 10]
        if fast and slow:
            gold_bimodal.append(
                f"{ref_dialect} E{epoch}: {len(fast)} runs at {min(fast):.1f}-{max(fast):.1f} ms, {len(slow)} at {min(slow):.1f}-{max(slow):.1f} ms"
            )
        else:
            gold_bimodal.append(
                f"{ref_dialect} E{epoch}: {len(gaps)} runs, {min(gaps):.1f}-{max(gaps):.1f} ms"
            )
    facts["gold_bimodal"] = "; ".join(gold_bimodal)
    reads_over = Counter(
        (row["model"], int(row["read_n"]))
        for row in measured
        if row.get("read_n") is not None and int(row["read_n"]) > PIN["read"]
    )
    facts["reads_over"] = (
        ", ".join(
            f"{tr.MODEL_LABEL[model]} {n_reads} reads x{count}"
            for (model, n_reads), count in sorted(reads_over.items())
        )
        or "none"
    )
    return facts


def write_readme(
    base: Path,
    figs: dict[tuple[str, str], str],
    key: Sequence[str],
    f: dict[str, str],
) -> None:
    fig_lines = "\n".join(
        f"- `fig/{figs[(dialect, view)]}.pdf|png`: {dialect}, view {VIEW_TEX[view]}."
        for dialect in tr.DIALECTS
        for view in tr.VIEWS
    )
    client_lines = "\n".join(
        f"- `fig/{figs[('client', view)]}.pdf|png`: client mode and client gap, view {VIEW_TEX[view]}."
        for view in tr.VIEWS
    )
    text = f"""# Stage B: client mode, per-request latency, TOST vs the same-day reference (RQ2b)

Generated by `uv run python scripts/thesis_stage_b_client.py` (reads only results/raw + results/artifacts;
imports `scripts/thesis_results.py` for arms, views and style). All captions are drafts.

## Which runs (three compared models)
- **12 main-matrix arms** (`thesis_results.ARMS`), N = 10 per level: flash = gemini-3.5-flash zero-shot E8 (Kestrel) /
  E9 (Tessera), multi-shot E13 (Kestrel) / E12 (Tessera); haiku = claude-haiku-5-5 zero- and multi-shot E14 (Kestrel) /
  E15 (Tessera); glm = glm-5.3 zero- and multi-shot E16 (Kestrel) / E17 (Tessera). {f["n_main"]} generated runs +
  {f["n_gold"]} gold reference runs (four series of 10: Tessera E12 and Kestrel E13 on 13 Sep 2026, Kestrel E16 and
  Tessera E17 on 9 Oct 2026).
- Only these runs are read; no other model or pipeline-test run enters any output of this script.
- **Delta / TOST need a same-day reference**: {f["arms_with_ref"]} of the 12 arms have one (E12/E13 -> gold E12/E13;
  E14/E16 -> gold Kestrel E16; E15/E17 -> gold Tessera E17). flash zero-shot E8/E9 has none, so it appears only in the
  client-mode figure, the per-mode table CSV and the strata test, never in a Delta. Epochs are never pooled; levels
  are pooled only inside one arm (= one epoch) in the per-mode table.

## Two views of equal rank
- **View "all"** = every run with a proxy value for the operation (harness rule `RunRecord.stage_b_value`: executed
  and >= 1 non-error request): {f["n_meas"]} of {f["n_main"]} generated runs have a write or read value. {f["n_failed"]} of
  them executed but failed Stage A ({f["failed_models"]} only; {f["failed_classes"]}: the workload wrote more points
  than specified). They are kept and marked: `x` in the model colour in the Delta figures, hatched bar parts, `x` gaps
  and `†` counts in the client-mode figure, `stage_a_failed` in the run table, a `†` column in the LaTeX tables.
- **View "success"** = Stage-A successes only: {f["n_succ"]} runs. Each Stage-B figure exists as two separate files
  (`*_all`, `*_success`), each with a view label and its run count printed above the panels. CSVs carry both views
  side by side (`*_measured` / `*_success`), the LaTeX tables give an extra "succ. only" row.
- The other {f["n_unmeasured"]} generated runs have no proxy value for write or read (did not execute / no request):
  they are Stage-A failures and belong to the Stage-A analysis, not here.
- **Stage-B failure** = a Stage-A success without a value for an operation: {f["n_sbf"]} runs (reported per cell as
  `n_stage_b_failed`, never imputed). `n_metric_na` counts measured runs whose derived indicator is not computable
  (e.g. fewer than 10 writes for the pinned throughput).
- Stage-A-failed measured runs per cell: {f["failed_cells"]}. Their client modes: {f["failed_modes"]}; libraries:
  {f["failed_libs"]}.

## No primary metric
Per-request latency (requests 2..n), pinned throughput (first 10 writes / 20 reads), harness throughput (all
requests), the client gap and the TOST / gold-band verdicts are separate indicators read together; none is ranked
above the others.

## Definitions
- **client_mode** from `steps.py` (comments stripped): `session` = `requests.Session(`/`Session()`/`httpx.Client(`/an
  `HTTPConnection` stored on `context`; otherwise `fresh+timeout` if a `timeout=` keyword appears, else `fresh`.
  **http_lib**: httpx > requests (incl. `__import__('requests')`) > http.client > urllib (incl. `from urllib import request`).
  The gold reference is `fresh+timeout` (urllib, `timeout=30`).
- From `evaluation/summary.json` -> `proxy_stats.requests` (sorted by `started_at`, per operation):
  `gap_med` = median of start(i+1) - end(i) in ms; `lat_first` = latency of request 1 (cold start);
  `lat2n` = median latency of requests 2..n; `tp_pinned` = 10 / span of the first 10 writes, 20 / span of the first
  20 reads (None if fewer requests; same request count as the gold workload); `tp_all` = n / span (checked against the
  harness value `tp_harness`, which counts every request); `lat_share` = sum latency / span; delete: `delete_lat_first`,
  `delete_lat_med` (covariate only). Runs with more reads than the gold workload's 20: {f["reads_over"]}.
- **HL / TOST** (tested, per cell): Hodges-Lehmann median of all pairwise log(x_i) - log(y_j) vs the 10 same-day gold
  runs, exact distribution-free (Moses) 90 % CI from the exact Mann-Whitney null distribution; TOST-equivalent if the CI
  lies inside [0.8, 1.2] (per cell, uncorrected). `tost_p` = max of the two shifted one-sided MWU tests, Holm-adjusted
  within indicator x op x view (cells with >= 2 runs: {f["cells_measured"]} in view all, {f["cells_success"]} in view
  success). `*_normal` columns repeat the CI with the normal approximation of the rank index.
- Gold band (descriptive) = 95 % percentile bootstrap (B = {BOOT_B}, seed {tr.JITTER_SEED}) of the gold median, as a ratio.
- **Strata test** (exploratory): inside arm x level of the 12 main arms (never pooled across arms), fresh+timeout vs
  (a) all other modes, (b) `fresh` only, for both views. Sign test over strata (two-sided), Holm over the 8 op x
  indicator rows of one comparison x view; Stouffer of signed two-sided MWU z, equal weights (uncorrected).
- The per-mode table (`stage_b_delta_by_client_mode`) and the client-gap-by-style table are descriptive (no test).

## Numbers in both views
{f["tost_lines"]}
- Holm-adjusted TOST equivalence survives in: view all: {f["holm_measured"]}; view success: {f["holm_success"]}.
  10 vs 10 runs can hardly reach p < 0.05 after Holm over this many cells; a survivor that depends on Stage-A-failed
  runs must not be read as a strong result.
- {f["n_diff"]} cell x indicator verdicts (TOST or gold band) differ between the views, all inside the
  {f["n_failed_cells"]} arm x level cells that contain Stage-A failures; changes occur in {f["diff_arms"]}. TOST flips:
  {f["tost_flips"]}.
- Pinned-throughput TOST-equivalent cells (write and read together): view all {f["eq_tp_share_measured"]}; view success {f["eq_tp_share_success"]}.
- Strata, fresh+timeout slower than all other modes (slower strata / strata): view all {f["strata_measured"]};
  view success {f["strata_success"]}.
- Per-request write latency Delta of glm Kestrel, levels pooled inside the arm: {f["glm_k_lat"]}.

## Caveats seen in the data
- The gold client gap is itself run-bimodal (read gap per run): {f["gold_bimodal"]}. The timeout penalty is
  intermittent (consistent with the 15.6 ms Windows timer resolution), so a gold median gap hides two modes and the
  gold bands are wide.
- In cells with few Stage-A successes left the success-view CI becomes very wide: dropping the failures there mostly
  removes information; read both views together.
- Delta sign: for latency a positive Delta/HL ratio means slower than the reference, for throughput faster.

## Outputs
{fig_lines}
  HL ratio + exact 90 % CI per arm x level vs the same-day gold series, for latency 2..n and pinned throughput
  (write/read); thick CI = TOST-equivalent, shaded +-20 % TOST margin, vertical bars = gold bootstrap band; colour =
  model, marker = prompting; `x` (view all only) = single run failing Stage A (value / gold median).
{client_lines}
  Per arm and gold series, share of runs per client mode (stacked) and median write/read client gap per mode;
  view all: hatched = Stage-A failures, `†` count at right, `x` = gap of a single failed run.
- `tables/stage_b_runs_detail.csv`: one row per main-matrix and gold run (client mode, library, `stage_a_failed`,
  `failure_class`, `<op>_measured`, `stage_b_failed`, n, gaps, latencies, pinned/harness throughput per op, `eval_date`).
- `tables/delta_latency_vs_reference.csv`: arm x level x op x indicator cells (lat2n, tp_pinned, tp_harness) with
  n_measured / n_success / n_stage_a_failed / n_stage_b_failed / n_metric_na, then medians, Delta, HL ratio, CI
  (exact + normal), TOST p (+Holm), gold band and fresh+timeout share for both views (`*_measured`, `*_success`).
- `tables/tost_summary.csv`: TOST-equivalent / in-band cell counts per indicator x op for both views, with the number
  of Stage-A-failed and Stage-B-failed runs.
- `tables/stage_b_delta_by_client_mode.csv`: per arm x mode (and `all`) n_measured / n_success / n_stage_a_failed,
  median pinned tp, lat2n, gap and Delta vs gold for both views (Delta empty for E8/E9).
- `tables/timeout_strata_tests.csv` (wide, both views), `tables/timeout_strata_detail.csv` (long, `subset` column).
- `tables/client_gap_by_style.csv`: gap/tp/latency by library x session x timeout (main-matrix runs, both views) + gold.
- `latex/stage_b_delta_by_client_mode.tex` (`tab:stage-b-client-mode`), `latex/stage_b_tost.tex` (`tab:stage-b-tost`),
  `latex/stage_b_timeout_strata.tex` (`tab:stage-b-timeout-strata`), `latex/figures.tex` (six figure floats, labels
  `fig:stage-b-latency-delta-<dialect>-<view>`, `fig:stage-b-client-mode-<view>`).

## Key numbers (printed by the script)
```
{chr(10).join(key)}
```
"""
    (base / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
