from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import thesis_results as tr
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from scipy import stats as sps

from benchmark_agent.evaluation.runs import SOURCE_PROXY, RunRecord

N_PIN = {"write": 10, "read": 20}
SLOW_DELETE_MS = 25.0
DAYS: tuple[str, ...] = ("2026-08-28", "2026-09-13", "2026-10-09")
DAY_LABEL = {"2026-08-28": "28 Aug 2026", "2026-09-13": "13 Sep 2026", "2026-10-09": "9 Oct 2026"}
EPOCHS: tuple[int, ...] = tuple(range(6, 18))
GOLD_PAIRS: tuple[tuple[str, int, int], ...] = (("Kestrel", 13, 16), ("Tessera", 12, 17))
CROSS_DAY_SOURCES: tuple[int, ...] = (7, 9)
CROSS_DAY_TARGET = 12
RNG_SEED = 20260715
N_SIM = 20000
PIPE_COLOR = "#bdbbb2"
PIPE_LABEL = "flash-lite (pipeline development)"
ROLE_PIPE = "pipeline development (flash-lite)"
ROLE_MAIN_MODELS = "main models (flash, haiku, glm)"
ROLE_GOLD = "gold reference"
NOISE_ROLE = "pipeline development (flash-lite): measurement noise only, not model evidence"


@dataclass
class RunPoint:
    run: RunRecord
    t: datetime
    day: str
    gold: bool
    epoch: int
    model: str
    prompting: str
    level: str
    rep: int
    dialect: str
    ok: bool = True
    write_req: float | None = None
    read_req: float | None = None
    delete_lat: float | None = None
    n_delete: int = 0
    tp: dict[str, float | None] = field(default_factory=dict)
    p50: dict[str, float | None] = field(default_factory=dict)
    art_hash: str | None = None

    @property
    def pipeline(self) -> bool:
        return self.model == tr.PIPELINE_MODEL

    @property
    def role(self) -> str:
        if self.gold:
            return ROLE_GOLD
        return ROLE_PIPE if self.pipeline else ROLE_MAIN_MODELS


MAIN_ARM_KEYS = {(arm["epoch"], arm["model"], arm["prompting"]) for arm in tr.ARMS}


def run_role(p: RunPoint) -> str:
    if p.gold:
        return ROLE_GOLD
    if p.pipeline:
        return ROLE_PIPE
    if (p.epoch, p.model, p.prompting) in MAIN_ARM_KEYS:
        return "main matrix (arm run)"
    return "main model, outside the 12 arms"


def _req_median(requests: Sequence[dict[str, Any]], op: str) -> float | None:
    op_requests = sorted(
        (request for request in requests if request.get("operation") == op),
        key=lambda request: request["started_at"],
    )
    pinned = op_requests[: N_PIN[op]]
    without_cold_start = pinned[1:]
    latencies = [
        float(request["latency_ms"])
        for request in without_cold_start
        if request.get("latency_ms") is not None
    ]
    return float(np.median(latencies)) if latencies else None


def _artefact_hash(run_dir: Path) -> str | None:
    try:
        blob = (
            (run_dir / "benchmark.feature").read_bytes()
            + b"\0"
            + (run_dir / "steps.py").read_bytes()
        )
    except OSError:
        return None
    return hashlib.sha256(blob).hexdigest()


def collect_points(runs: Sequence[RunRecord], project_root: Path) -> list[RunPoint]:
    points: list[RunPoint] = []
    for run in runs:
        if run.epoch not in EPOCHS or not run.evaluated_at or run.run_directory is None:
            continue
        gold = bool(run.is_fixture and run.condition_name.startswith("_gold_"))
        if run.is_fixture and not gold:
            continue
        throughput = {op: run.stage_b_value(SOURCE_PROXY, op, "throughput_ops_s") for op in tr.OPS}
        if all(value is None for value in throughput.values()):
            continue
        run_dir = project_root / run.run_directory
        summary_path = run_dir / "evaluation" / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        requests = (summary.get("proxy_stats") or {}).get("requests") or []
        deletes = sorted(
            (request for request in requests if request.get("operation") == "delete"),
            key=lambda request: request["started_at"],
        )
        evaluated = datetime.fromisoformat(run.evaluated_at.replace("Z", "+00:00"))
        if gold:
            dialect = "Kestrel" if "kestrel" in run.condition_name else "Tessera"
        else:
            dialect = tr.DIALECT_OF_EPOCH[run.epoch]
        points.append(
            RunPoint(
                run=run,
                t=evaluated,
                day=evaluated.date().isoformat(),
                gold=gold,
                epoch=int(run.epoch),
                model="gold" if gold else str(run.model),
                prompting=str(run.prompting),
                level="gold" if gold else str(run.guidance_level),
                rep=int(run.repetition or 0),
                dialect=dialect,
                ok=bool(gold or run.success),
                write_req=(
                    _req_median(requests, "write") if throughput["write"] is not None else None
                ),
                read_req=_req_median(requests, "read") if throughput["read"] is not None else None,
                delete_lat=(
                    float(deletes[0]["latency_ms"])
                    if deletes and throughput["delete"] is not None
                    else None
                ),
                n_delete=len(deletes) if throughput["delete"] is not None else 0,
                tp=throughput,
                p50={op: run.stage_b_value(SOURCE_PROXY, op, "latency_p50_ms") for op in tr.OPS},
                art_hash=None if gold else _artefact_hash(run_dir),
            )
        )
    return points


VIEWS = tr.VIEWS
VIEW_LABEL = {"all": "all measured runs", "success": "Stage-A successes only"}


def in_view(points: Sequence[RunPoint], view: str) -> list[RunPoint]:
    if view == "all":
        return list(points)
    if view == "success":
        return [p for p in points if p.ok]
    if view == "stage_a_failed":
        return [p for p in points if not p.ok]
    raise ValueError(view)


def _matches_role(run: RunRecord, role: str | None) -> bool:
    if role is None:
        return True
    wants_pipeline = role == ROLE_PIPE
    is_pipeline_run: bool = run.model == tr.PIPELINE_MODEL
    return wants_pipeline == is_pipeline_run


def stage_b_failures(runs: Sequence[RunRecord], epoch: int, role: str | None = None) -> int:
    return sum(
        1
        for run in runs
        if run.epoch == epoch
        and not run.is_fixture
        and _matches_role(run, role)
        and run.success
        and any(run.stage_b_value(SOURCE_PROXY, op, "throughput_ops_s") is None for op in tr.OPS)
    )


def is_slow(p: RunPoint) -> bool | None:
    if p.n_delete != 1 or p.delete_lat is None:
        return None
    return p.delete_lat > SLOW_DELETE_MS


def slow_share(points: Sequence[RunPoint]) -> tuple[int, int]:
    slow_flags = [is_slow(p) for p in points]
    known_flags = [flag for flag in slow_flags if flag is not None]
    return sum(known_flags), len(known_flags)


def executed_slow_share(runs: Sequence[RunRecord], day: str) -> tuple[int, int]:
    n_slow = n = 0
    for run in runs:
        if run.epoch not in EPOCHS or not (run.evaluated_at or "").startswith(day):
            continue
        if run.is_fixture and not run.condition_name.startswith("_gold_"):
            continue
        delete_stats = run.op(SOURCE_PROXY, "delete")
        throughput = run.stage_b_value(SOURCE_PROXY, "delete", "throughput_ops_s")
        if throughput is None or delete_stats is None or delete_stats.n_requests != 1:
            continue
        n += 1
        n_slow += throughput < 1000.0 / SLOW_DELETE_MS
    return n_slow, n


def hl_ratio(
    a: Sequence[float], b: Sequence[float], conf: float = 0.95
) -> tuple[float, float, float]:
    log_a, log_b = np.log(np.asarray(a, float)), np.log(np.asarray(b, float))
    log_diffs = np.sort(np.subtract.outer(log_b, log_a).ravel())
    m, n = len(a), len(b)
    z = sps.norm.isf((1 - conf) / 2)
    k = max(int(math.floor(m * n / 2 - z * math.sqrt(m * n * (m + n + 1) / 12))), 0)
    return (
        float(np.exp(np.median(log_diffs))),
        float(np.exp(log_diffs[k])),
        float(np.exp(log_diffs[len(log_diffs) - 1 - k])),
    )


def residuals(
    points: Sequence[RunPoint], value: Callable[[RunPoint], float | None]
) -> list[tuple[RunPoint, float]]:
    cells: dict[tuple[str, str, str], list[tuple[RunPoint, float]]] = defaultdict(list)
    for p in points:
        measured = value(p)
        if measured is not None and measured > 0:
            cells[(p.model, p.prompting, p.level)].append((p, math.log(measured)))
    result: list[tuple[RunPoint, float]] = []
    for members in cells.values():
        if len(members) < 3:
            continue
        cell_median = float(np.median([log_value for _, log_value in members]))
        result += [(p, log_value - cell_median) for p, log_value in members]
    return result


def spearman_time(res: Sequence[tuple[RunPoint, float]]) -> tuple[float | None, float | None]:
    if len(res) < 10:
        return None, None
    result = sps.spearmanr([p.t.timestamp() for p, _ in res], [residual for _, residual in res])
    return float(result.statistic), float(result.pvalue)


def kw_repetition(res: Sequence[tuple[RunPoint, float]]) -> float | None:
    groups: dict[int, list[float]] = defaultdict(list)
    for p, residual in res:
        groups[p.rep].append(residual)
    samples = [group for group in groups.values() if len(group) >= 2]
    if len(samples) < 3:
        return None
    return float(sps.kruskal(*samples).pvalue)


def fmt_ratio_iqr(r: np.ndarray) -> tuple[str, str, str]:
    q1, med, q3 = tr.quart(r.tolist())
    return f"{med:.2f}", f"{q1:.2f}--{q3:.2f}", f"{r.min():.2f}--{r.max():.2f}"


def drift_rows(points: Sequence[RunPoint], runs: Sequence[RunRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in (ROLE_MAIN_MODELS, ROLE_PIPE):
        for view in VIEWS:
            rows += _drift_rows_view(points, runs, view, role)
    return rows


def _drift_rows_view(
    points: Sequence[RunPoint], runs: Sequence[RunRecord], view: str, role: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variants: list[tuple[str, int, Callable[[RunPoint], bool]]] = [
        (f"E{epoch}", epoch, lambda p: True) for epoch in EPOCHS
    ]
    if role == ROLE_MAIN_MODELS:
        variants.insert(7, ("E12 without L3", 12, lambda p: p.level != "L3"))
    for label, epoch, keep in variants:
        shown = [
            p for p in points if p.epoch == epoch and not p.gold and p.role == role and keep(p)
        ]
        members = in_view(shown, view)
        if not members:
            continue
        times = sorted(p.t for p in members)
        lat_residuals = residuals(members, lambda p: p.write_req)
        tp_residuals = residuals(members, lambda p: p.tp["write"])
        rho_lat, p_lat = spearman_time(lat_residuals)
        rho_tp, p_tp = spearman_time(tp_residuals)
        n_slow, n_delete = slow_share(members)
        models = sorted({tr.MODEL_LABEL[p.model] for p in members})
        rows.append(
            {
                "role": role,
                "view": view,
                "epoch": label,
                "day": members[0].day,
                "models": "/".join(models),
                "dialect": tr.DIALECT_OF_EPOCH[epoch],
                "n_runs": len(members),
                "n_outside_arms": sum(
                    run_role(p) == "main model, outside the 12 arms" for p in members
                ),
                "n_stage_a_failed": sum(not p.ok for p in members),
                "n_stage_a_failed_measured": sum(not p.ok for p in shown),
                "n_stage_b_failed": (
                    stage_b_failures(runs, epoch, role) if label == f"E{epoch}" else None
                ),
                "start_utc": times[0].strftime("%H:%M"),
                "end_utc": times[-1].strftime("%H:%M"),
                "n_resid": len(lat_residuals),
                "rho_write_req_lat": rho_lat,
                "p_write_req_lat": p_lat,
                "rho_write_tp": rho_tp,
                "p_write_tp": p_tp,
                "kw_rep_p_write_req_lat": kw_repetition(lat_residuals),
                "n_slow_delete": n_slow,
                "n_delete": n_delete,
                "slow_delete_share": n_slow / n_delete if n_delete else None,
                "sensitivity": label != f"E{epoch}",
            }
        )
    main_rows = [row for row in rows if not row["sensitivity"]]
    for key in ("p_write_req_lat", "p_write_tp"):
        adjusted = tr.holm([row[key] for row in main_rows])
        for row, adjusted_p in zip(main_rows, adjusted, strict=True):
            row[key + "_holm"] = adjusted_p
    return rows


def gold_points(points: Sequence[RunPoint], dialect: str, epoch: int) -> list[RunPoint]:
    return [p for p in points if p.gold and p.epoch == epoch and p.dialect == dialect]


GOLD_METRICS: tuple[tuple[str, str, str, Callable[[RunPoint], float | None]], ...] = (
    ("write", "req_lat", "per-request latency (ms)", lambda p: p.write_req),
    ("read", "req_lat", "per-request latency (ms)", lambda p: p.read_req),
    ("delete", "req_lat", "latency (ms)", lambda p: p.delete_lat),
    ("write", "tp", "throughput (ops/s)", lambda p: p.tp["write"]),
    ("read", "tp", "throughput (ops/s)", lambda p: p.tp["read"]),
)


def gold_drift_rows(points: Sequence[RunPoint]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dialect, first_epoch, second_epoch in GOLD_PAIRS:
        first_points = gold_points(points, dialect, first_epoch)
        second_points = gold_points(points, dialect, second_epoch)
        for op, metric, label, value_of in GOLD_METRICS:
            first = _present_values(first_points, value_of)
            second = _present_values(second_points, value_of)
            estimate, lo, hi = hl_ratio(first, second)
            _, p_value = tr.mwu(first, second)
            rows.append(
                {
                    "dialect": dialect,
                    "first": f"E{first_epoch}",
                    "second": f"E{second_epoch}",
                    "op": op,
                    "metric": metric,
                    "metric_label": label,
                    "n_first": len(first),
                    "n_second": len(second),
                    "median_first": float(np.median(first)),
                    "median_second": float(np.median(second)),
                    "median_ratio": float(np.median(second) / np.median(first)),
                    "hl_ratio": estimate,
                    "hl_lo95": lo,
                    "hl_hi95": hi,
                    "mwu_p": p_value,
                    "slow_first": slow_share(first_points)[0] if op == "delete" else None,
                    "slow_second": slow_share(second_points)[0] if op == "delete" else None,
                }
            )
    latency_family = [row for row in rows if row["metric"] == "req_lat"]
    adjusted = tr.holm([row["mwu_p"] for row in latency_family])
    for row, adjusted_p in zip(latency_family, adjusted, strict=True):
        row["mwu_p_holm"] = adjusted_p
    return rows


def _present_values(
    points: Sequence[RunPoint], value_of: Callable[[RunPoint], float | None]
) -> list[float]:
    values = [value_of(p) for p in points]
    return [value for value in values if value is not None]


PAIR_METRICS: tuple[tuple[str, str, Callable[[RunPoint], float | None]], ...] = (
    ("write", "tp", lambda p: p.tp["write"]),
    ("write", "req_lat", lambda p: p.write_req),
    ("read", "tp", lambda p: p.tp["read"]),
    ("read", "req_lat", lambda p: p.read_req),
    ("read", "p50", lambda p: p.p50["read"]),
    ("delete", "tp", lambda p: p.tp["delete"]),
)


def same_day_pairs(
    runs: Sequence[RunRecord], points: Sequence[RunPoint], project_root: Path
) -> tuple[list[dict[str, Any]], list[tuple[RunPoint, RunPoint]]]:
    by_id = {p.run.run_id: p for p in points if not p.gold}
    retest_rows = tr.retest_pairs(runs, project_root)
    pairs = [
        (by_id[entry["run_first"]], by_id[entry["run_second"]])
        for entry in retest_rows
        if entry["identical_artifact"]
        and entry["run_first"] in by_id
        and entry["run_second"] in by_id
    ]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in retest_rows:
        prompting = "ms" if "multishot" in entry["condition"] else "zs"
        groups[(entry["pair"], entry["level"], prompting)].append(entry)
    rows: list[dict[str, Any]] = []
    for (pair, level, prompting), members in sorted(groups.items()):
        identical = [entry for entry in members if entry["identical_artifact"]]
        measured = [
            (by_id[entry["run_first"]], by_id[entry["run_second"]])
            for entry in identical
            if entry["run_first"] in by_id and entry["run_second"] in by_id
        ]
        rows.append(_pair_summary(pair, level, prompting, len(members), len(identical), measured))
    for pair in ("E6->E10", "E7->E11", "all"):
        selected = [
            (first, second)
            for first, second in pairs
            if pair == "all" or f"E{first.epoch}->E{second.epoch}" == pair
        ]
        n_all = sum(1 for entry in retest_rows if pair == "all" or entry["pair"] == pair)
        n_identical = sum(
            1
            for entry in retest_rows
            if entry["identical_artifact"] and (pair == "all" or entry["pair"] == pair)
        )
        rows.append(_pair_summary(pair, "all", "all", n_all, n_identical, selected))
    return rows, pairs


def _ratios(
    pairs: Sequence[tuple[RunPoint, RunPoint]], value_of: Callable[[RunPoint], float | None]
) -> np.ndarray:
    values = [(value_of(first), value_of(second)) for first, second in pairs]
    return np.array([second / first for first, second in values if first and second])


def _pair_summary(
    pair: str,
    level: str,
    prompting: str,
    n_pairs: int,
    n_ident: int,
    measured: Sequence[tuple[RunPoint, RunPoint]],
) -> dict[str, Any]:
    ok_pairs = [(first, second) for first, second in measured if first.ok and second.ok]
    row: dict[str, Any] = {
        "role": NOISE_ROLE,
        "pair": pair,
        "level": level,
        "prompting": prompting,
        "n_pairs": n_pairs,
        "n_identical": n_ident,
        "n_identical_measured": len(measured),
        "n_identical_success": len(ok_pairs),
        "n_identical_stage_a_failed": len(measured) - len(ok_pairs),
    }
    for suffix, selected in (("", measured), ("_success", ok_pairs)):
        for name, value_of in (
            ("tp", lambda p: p.tp["write"]),
            ("req_lat", lambda p: p.write_req),
        ):
            ratios = _ratios(selected, value_of)
            if len(ratios):
                q1, med, q3 = tr.quart(ratios.tolist())
                sd_log = float(np.log(ratios).std(ddof=1)) if len(ratios) > 1 else None
                row |= {
                    f"write_{name}_ratio_median{suffix}": med,
                    f"write_{name}_ratio_q1{suffix}": q1,
                    f"write_{name}_ratio_q3{suffix}": q3,
                    f"write_{name}_ratio_min{suffix}": float(ratios.min()),
                    f"write_{name}_ratio_max{suffix}": float(ratios.max()),
                    f"write_{name}_sd_log_ratio{suffix}": sd_log,
                }
    return row


def noise_band_rows(pairs: Sequence[tuple[RunPoint, RunPoint]]) -> list[dict[str, Any]]:
    rows = []
    for view in VIEWS:
        selected = [
            (first, second) for first, second in pairs if view == "all" or (first.ok and second.ok)
        ]
        rows += _noise_band_view(selected, view)
    return rows


def _noise_band_view(pairs: Sequence[tuple[RunPoint, RunPoint]], view: str) -> list[dict[str, Any]]:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for op, metric, value_of in PAIR_METRICS:
        if metric == "p50":
            continue
        values = [(value_of(first), value_of(second)) for first, second in pairs]
        present = [(first, second) for first, second in values if first and second]
        log_ratios = np.log(np.array([second / first for first, second in present]))
        sd_pair = float(log_ratios.std(ddof=1))
        sd_single = sd_pair / math.sqrt(2)
        first_medians = np.median(rng.normal(0, sd_single, (N_SIM, 10)), 1)
        second_medians = np.median(rng.normal(0, sd_single, (N_SIM, 10)), 1)
        sims = first_medians - second_medians
        lo, hi = np.exp(np.percentile(sims, [2.5, 97.5]))
        ratio = np.exp(log_ratios)
        rows.append(
            {
                "role": NOISE_ROLE,
                "view": view,
                "op": op,
                "metric": metric,
                "n_pairs": len(log_ratios),
                "n_pairs_stage_a_failed": sum(
                    1
                    for first, second in pairs
                    if not (first.ok and second.ok) and value_of(first) and value_of(second)
                ),
                "median_ratio": float(np.median(ratio)),
                "median_abs_log": float(np.median(np.abs(log_ratios))),
                "share_0.8_1.25": float(np.mean((ratio >= 0.8) & (ratio <= 1.25))),
                "ratio_min": float(ratio.min()),
                "ratio_max": float(ratio.max()),
                "sd_log_ratio": sd_pair,
                "sd_log_single_run": sd_single,
                "spearman_first_second": float(
                    sps.spearmanr(
                        [first for first, _ in present], [second for _, second in present]
                    ).statistic
                ),
                "band10_lo_pct": float((lo - 1) * 100),
                "band10_hi_pct": float((hi - 1) * 100),
            }
        )
    return rows


def cross_day_rows(
    points: Sequence[RunPoint],
) -> tuple[list[dict[str, Any]], list[tuple[RunPoint, RunPoint]]]:
    sources = [p for p in points if not p.gold and p.epoch in CROSS_DAY_SOURCES and p.art_hash]
    targets = sorted(
        (p for p in points if not p.gold and p.epoch == CROSS_DAY_TARGET and p.art_hash),
        key=lambda p: p.t,
    )
    pairs: list[tuple[RunPoint, RunPoint]] = []
    for target in targets:
        candidates = [
            source
            for source in sources
            if source.art_hash == target.art_hash
            and source.run.seed == target.run.seed
            and source.level == target.level
        ]
        if candidates:
            earliest = min(candidates, key=lambda source: source.t)
            pairs.append((earliest, target))
    assert all(first.model == second.model for first, second in pairs), (
        "a cross-day pair mixes models"
    )
    rows: list[dict[str, Any]] = []
    for group, role, keep in CROSS_DAY_GROUPS:
        for view in VIEWS:
            selected = [
                (first, second)
                for first, second in pairs
                if keep(first) and (view == "all" or (first.ok and second.ok))
            ]
            rows += _cross_day_view(selected, view, group, role)
    return rows, pairs


CROSS_DAY_GROUPS: tuple[tuple[str, str, Callable[[RunPoint], bool]], ...] = (
    ("flash E9->E12", ROLE_MAIN_MODELS, lambda p: not p.pipeline),
    ("flash-lite E7->E12", ROLE_PIPE, lambda p: p.pipeline),
    (
        "pooled",
        "measurement noise only (identical artefacts, includes the flash-lite pipeline pairs)",
        lambda p: True,
    ),
)
CROSS_DAY_FAMILY = {("write", "tp"), ("read", "tp"), ("read", "p50"), ("delete", "tp")}


def _cross_day_view(
    pairs: Sequence[tuple[RunPoint, RunPoint]], view: str, group: str, role: str
) -> list[dict[str, Any]]:
    view_rows = []
    for op, metric, value_of in PAIR_METRICS:
        ratios = _ratios(pairs, value_of)
        log_ratios = np.log(ratios)
        p_value = float(sps.wilcoxon(log_ratios).pvalue) if len(log_ratios) >= 3 else None
        q1, med, q3 = tr.quart(ratios.tolist())
        view_rows.append(
            {
                "group": group,
                "role": role,
                "view": view,
                "op": op,
                "metric": metric,
                "n_pairs": len(ratios),
                "n_pairs_stage_a_failed": sum(
                    1
                    for first, second in pairs
                    if not (first.ok and second.ok) and value_of(first) and value_of(second)
                ),
                "median_ratio": med,
                "ratio_q1": q1,
                "ratio_q3": q3,
                "median_abs_log": float(np.median(np.abs(log_ratios))),
                "wilcoxon_p": p_value,
                "holm_family": (op, metric) in CROSS_DAY_FAMILY,
            }
        )
    family = [row for row in view_rows if row["holm_family"]]
    adjusted = tr.holm([row["wilcoxon_p"] for row in family])
    for row, adjusted_p in zip(family, adjusted, strict=True):
        row["wilcoxon_p_holm"] = adjusted_p
    return view_rows


def window_rows(points: Sequence[RunPoint]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[RunPoint]] = defaultdict(list)
    for p in points:
        groups[(p.day, f"{p.t.hour:02d}:{0 if p.t.minute < 30 else 30:02d}")].append(p)
    rows = []
    for (day, window), members in sorted(groups.items()):
        row: dict[str, Any] = {
            "day": day,
            "window_utc": window,
            "n_runs": len(members),
            "n_stage_a_failed": sum(not p.ok for p in members),
            "n_gold": sum(p.gold for p in members),
            "n_pipeline_flash_lite": sum(p.pipeline for p in members),
            "epochs": "/".join(f"E{epoch}" for epoch in sorted({p.epoch for p in members})),
        }
        for suffix, selected in (("", members), ("_success", in_view(members, "success"))):
            write_lats = [p.write_req for p in selected if p.write_req]
            n_slow, n_delete = slow_share(selected)
            row |= {
                f"median_write_req_lat_ms{suffix}": (
                    float(np.median(write_lats)) if write_lats else None
                ),
                f"slow_delete_share{suffix}": n_slow / n_delete if n_delete else None,
                f"n_delete{suffix}": n_delete,
            }
        rows.append(row)
    return rows


BLOCKS: dict[str, list[tuple[str, tuple[int, ...], bool]]] = {
    "2026-08-28": [
        ("E6", (6,), False),
        ("E7", (7,), False),
        ("E8", (8,), False),
        ("E9", (9,), False),
        ("E10", (10,), False),
        ("E11", (11,), False),
    ],
    "2026-09-13": [
        ("E12", (12,), False),
        ("E13", (13,), False),
    ],
    "2026-10-09": [
        ("E14 haiku-K", (14,), False),
        ("E16 glm-K", (16,), False),
        ("E15 haiku-T", (15,), False),
        ("E17 glm-T", (17,), False),
    ],
}


def _block_span(points: Sequence[RunPoint], epochs: tuple[int, ...]) -> tuple[datetime, datetime]:
    times = sorted(p.t for p in points if p.epoch in epochs and not p.gold)
    if len(times) > 8:
        gaps = np.diff([t.timestamp() for t in times])
        cut = int(np.argmax(gaps)) if gaps.max() > 3600 else len(times) - 1
        times = times[: cut + 1]
    return times[0], times[-1]


def _rolling_segments(
    xs: Sequence[datetime], ys: Sequence[float], k: int = 15, gap_s: float = 1200.0
) -> list[tuple[list[datetime], list[float]]]:
    order = np.argsort([x.timestamp() for x in xs])
    x_sorted = [xs[i] for i in order]
    y_sorted = np.asarray(ys, float)[order]
    half = k // 2
    med = [
        float(np.median(y_sorted[max(0, i - half) : i + half + 1])) for i in range(len(y_sorted))
    ]
    segments: list[tuple[list[datetime], list[float]]] = []
    start = 0
    for i in range(1, len(x_sorted) + 1):
        if i == len(x_sorted) or (x_sorted[i] - x_sorted[i - 1]).total_seconds() > gap_s:
            if i - start >= 5:
                segments.append((x_sorted[start:i], med[start:i]))
            start = i
    return segments


PIPELINE_EPOCHS = (6, 7, 10, 11)
TIMELINE_ROWS: tuple[tuple[str, Callable[[RunPoint], float | None], str], ...] = (
    ("write", lambda p: p.write_req, "write latency per request,\nmedian of req. 2–10 (ms)"),
    ("delete", lambda p: p.delete_lat, "delete latency (ms)"),
)


def timeline_counts(points: Sequence[RunPoint]) -> dict[str, int]:
    shown = [p for p in points if any(value_of(p) for _, value_of, _ in TIMELINE_ROWS)]
    return {
        "n": len(shown),
        "main": sum(p.role == ROLE_MAIN_MODELS for p in shown),
        "pipeline": sum(p.pipeline for p in shown),
        "gold": sum(p.gold for p in shown),
        "failed": sum(not p.ok for p in shown),
        "failed_main": sum(not p.ok and not p.pipeline for p in shown),
        "failed_pipeline": sum(not p.ok and p.pipeline for p in shown),
    }


def timeline_header(points: Sequence[RunPoint], view: str) -> str:
    counts = timeline_counts(points)
    base = tr.view_label(counts["n"], counts["failed"], view)
    return (
        f"{base}\n{counts['main']} main-model, {counts['pipeline']} flash-lite (pipeline development) "
        f"and {counts['gold']} gold runs"
    )


def fig_timeline(all_points: Sequence[RunPoint], out: Path, view: str) -> str:
    spans = []
    for day in DAYS:
        timestamps = [p.t.timestamp() for p in all_points if p.day == day]
        spans.append((max(timestamps) - min(timestamps)) / 3600)
    points = in_view(all_points, view)
    ratios = [max(span, 3.2) for span in spans]
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(tr.TEXT_WIDTH_IN, 3.9),
        sharey="row",
        gridspec_kw={"width_ratios": ratios, "wspace": 0.13, "hspace": 0.16},
    )
    for col, day in enumerate(DAYS):
        day_pts = [p for p in points if p.day == day]
        for row, (_, value_of, ylabel) in enumerate(TIMELINE_ROWS):
            ax = axes[row, col]
            generated = [p for p in day_pts if not p.gold and value_of(p)]
            lite = [p for p in generated if p.pipeline]
            for selected, marker, size in (
                ([p for p in lite if p.ok], "o", 4),
                ([p for p in lite if not p.ok], "x", 10),
            ):
                if selected:
                    ax.scatter(
                        [p.t for p in selected],
                        [value_of(p) for p in selected],
                        s=size,
                        marker=marker,
                        color=PIPE_COLOR,
                        linewidth=0 if marker == "o" else 0.7,
                        zorder=2,
                        rasterized=True,
                    )
            for model in tr.MODELS:
                selected = [p for p in generated if p.model == model and p.ok]
                if selected:
                    ax.scatter(
                        [p.t for p in selected],
                        [value_of(p) for p in selected],
                        s=5,
                        color=tr.MODEL_COLOR[model],
                        alpha=0.8,
                        linewidth=0,
                        zorder=3,
                        rasterized=True,
                    )
                failed = [p for p in generated if p.model == model and not p.ok]
                if failed:
                    ax.scatter(
                        [p.t for p in failed],
                        [value_of(p) for p in failed],
                        s=14,
                        marker="x",
                        color=tr.MODEL_COLOR[model],
                        linewidth=0.8,
                        zorder=4,
                        rasterized=True,
                    )
            gold = [p for p in day_pts if p.gold and value_of(p)]
            if gold:
                ax.scatter(
                    [p.t for p in gold],
                    [value_of(p) for p in gold],
                    s=13,
                    marker="D",
                    color=tr.INK,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=5,
                )
            for xs, ys in _rolling_segments(
                [p.t for p in generated], [float(value_of(p) or 0) for p in generated]
            ):
                ax.plot(xs, ys, color=tr.INK_2, lw=0.9, zorder=4)
            if row == 1:
                ax.axhline(SLOW_DELETE_MS, color=tr.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=2)
            if day == "2026-09-13":
                l3_times = [
                    p.t for p in day_pts if p.epoch == 12 and p.level == "L3" and not p.gold
                ]
                ax.axvspan(min(l3_times), max(l3_times), color=tr.GRID, alpha=0.8, lw=0, zorder=0)
                if row == 0:
                    ax.text(
                        max(l3_times),
                        0.97,
                        " E12 L3",
                        transform=blended_transform_factory(ax.transData, ax.transAxes),
                        fontsize=6.5,
                        color=tr.INK_2,
                        ha="left",
                        va="top",
                    )
            ax.set_yscale("log")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=None))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2 if spans[col] > 4 else 1))
            ax.spines["bottom"].set_color(tr.BASELINE)
            if row == 0:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel(f"UTC, {DAY_LABEL[day]}")
            if col == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(axis="y", left=False)
        top_ax = axes[0, col]
        trans = blended_transform_factory(top_ax.transData, top_ax.transAxes)
        for i, (label, epochs, _) in enumerate(BLOCKS[day]):
            t0, t1 = _block_span(day_pts, epochs)
            level = 1.04 if i % 2 == 0 else 1.15
            pipeline_only = all(epoch in PIPELINE_EPOCHS for epoch in epochs)
            top_ax.plot(
                [t0, t1],
                [level, level],
                color=PIPE_COLOR if pipeline_only else tr.INK_2,
                lw=1.4,
                transform=trans,
                clip_on=False,
                solid_capstyle="butt",
            )
            top_ax.text(
                t0 + (t1 - t0) / 2,
                level + 0.02,
                label,
                transform=trans,
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=tr.MUTED if pipeline_only else tr.INK_2,
                clip_on=False,
            )
        for gold_epoch in sorted({p.epoch for p in day_pts if p.gold}):
            gold_times = [p.t for p in day_pts if p.gold and p.epoch == gold_epoch]
            mid = min(gold_times) + (max(gold_times) - min(gold_times)) / 2
            for ax in axes[:, col]:
                ax.axvline(mid, color=tr.INK, lw=0.5, alpha=0.35, zorder=1)
    for ax in axes.flat:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda tick, _: f"{tick:g}"))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
    axes[0, 0].set_yticks([5, 10, 20, 50])
    axes[1, 0].set_yticks([5, 10, 20, 50, 100])
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=3.5,
            color=tr.MODEL_COLOR[model],
            label=tr.MODEL_FULL[model],
        )
        for model in tr.MODELS
    ]
    handles.append(Line2D([], [], marker="o", ls="", ms=3, color=PIPE_COLOR, label=PIPE_LABEL))
    if any(not p.ok for p in points):
        handles.append(
            Line2D(
                [],
                [],
                marker="x",
                ls="",
                ms=4,
                mew=0.9,
                color=tr.INK_2,
                label="failed Stage A (measured, kept)",
            )
        )
    handles += [
        Line2D([], [], marker="D", ls="", ms=4, color=tr.INK, label="reference benchmark (block)"),
        Line2D([], [], color=tr.INK_2, lw=0.9, label="rolling median, 15 runs shown"),
        Line2D([], [], color=tr.MUTED, lw=0.7, ls=(0, (3, 2)), label="slow-delete threshold 25 ms"),
        Patch(facecolor=tr.GRID, label="E12 L3 block (after gold)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.2,
        handletextpad=0.4,
    )
    tr._header(fig, timeline_header(points, view))
    return tr.save(fig, out, f"measurement_timeline_{view}")


MAIN_ARM_ROLE = "main matrix (arm run)"
THESIS_TIMELINE: dict[str, tuple[Callable[[RunPoint], float | None], str, str, list[float]]] = {
    "write": (
        lambda p: p.write_req,
        "write latency per request,\nmedian of requests 2 to 10 (ms)",
        "measurement_timeline",
        [3, 5, 10, 20, 50],
    ),
    "delete": (
        lambda p: p.delete_lat,
        "delete latency (ms)",
        "measurement_timeline_delete",
        [3, 5, 10, 20, 50, 100],
    ),
}
PHASE_WINDOW_MIN = 30
PHASE_MIN_RUNS = 5
PHASE_COLOR = {"slow": "#e6e4dc", "fast": "#f4f3ef"}
X_PAD_MIN = 8
SEGMENT_GAP_S = 3600
SEGMENT_MIN_H = 0.6


def thesis_points(points: Sequence[RunPoint]) -> list[RunPoint]:
    return [p for p in points if p.gold or (run_role(p) == MAIN_ARM_ROLE and p.ok)]


def _window_start(t: datetime) -> datetime:
    return t.replace(
        minute=(t.minute // PHASE_WINDOW_MIN) * PHASE_WINDOW_MIN, second=0, microsecond=0
    )


def two_means_boundary(values: Sequence[float]) -> float:
    ordered = sorted(values)
    best: tuple[float, float] | None = None
    for i in range(1, len(ordered)):
        lo, hi = np.asarray(ordered[:i]), np.asarray(ordered[i:])
        sse = float(((lo - lo.mean()) ** 2).sum() + ((hi - hi.mean()) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, (ordered[i - 1] + ordered[i]) / 2)
    assert best is not None
    return best[1]


def phase_windows(points: Sequence[RunPoint]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    groups: dict[tuple[str, datetime], list[RunPoint]] = defaultdict(list)
    for p in points:
        groups[(p.day, _window_start(p.t))].append(p)
    rows: list[dict[str, Any]] = []
    for (day, start), members in sorted(groups.items()):
        write_lats = [p.write_req for p in members if p.write_req]
        n_slow, n_delete = slow_share(members)
        rows.append(
            {
                "day": day,
                "window_start_utc": start,
                "n_runs": len(members),
                "n_gold": sum(p.gold for p in members),
                "epochs": "/".join(f"E{epoch}" for epoch in sorted({p.epoch for p in members})),
                "median_write_req_lat_ms": float(np.median(write_lats)) if write_lats else None,
                "n_slow_delete": n_slow,
                "n_delete": n_delete,
                "slow_delete_share": n_slow / n_delete if n_delete else None,
            }
        )
    usable = [
        row
        for row in rows
        if row["n_runs"] >= PHASE_MIN_RUNS
        and row["median_write_req_lat_ms"] is not None
        and row["slow_delete_share"] is not None
    ]
    log_cut = two_means_boundary([math.log(row["median_write_req_lat_ms"]) for row in usable])
    share_cut = two_means_boundary([row["slow_delete_share"] for row in usable])
    cuts = {"write_ms": math.exp(log_cut), "slow_delete_share": share_cut}
    for row in rows:
        if row in usable:
            by_lat = "slow" if row["median_write_req_lat_ms"] > cuts["write_ms"] else "fast"
            by_del = "slow" if row["slow_delete_share"] > share_cut else "fast"
            row["class_write_latency"], row["class_slow_delete"] = by_lat, by_del
            row["phase"] = by_lat if by_lat == by_del else ""
        else:
            row["class_write_latency"] = row["class_slow_delete"] = ""
            row["phase"] = ""
    return rows, cuts


def phase_spans(rows: Sequence[dict[str, Any]]) -> list[tuple[str, str, datetime, datetime]]:
    spans: list[tuple[str, str, datetime, datetime]] = []
    step = timedelta(minutes=PHASE_WINDOW_MIN)
    for row in rows:
        if not row["phase"]:
            continue
        start = row["window_start_utc"]
        end = start + step
        if (
            spans
            and spans[-1][0] == row["day"]
            and spans[-1][1] == row["phase"]
            and spans[-1][3] == start
        ):
            spans[-1] = (row["day"], row["phase"], spans[-1][2], end)
        else:
            spans.append((row["day"], row["phase"], start, end))
    return spans


def late_reevaluations(points: Sequence[RunPoint]) -> list[RunPoint]:
    late: list[RunPoint] = []
    for day in DAYS:
        for epoch in sorted({p.epoch for p in points if p.day == day and not p.gold}):
            members = [p for p in points if p.day == day and p.epoch == epoch and not p.gold]
            _, block_end = _block_span(members, (epoch,))
            late += [p for p in members if (p.t - block_end).total_seconds() > 3600]
    return late


def day_segments(points: Sequence[RunPoint], day: str) -> list[tuple[datetime, datetime]]:
    times = sorted(p.t for p in points if p.day == day)
    groups: list[list[datetime]] = [[times[0]]]
    for t in times[1:]:
        if (t - groups[-1][-1]).total_seconds() > SEGMENT_GAP_S:
            groups.append([t])
        else:
            groups[-1].append(t)
    pad = timedelta(minutes=X_PAD_MIN)
    segments = []
    for group in groups:
        start, end = group[0] - pad, group[-1] + pad
        missing = timedelta(hours=SEGMENT_MIN_H) - (end - start)
        if missing.total_seconds() > 0:
            start, end = start - missing / 2, end + missing / 2
        segments.append((start, end))
    return segments


def _x_break(ax: Axes, side: str) -> None:
    ax.plot(
        [1.0 if side == "right" else 0.0],
        [0.0],
        transform=ax.transAxes,
        marker=[(-0.5, -1), (0.5, 1)],
        markersize=6,
        linestyle="none",
        color=tr.INK_2,
        mec=tr.INK_2,
        mew=0.9,
        clip_on=False,
        zorder=10,
    )


def fig_thesis_timeline(
    points: Sequence[RunPoint],
    spans: Sequence[tuple[str, str, datetime, datetime]],
    metric: str,
    out: Path,
) -> str:
    value_of, ylabel, name, yticks = THESIS_TIMELINE[metric]
    shown = [p for p in points if value_of(p)]
    segments = {day: day_segments(points, day) for day in DAYS}
    segment_hours = {
        day: [(end - start).total_seconds() / 3600 for start, end in segments[day]] for day in DAYS
    }
    fig = plt.figure(figsize=(tr.TEXT_WIDTH_IN, 2.55))
    outer = fig.add_gridspec(
        1,
        len(DAYS),
        width_ratios=[
            sum(segment_hours[day]) * (1.12 if len(segment_hours[day]) > 1 else 1) for day in DAYS
        ],
        wspace=0.1,
    )
    axes: list[Axes] = []
    for i, day in enumerate(DAYS):
        inner = outer[i].subgridspec(
            1, len(segments[day]), width_ratios=segment_hours[day], wspace=0.4
        )
        day_axes = []
        for j, (x0, x1) in enumerate(segments[day]):
            ax = fig.add_subplot(inner[j], sharey=axes[0] if axes else None)
            axes.append(ax)
            day_axes.append(ax)
            day_pts = [p for p in shown if p.day == day]
            seg_pts = [p for p in day_pts if x0 <= p.t <= x1]
            day_spans = [span for span in spans if span[0] == day]
            for _, phase, phase_start, phase_end in day_spans:
                start, end = max(phase_start, x0), min(phase_end, x1)
                if end <= start:
                    continue
                ax.axvspan(start, end, color=PHASE_COLOR[phase], lw=0, zorder=0)
                if (end - start).total_seconds() >= 20 * 60:
                    ax.text(
                        start + (end - start) / 2,
                        0.965,
                        phase,
                        transform=blended_transform_factory(ax.transData, ax.transAxes),
                        ha="center",
                        va="top",
                        fontsize=6.0,
                        color=tr.MUTED,
                    )
            for model in tr.MODELS:
                selected = [p for p in seg_pts if p.model == model]
                if selected:
                    ax.scatter(
                        [p.t for p in selected],
                        [value_of(p) for p in selected],
                        s=6,
                        color=tr.MODEL_COLOR[model],
                        alpha=0.85,
                        linewidth=0,
                        zorder=3,
                        rasterized=True,
                    )
            gold = [p for p in seg_pts if p.gold]
            if gold:
                ax.scatter(
                    [p.t for p in gold],
                    [value_of(p) for p in gold],
                    s=12,
                    marker="D",
                    color=tr.INK,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=5,
                )
            for xs, ys in _rolling_segments(
                [p.t for p in day_pts], [float(value_of(p) or 0) for p in day_pts]
            ):
                ax.plot(xs, ys, color=tr.INK_2, lw=0.9, zorder=4)
            if metric == "delete":
                ax.axhline(SLOW_DELETE_MS, color=tr.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=2)
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            generated = [p for p in points if p.day == day and not p.gold and x0 <= p.t <= x1]
            for epoch in sorted({p.epoch for p in generated}):
                t0, t1 = _block_span(generated, (epoch,))
                ax.text(
                    t0 + (t1 - t0) / 2,
                    1.015,
                    f"E{epoch}",
                    transform=trans,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color=tr.INK_2,
                    clip_on=False,
                )
            ax.set_xlim(x0, x1)
            ax.set_yscale("log")
            width = segment_hours[day][j]
            if width > 4:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            elif width > 1.5:
                ax.xaxis.set_major_locator(mdates.HourLocator())
            else:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.tick_params(axis="x", labelsize=6.5)
            ax.spines["bottom"].set_color(tr.BASELINE)
            if ax is not axes[0]:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            if len(segments[day]) > 1:
                if j > 0:
                    _x_break(ax, "left")
                if j < len(segments[day]) - 1:
                    _x_break(ax, "right")
        main_ax = day_axes[int(np.argmax(segment_hours[day]))]
        main_ax.set_xlabel(f"{DAY_LABEL[day]} (UTC)", fontsize=7)
    axes[0].set_ylabel(ylabel)
    axes[0].set_yticks(yticks)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda tick, _: f"{tick:g}"))
    axes[0].yaxis.set_minor_formatter(plt.NullFormatter())
    values = [float(value_of(p) or 0) for p in shown]
    axes[0].set_ylim(min(values) / 1.25, max(values) * 1.45)
    handles: list[Any] = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            ms=3.5,
            color=tr.MODEL_COLOR[model],
            label=tr.MODEL_FULL[model],
        )
        for model in tr.MODELS
    ]
    handles += [
        Line2D([], [], marker="D", ls="", ms=3.5, color=tr.INK, label="reference benchmark"),
        Line2D([], [], color=tr.INK_2, lw=0.9, label="rolling median, 15 runs"),
        Patch(facecolor=PHASE_COLOR["slow"], label="slow phase"),
        Patch(facecolor=PHASE_COLOR["fast"], edgecolor=tr.GRID, lw=0.5, label="fast phase"),
    ]
    if metric == "delete":
        handles.append(
            Line2D([], [], color=tr.MUTED, lw=0.7, ls=(0, (3, 2)), label="slow delete, 25 ms")
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.2,
        handletextpad=0.4,
    )
    n_gold = sum(p.gold for p in shown)
    tr._header(
        fig,
        f"Stage-A successes of the 12 main arms ({len(shown) - n_gold} runs) "
        f"and {n_gold} reference runs",
    )
    return tr.save(fig, out, name)


def _span_text(spans: Sequence[tuple[str, str, datetime, datetime]]) -> str:
    parts = []
    for day in DAYS:
        items = [
            f"{phase} {start:%H:%M} to {end:%H:%M}"
            for span_day, phase, start, end in spans
            if span_day == day
        ]
        if items:
            parts.append(f"{DAY_LABEL[day]}: " + ", ".join(items))
    return "; ".join(parts)


def tex_thesis_figures(
    points: Sequence[RunPoint],
    spans: Sequence[tuple[str, str, datetime, datetime]],
    cuts: dict[str, float],
) -> str:
    late = late_reevaluations(points)
    words = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
    late_txt = ""
    late_groups: dict[tuple[str, int], list[RunPoint]] = defaultdict(list)
    for p in late:
        late_groups[(p.day, p.epoch)].append(p)
    for (day, epoch), members in sorted(late_groups.items()):
        n = len(members)
        when = sorted({f"{p.t:%H:%M}" for p in members})
        late_txt += (
            f" The {words[n] if n < len(words) else n} isolated E{epoch} points on {DAY_LABEL[day]}, behind the axis break, "
            f"are later re-evaluations ({', '.join(when)} UTC) of runs of that epoch."
        )
    days_text = (
        "28 Aug 2026: flash zero-shot E8/E9; 13 Sep 2026: flash multi-shot E12/E13 and two gold series; "
        "9 Oct 2026: haiku E14/E15, glm E16/E17 and two gold series"
    )
    phase_rule = (
        f" Background: slow and fast phases from 30-minute windows with at least {PHASE_MIN_RUNS} of these runs. "
        "Two indicators per window, the median write latency per request and the share of runs with a slow delete "
        f"(delete latency above {SLOW_DELETE_MS:.0f}\\,ms), are each split into two groups by an exact two-means split over the windows "
        f"(boundaries {cuts['write_ms']:.1f}\\,ms and {100 * cuts['slow_delete_share']:.0f}\\,\\%); a window is shaded only when both "
        f"indicators agree, and adjacent windows of one class form a phase ({_span_text(spans)} UTC)."
    )
    floats = []
    for metric, label in (
        ("write", "fig:measurement-timeline"),
        ("delete", "fig:measurement-timeline-delete"),
    ):
        value_of, _, name, _ = THESIS_TIMELINE[metric]
        shown = [p for p in points if value_of(p)]
        n_gold = sum(p.gold for p in shown)
        n_main = len(shown) - n_gold
        if metric == "write":
            caption = (
                "Per-run write latency per request (median of requests 2 to 10, log scale) against the UTC evaluation time, "
                f"one panel per measurement day ({days_text}). Shown are the {n_main} Stage-A successes of the 12 main arms "
                f"(colour = model) and the {n_gold} gold reference runs (black diamonds); the line is the rolling median over "
                "15 consecutive runs of a day, and the labels at the top mark the epoch blocks."
                + phase_rule
                + late_txt
                + " The models ran in separate blocks and on different days, so the timeline describes the measurement "
                "environment and does not compare models."
            )
        else:
            caption = (
                "Per-run latency of the single delete request (log scale) against the UTC evaluation time, with the same "
                f"runs ({n_main} Stage-A successes of the 12 main arms and {n_gold} gold reference runs), slow and fast phases "
                "and layout as \\cref{fig:measurement-timeline}. Dashed line: slow-delete threshold "
                f"{SLOW_DELETE_MS:.0f}\\,ms, which defines the second phase indicator."
            )
        floats.append(
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
    return "\n\n".join(floats) + "\n"


def _rho_tex(x: float | None) -> str:
    return "--" if x is None else f"{x:+.2f}".replace("-", "$-$")


def _rho_plain(x: float | None) -> str:
    return "--" if x is None else f"{x:+.2f}"


def tex_drift(rows: Sequence[dict[str, Any]]) -> str:
    rows = [row for row in rows if row["role"] == ROLE_MAIN_MODELS]
    body: list[list[str]] = []
    breaks: list[int] = []
    success_rows = {row["epoch"]: row for row in rows if row["view"] == "success"}
    shown: list[tuple[dict[str, Any], bool]] = []
    for row in rows:
        if row["view"] != "all":
            continue
        shown.append((row, False))
        if row["n_stage_a_failed"]:
            shown.append((success_rows[row["epoch"]], True))
    for k, (row, success_only) in enumerate(shown):
        if k and shown[k - 1][0]["day"] != row["day"]:
            breaks.append(len(body) - 1)
        body.append(
            [
                r"\quad A-ok only"
                if success_only
                else tr.tex(row["epoch"].replace(" without ", " w/o ")),
                "" if success_only else tr.tex(row["models"]),
                "" if success_only else row["dialect"][0],
                "" if success_only else f"{row['start_utc']}--{row['end_utc']}",
                str(row["n_resid"]),
                "" if success_only else str(row["n_stage_a_failed"] or "--"),
                _rho_tex(row["rho_write_req_lat"]),
                tr.fmt_p(row["p_write_req_lat"]),
                tr.fmt_p(row.get("p_write_req_lat_holm")),
                _rho_tex(row["rho_write_tp"]),
                tr.fmt_p(row["p_write_tp"]),
                tr.fmt_p(row["kw_rep_p_write_req_lat"]),
                f"{row['n_slow_delete']}/{row['n_delete']}",
            ]
        )
    n_main = sum(1 for row in rows if row["view"] == "all" and not row["sensitivity"])
    n_outside = {
        row["epoch"]: row["n_outside_arms"]
        for row in rows
        if row["view"] == "all" and row["n_outside_arms"]
    }
    outside_txt = " and ".join(f"{count} in {epoch}" for epoch, count in n_outside.items())
    late_epochs = [
        row["epoch"]
        for row in rows
        if row["view"] == "all" and not row["sensitivity"] and row["epoch"] in ("E7", "E9")
    ]
    late_txt = (
        f"The UTC span of {' and '.join(late_epochs)} includes two late re-evaluations at 15:31. "
        if late_epochs
        else ""
    )
    return tr.latex_table(
        "Drift within each epoch of the main models (flash, haiku, glm; exploratory). "
        "Residual = log value minus the median of its model $\\times$ prompting $\\times$ level cell; "
        "$\\rho$ = Spearman correlation of the residual with the UTC evaluation time. KW = Kruskal--Wallis test of the latency residual over the "
        "repetition index 1--10. Slow deletes = runs with one delete request slower than 25 ms.",
        "tab:drift-by-epoch",
        [
            "Epoch",
            "Models",
            "D",
            "UTC",
            "n",
            "\\dag",
            "$\\rho_{\\mathrm{lat}}$",
            "p",
            "p\\textsubscript{Holm}",
            "$\\rho_{\\mathrm{tp}}$",
            "p",
            "KW p",
            "Slow del.",
        ],
        body,
        "lllrrrrrrrrrr",
        note="Write per-request latency = median of requests 2--10; tp = write throughput. All measured runs (executed, proxy value present), gold runs excluded; "
        "\\dag{} = runs among them that failed Stage A after executing (kept and counted). An epoch with such runs gets an ``A-ok only'' row below it with the same "
        "statistics over the Stage-A successes only. "
        + _stage_b_note(rows)
        + f"p-values are uncorrected; p\\textsubscript{{Holm}} adjusts the latency test over the {n_main} epochs, separately per view. "
        f"The rows include measured flash multi-shot L3 runs that belong to no main arm ({outside_txt}). "
        "``E12 w/o L3'' is a sensitivity row: the E12 L3 runs were evaluated as one block after the gold block (18:02--18:12 UTC). "
        + late_txt
        + "D = dialect (K = Kestrel, T = Tessera).",
        size=r"\scriptsize",
        midrules_after=breaks,
        tabcolsep="3pt",
    )


def tex_gold(rows: Sequence[dict[str, Any]]) -> str:
    slow_counts = []
    for row in rows:
        if row["op"] == "delete":
            slow_counts += [row["slow_first"], row["slow_second"]]
    body = []
    for i, row in enumerate(rows):
        first = i == 0 or rows[i - 1]["dialect"] != row["dialect"]
        body.append(
            [
                f"{row['dialect']} {row['first']}$\\rightarrow${row['second']}" if first else "",
                row["op"],
                "latency" if row["metric"] == "req_lat" else "tp",
                f"{row['median_first']:.1f}",
                f"{row['median_second']:.1f}",
                f"{row['hl_ratio']:.2f}",
                f"{row['hl_lo95']:.2f}--{row['hl_hi95']:.2f}",
                tr.fmt_p(row["mwu_p"]),
                tr.fmt_p(row.get("mwu_p_holm")),
            ]
        )
    return tr.latex_table(
        "Gold reference across measurement days (N = 10 per series): Kestrel 13 Sep (E13) vs.\\ 9 Oct (E16), Tessera 13 Sep (E12) vs.\\ 9 Oct (E17). "
        "HL = Hodges--Lehmann estimate of the ratio later/earlier with its 95\\,\\% confidence interval; p = Mann--Whitney U.",
        "tab:gold-drift",
        [
            "Series",
            "Op.",
            "Metric",
            "Earlier",
            "Later",
            "HL ratio",
            "95\\,\\% CI",
            "p",
            "p\\textsubscript{Holm}",
        ],
        body,
        "lllrrrrrr",
        note="Latency = per-request latency in ms (write/read: median of requests 2..n; delete: the single request); tp = throughput in ops/s. "
        "A ratio above 1 means slower for latency and faster for throughput. Holm over the six latency tests; throughput rows are uncorrected. "
        "Slow gold deletes ($>$25 ms): Kestrel "
        + "{}/10 vs.\\ {}/10, Tessera {}/10 vs.\\ {}/10.".format(*slow_counts),
        size=r"\scriptsize",
        midrules_after=(len(rows) // 2 - 1,),
        tabcolsep="4pt",
    )


def tex_retest(rows: Sequence[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        if "write_tp_ratio_median" in row:
            ratio_cells = [
                f"{row['write_tp_ratio_median']:.2f}",
                f"{row['write_tp_ratio_q1']:.2f}--{row['write_tp_ratio_q3']:.2f}",
                f"{row['write_tp_ratio_min']:.2f}--{row['write_tp_ratio_max']:.2f}",
                tr.fmt(row["write_tp_sd_log_ratio"], 2),
                f"{row['write_req_lat_ratio_median']:.2f}",
                tr.fmt(row["write_req_lat_sd_log_ratio"], 2),
            ]
        else:
            ratio_cells = ["--"] * 6
        body.append(
            [
                tr.tex(row["pair"].replace("->", "→")),
                row["level"],
                row["prompting"],
                str(row["n_pairs"]),
                str(row["n_identical"]),
                str(row["n_identical_measured"]),
                str(row["n_identical_stage_a_failed"] or "--"),
                *ratio_cells,
            ]
        )
    midrules = [
        i
        for i in range(len(rows) - 1)
        if rows[i]["level"] != "all" and rows[i + 1]["level"] == "all"
    ]
    midrules += [
        i
        for i in range(len(rows) - 1)
        if rows[i]["pair"] != rows[i + 1]["pair"]
        and rows[i]["level"] != "all"
        and rows[i + 1]["level"] != "all"
    ]
    return tr.latex_table(
        "Pipeline development (flash-lite), measurement noise only: same-day test--retest (28 Aug 2026), the same condition and seed generated "
        "in two flash-lite epochs. Identical = byte-identical "
        "\\texttt{benchmark.feature} and \\texttt{steps.py}; ratios second/first over the identical pairs where both runs were measured.",
        "tab:retest-noise",
        [
            "Pair",
            "Level",
            "Pr.",
            "Pairs",
            "Ident.",
            "Meas.",
            "\\dag",
            "tp ratio",
            "IQR",
            "Range",
            "sd(log)",
            "lat ratio",
            "sd(log)",
        ],
        body,
        "lllrrrrrrrrrr",
        note="Meas. = identical pairs with both runs measured (executed, proxy value present); \\dag{} = of these, pairs with a run that failed Stage A (kept). "
        + _views_note(
            sum(row["n_identical_stage_a_failed"] for row in rows if row["pair"] == "all")
        )
        + "tp = write throughput, lat = write per-request latency (median of requests 2--10); sd(log) = standard deviation of the log ratio "
        "(single-run sd = sd(log)/$\\sqrt{2}$). Identical pairs measure measurement noise only; they are not independent replications of the generation. "
        "zs = zero-shot, ms = multi-shot (only L3 has multi-shot runs in E6/E7/E10/E11). "
        + tr.PIPELINE_LEGEND,
        size=r"\scriptsize",
        midrules_after=sorted(set(midrules)),
        tabcolsep="3pt",
    )


def _stage_b_note(rows: Sequence[dict[str, Any]]) -> str:
    n = sum(row["n_stage_b_failed"] or 0 for row in rows if row["view"] == "all")
    if n == 0:
        return "No epoch has a Stage-B failure (a Stage-A success without a proxy value). "
    return f"{n} Stage-A successes lack a proxy value for some operation (Stage-B failures, column n\\_stage\\_b\\_failed in the CSV). "


METRIC_TEX = {"tp": "throughput", "req_lat": "latency/request", "p50": "latency p50"}


def _views_note(n_failed: int) -> str:
    if n_failed == 0:
        return (
            "No pair contains a run that failed Stage A, so the view over Stage-A successes only is identical "
            "(CSV: view = success). "
        )
    return f"{n_failed} pairs contain a run that failed Stage A (kept); the ``A-ok only'' rows exclude them. "


def _with_sensitivity(rows: Sequence[dict[str, Any]]) -> list[tuple[dict[str, Any], bool]]:
    measured = [row for row in rows if row["view"] == "all"]
    shown = [(row, False) for row in measured]
    if any(row["n_pairs_stage_a_failed"] for row in measured):
        shown += [(row, True) for row in rows if row["view"] == "success"]
    return shown


def tex_noise_band(rows: Sequence[dict[str, Any]]) -> str:
    shown = _with_sensitivity(rows)
    n_failed = max(row["n_pairs_stage_a_failed"] for row in rows if row["view"] == "all")
    body = [
        [
            row["op"] + (" (A-ok only)" if success_only else ""),
            METRIC_TEX[row["metric"]],
            str(row["n_pairs"]),
            f"{row['median_ratio']:.2f}",
            f"{row['median_abs_log']:.2f}",
            f"{row['share_0.8_1.25'] * 100:.0f}\\,\\%",
            f"{row['ratio_min']:.2f}--{row['ratio_max']:.2f}",
            f"{row['sd_log_single_run']:.2f}",
            _rho_tex(row["spearman_first_second"]),
            f"{row['band10_lo_pct']:+.0f}..{row['band10_hi_pct']:+.0f}\\,\\%".replace("-", "$-$"),
        ]
        for row, success_only in shown
    ]
    return tr.latex_table(
        "Pipeline development (flash-lite), measurement noise only: noise floor from the same-day identical measured pairs "
        "(E6$\\rightarrow$E10, E7$\\rightarrow$E11, 28 Aug 2026), single-run spread and the "
        "95\\,\\% band of the ratio of two medians of ten runs each (parametric bootstrap, lognormal with the single-run sd).",
        "tab:noise-band",
        [
            "Op.",
            "Metric",
            "Pairs",
            "Med. ratio",
            "Med. $|\\log|$",
            "0.8--1.25",
            "Range",
            "sd\\textsubscript{run}",
            "$\\rho_{1,2}$",
            "Band 10 vs.\\ 10",
        ],
        body,
        "llrrrrrrrr",
        note=_views_note(n_failed)
        + "sd\\textsubscript{run} = sd(log ratio)/$\\sqrt{2}$; $\\rho_{1,2}$ = Spearman correlation between first and second measurement. "
        "A lower bound for the noise: all pairs come from one day and one model (flash-lite, mostly fast HTTP clients). "
        "The pairs are identical code measured twice, so they describe the measurement, not a model. "
        + tr.PIPELINE_LEGEND
        + " "
        f"Band from {N_SIM} simulated pairs of medians (seed {RNG_SEED}).",
        size=r"\scriptsize",
        tabcolsep="3pt",
    )


CROSS_DAY_THESIS_GROUP = "flash E9->E12"


def tex_crossday(rows: Sequence[dict[str, Any]], pairs: Sequence[tuple[RunPoint, RunPoint]]) -> str:
    keep = next(
        group_keep for group, _, group_keep in CROSS_DAY_GROUPS if group == CROSS_DAY_THESIS_GROUP
    )
    group_rows = [row for row in rows if row["group"] == CROSS_DAY_THESIS_GROUP]
    n_pairs = sum(1 for first, _ in pairs if keep(first))
    n_failed = max(row["n_pairs_stage_a_failed"] for row in group_rows if row["view"] == "all")
    body = [
        [
            row["op"] + (" (A-ok only)" if success_only else ""),
            METRIC_TEX[row["metric"]],
            str(row["n_pairs"]),
            f"{row['median_ratio']:.2f}",
            f"{row['ratio_q1']:.2f}--{row['ratio_q3']:.2f}",
            f"{row['median_abs_log']:.2f}",
            tr.fmt_p(row["wilcoxon_p"]),
            tr.fmt_p(row.get("wilcoxon_p_holm")) if row["holm_family"] else "(not in family)",
        ]
        for row, success_only in _with_sensitivity(group_rows)
    ]
    return tr.latex_table(
        f"Identical artefacts measured 16 days apart: {n_pairs} measured gemini-3.5-flash Tessera L3 multi-shot artefacts first measured on 28 Aug 2026 "
        "(the L3 multi-shot runs inside E9) and byte-identical again on 13 Sep 2026 (E12). Ratio = 13 Sep / 28 Aug; "
        "p = Wilcoxon signed-rank on the log ratio. Measurement evidence, not model evidence.",
        "tab:retest-crossday",
        [
            "Op.",
            "Metric",
            "Pairs",
            "Med. ratio",
            "IQR",
            "Med. $|\\log|$",
            "p",
            "p\\textsubscript{Holm}",
        ],
        body,
        "llrrrrrr",
        note=_views_note(n_failed)
        + "Both runs of a pair are the same artefact with the same seed, so a pair shows how much the same code moves between two days. "
        "Holm over the four metrics write/read/delete throughput and read p50; the per-request latency rows are supplementary and uncorrected. "
        f"With {n_pairs} pairs the smallest attainable exact two-sided Wilcoxon p is {2 / 2**n_pairs:.4g}. The E12 L3 runs ran as a block after the gold block (fast late phase), so the shift mixes "
        "day and time of day.",
        size=r"\scriptsize",
        tabcolsep="4pt",
    )


def _csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        keys += [key for key in row if key not in keys]
    tr.write_csv(path, keys, [[row.get(key) for key in keys] for row in rows])


STALE_OUTPUTS: tuple[str, ...] = ("latex/retest_noise.tex", "latex/noise_band.tex")

SLOW_KEYS: tuple[tuple[str, str], ...] = (
    ("all", "every run measured that day (environment, incl. flash-lite)"),
    ("success", "same, Stage-A successes only"),
    ("main_all", "main arms + gold, all measured"),
    ("main_success", "main arms + gold, Stage-A successes only"),
    ("pipe_all", "flash-lite pipeline runs, all measured"),
    ("pipe_success", "flash-lite pipeline runs, Stage-A successes only"),
    ("executed_ok", "executed_ok cross-check (every run)"),
)


def main() -> None:
    tr._style()
    config, runs = tr.load_runs()
    root = config.paths.project_root
    out = root / "results" / "reports" / "thesis_results" / "extra" / "timeline"
    fig_dir, tab_dir, tex_dir = out / "fig", out / "tables", out / "latex"
    for directory in (fig_dir, tab_dir, tex_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for stale in STALE_OUTPUTS:
        (out / stale).unlink(missing_ok=True)

    points = collect_points(runs, root)
    n_fail = sum(not p.ok for p in points)
    failed_cells = Counter((p.epoch, p.model, p.level) for p in points if not p.ok)
    failed_text = ", ".join(
        f"E{epoch} {tr.MODEL_LABEL[model]} {level} {n}"
        for (epoch, model, level), n in sorted(failed_cells.items())
    )
    print(
        f"Measured points: {sum(p.role == ROLE_MAIN_MODELS for p in points)} main-model + "
        f"{sum(p.pipeline for p in points)} flash-lite (pipeline development) + {sum(p.gold for p in points)} gold; "
        f"{n_fail} failed Stage A (kept): " + failed_text
    )
    print(
        "  Stage-B failures (Stage-A success without a proxy value), E6-E17: "
        + str(sum(stage_b_failures(runs, epoch) for epoch in EPOCHS))
    )

    timeline_views = {}
    for view in VIEWS:
        fig_timeline(points, fig_dir, view)
        timeline_views[view] = timeline_counts(in_view(points, view))
        print(f"  measurement_timeline_{view}: {timeline_header(in_view(points, view), view)!r}")
    thesis_pts = thesis_points(points)
    phase_rows, cuts = phase_windows(thesis_pts)
    spans = phase_spans(phase_rows)
    _csv(
        tab_dir / "timeline_phases.csv",
        [
            {**row, "window_start_utc": row["window_start_utc"].strftime("%H:%M")}
            for row in phase_rows
        ],
    )
    for metric in THESIS_TIMELINE:
        fig_thesis_timeline(thesis_pts, spans, metric, fig_dir)
    (tex_dir / "figures.tex").write_text(
        tex_thesis_figures(thesis_pts, spans, cuts), encoding="utf-8"
    )
    print(
        f"  thesis timeline: {len(thesis_pts)} runs ({sum(p.gold for p in thesis_pts)} gold); phase boundaries: "
        f"write {cuts['write_ms']:.2f} ms, slow-delete share {cuts['slow_delete_share']:.3f}; phases: {_span_text(spans)}"
    )
    _csv(
        tab_dir / "timeline_runs.csv",
        [
            {
                "run_id": p.run.run_id,
                "role": run_role(p),
                "epoch": p.epoch,
                "gold": p.gold,
                "model": p.model,
                "prompting": p.prompting,
                "level": p.level,
                "repetition": p.rep,
                "dialect": p.dialect,
                "stage_a_success": p.ok,
                "in_view_all": True,
                "in_view_success": p.ok,
                "evaluated_at": p.t.isoformat(),
                "write_req_lat_ms": p.write_req,
                "read_req_lat_ms": p.read_req,
                "delete_lat_ms": p.delete_lat,
                "n_delete": p.n_delete,
                "slow_delete": is_slow(p),
            }
            for p in sorted(points, key=lambda p: p.t)
        ],
    )

    windows = window_rows(points)
    _csv(tab_dir / "timeline_windows.csv", windows)
    print(
        "\n9 Oct 2026, 30-min windows (UTC): median write latency/request, slow-delete share "
        "(all measured | Stage-A successes only)"
    )
    for window in windows:
        if window["day"] == "2026-10-09":
            print(
                f"  {window['window_utc']} {window['epochs']:<9} n={window['n_runs']:>3} (x{window['n_stage_a_failed']}) "
                f"lat {tr.fmt(window['median_write_req_lat_ms'])} | {tr.fmt(window['median_write_req_lat_ms_success'])} ms  "
                f"slow {tr.fmt(window['slow_delete_share'], 2)} | {tr.fmt(window['slow_delete_share_success'], 2)}"
            )

    print("\nSlow-delete share per day (delete > 25 ms, single-delete runs)")
    slow_summary: dict[str, dict[str, tuple[int, int]]] = {}
    for day in DAYS:
        day_pts = [p for p in points if p.day == day]
        main_pts = [p for p in day_pts if p.gold or run_role(p) == "main matrix (arm run)"]
        pipe_pts = [p for p in day_pts if p.pipeline]
        slow_summary[day] = {
            "all": slow_share(day_pts),
            "success": slow_share(in_view(day_pts, "success")),
            "main_all": slow_share(main_pts),
            "main_success": slow_share(in_view(main_pts, "success")),
            "pipe_all": slow_share(pipe_pts),
            "pipe_success": slow_share(in_view(pipe_pts, "success")),
            "executed_ok": executed_slow_share(runs, day),
        }
        print(
            f"  {day}: "
            + "; ".join(f"{label} {_share(slow_summary[day][key])}" for key, label in SLOW_KEYS)
        )

    drift = drift_rows(points, runs)
    _csv(tab_dir / "drift_by_epoch.csv", drift)
    (tex_dir / "drift_by_epoch.tex").write_text(tex_drift(drift), encoding="utf-8")
    print("\nResidual drift per epoch (write per-request latency / write tp; uncorrected)")
    for row in drift:
        if row["view"] == "success" and not row["n_stage_a_failed_measured"]:
            continue
        tag = "all measured" if row["view"] == "all" else "A-ok only"
        role = "main" if row["role"] == ROLE_MAIN_MODELS else "flash-lite pipe"
        print(
            f"  {role:<15} {row['epoch']:<15} {tag:<12} n={row['n_resid']:>3} x{row['n_stage_a_failed']} rho_lat {_rho_plain(row['rho_write_req_lat'])} "
            f"p={tr.fmt_p_plain(row['p_write_req_lat'])} Holm={tr.fmt_p_plain(row.get('p_write_req_lat_holm'))} "
            f"rho_tp {_rho_plain(row['rho_write_tp'])} p={tr.fmt_p_plain(row['p_write_tp'])} KW_rep p={tr.fmt_p_plain(row['kw_rep_p_write_req_lat'])} "
            f"slow {row['n_slow_delete']}/{row['n_delete']}"
        )

    gold = gold_drift_rows(points)
    _csv(tab_dir / "gold_drift.csv", gold)
    (tex_dir / "gold_drift.tex").write_text(tex_gold(gold), encoding="utf-8")
    print("\nGold across days (ratio later/earlier, HL 95% CI, MWU p)")
    for row in gold:
        print(
            f"  {row['dialect']} {row['first']}->{row['second']} {row['op']:<6} {row['metric']:<7} {row['median_first']:.1f}->{row['median_second']:.1f} "
            f"(median x{row['median_ratio']:.2f}) HL x{row['hl_ratio']:.2f} [{row['hl_lo95']:.2f}, {row['hl_hi95']:.2f}] p={row['mwu_p']:.3f}"
            + (f" Holm {row['mwu_p_holm']:.3f}" if "mwu_p_holm" in row else "")
        )

    retest, sd_pairs = same_day_pairs(runs, points, root)
    _csv(tab_dir / "retest_noise.csv", retest)
    (tex_dir / "pipeline_retest_noise.tex").write_text(tex_retest(retest), encoding="utf-8")
    print(
        "\nSame-day identical-artefact pairs, flash-lite pipeline development "
        "(pairs / identical / both measured / both Stage-A ok)"
    )
    for row in retest:
        tp_ratio = row.get("write_tp_ratio_median")
        print(
            f"  {row['pair']:<8} {row['level']:<4} {row['prompting']:<3} {row['n_pairs']:>3} {row['n_identical']:>3} "
            f"{row['n_identical_measured']:>3} {row['n_identical_success']:>3}"
            + (
                f"  write tp ratio {tp_ratio:.2f} sd(log) {tr.fmt(row.get('write_tp_sd_log_ratio'), 2)}"
                if tp_ratio
                else ""
            )
        )

    band = noise_band_rows(sd_pairs)
    _csv(tab_dir / "noise_band.csv", band)
    (tex_dir / "pipeline_noise_band.tex").write_text(tex_noise_band(band), encoding="utf-8")
    print("\nNoise floor (same-day identical flash-lite pairs; view all | success)")
    for row in band:
        print(
            f"  {row['view']:<8} {row['op']:<6} {row['metric']:<7} n={row['n_pairs']} median ratio {row['median_ratio']:.2f} med|log| {row['median_abs_log']:.2f} "
            f"in 0.8-1.25 {row['share_0.8_1.25']:.2f} range {row['ratio_min']:.2f}-{row['ratio_max']:.2f} sd(log ratio) {row['sd_log_ratio']:.2f} "
            f"sd_run {row['sd_log_single_run']:.2f} rho12 {row['spearman_first_second']:+.2f} band10 {row['band10_lo_pct']:+.0f}..{row['band10_hi_pct']:+.0f}%"
        )

    cross, cd_pairs = cross_day_rows(points)
    _csv(tab_dir / "retest_crossday.csv", cross)
    (tex_dir / "retest_crossday.tex").write_text(tex_crossday(cross, cd_pairs), encoding="utf-8")
    pair_kinds = Counter(
        f"E{first.epoch} {tr.MODEL_LABEL[first.model]} -> E{second.epoch}"
        for first, second in cd_pairs
    )
    print(
        f"\nCross-day identical measured pairs: {len(cd_pairs)} "
        f"(with a Stage-A failure: {sum(not (first.ok and second.ok) for first, second in cd_pairs)}); "
        + ", ".join(f"{kind} ({count})" for kind, count in sorted(pair_kinds.items()))
    )
    for row in cross:
        print(
            f"  {row['group']:<18} {row['view']:<8} {row['op']:<6} {row['metric']:<7} n={row['n_pairs']} median ratio x{row['median_ratio']:.2f} "
            f"Wilcoxon p={tr.fmt_p_plain(row['wilcoxon_p'])}"
            + (
                f" Holm {tr.fmt_p_plain(row['wilcoxon_p_holm'])}"
                if row["holm_family"]
                else " (not in Holm family)"
            )
        )

    n_pair_fail = {
        "same-day": sum(not (first.ok and second.ok) for first, second in sd_pairs),
        "cross-day": sum(not (first.ok and second.ok) for first, second in cd_pairs),
    }
    thesis_info = {
        "n_main": str(sum(not p.gold for p in thesis_pts)),
        "n_gold": str(sum(p.gold for p in thesis_pts)),
        "cuts": f"{cuts['write_ms']:.2f} ms (window median write latency) and {cuts['slow_delete_share']:.3f} (slow-delete share)",
        "spans": _span_text(spans),
    }
    _write_readme(out, slow_summary, timeline_views, drift, n_pair_fail, cross, thesis_info)
    print(f"\nWritten to {out}")


def _share(counts: tuple[int, int]) -> str:
    n_slow, n = counts
    return f"{n_slow}/{n} = {n_slow / n:.2f}" if n else "--"


def _write_readme(
    out: Path,
    slow: dict[str, dict[str, tuple[int, int]]],
    views: dict[str, dict[str, int]],
    drift: Sequence[dict[str, Any]],
    n_pair_fail: dict[str, int],
    cross: Sequence[dict[str, Any]],
    thesis: dict[str, str],
) -> None:
    n_stage_b = sum(row["n_stage_b_failed"] or 0 for row in drift if row["view"] == "all")
    view_all, view_success = views["all"], views["success"]
    if any(n_pair_fail.values()):
        pair_line = (
            "- Identical pairs with a Stage-A failure: "
            + ", ".join(f"{kind} {count}" for kind, count in n_pair_fail.items())
            + "; see the `view` / `_success` columns of `noise_band.csv`, `retest_crossday.csv` and `retest_noise.csv`."
        )
    else:
        pair_line = "- The same-day and cross-day identical pairs contain no Stage-A failure, so the noise floor and the cross-day shift are the same in both views."
    slow_lines = "\n".join(
        f"  - {DAY_LABEL[day]}: "
        + "; ".join(f"{label} {_share(shares[key])}" for key, label in SLOW_KEYS[:-1])
        for day, shares in slow.items()
    )
    drift_lines = []
    for row in drift:
        if row["view"] != "all" or not row["n_stage_a_failed"]:
            continue
        success = next(
            other
            for other in drift
            if other["view"] == "success"
            and other["role"] == row["role"]
            and other["epoch"] == row["epoch"]
        )
        drift_lines.append(
            f"  - {row['epoch']} [{row['role']}] ({row['n_stage_a_failed']} of {row['n_runs']} measured runs failed Stage A): "
            f"rho_lat {_rho_plain(row['rho_write_req_lat'])} (p {tr.fmt_p_plain(row['p_write_req_lat'])}) vs. "
            f"{_rho_plain(success['rho_write_req_lat'])} (p {tr.fmt_p_plain(success['p_write_req_lat'])}); "
            f"rho_tp {_rho_plain(row['rho_write_tp'])} vs. {_rho_plain(success['rho_write_tp'])}; "
            f"slow deletes {row['n_slow_delete']}/{row['n_delete']} vs. {success['n_slow_delete']}/{success['n_delete']}"
        )
    cross_n = {
        group: next(
            row["n_pairs"] for row in cross if row["group"] == group and row["view"] == "all"
        )
        for group, _, _ in CROSS_DAY_GROUPS
    }
    text = f"""# Measurement timeline, drift and noise floor

Generated by `uv run python scripts/thesis_timeline.py` (imports `scripts/thesis_results.py`).
Thesis sections 4.1 (setup, measurement schedule) and 4.5 (threats to validity). All tests are exploratory;
uncorrected unless a Holm column is given. Epochs are never pooled. Measurement days: 28 Aug 2026 (E6-E11),
13 Sep 2026 (E12/E13 incl. the gold series), 9 Oct 2026 (E14-E17 incl. the gold series).

## For the thesis
- `fig/measurement_timeline.pdf/.png` (main text) and `fig/measurement_timeline_delete.pdf/.png` (appendix), floats with
  captions in `latex/figures.tex` (`fig:measurement-timeline`, `fig:measurement-timeline-delete`). One row, one panel per
  measurement day, the same hours per inch in every panel; Stage-A successes of the 12 main arms ({thesis["n_main"]} runs) and the
  {thesis["n_gold"]} gold reference runs only (no other run; the flash multi-shot L3 runs of E8/E9 outside the arms are left out).
  Model colour, gold = black diamonds, grey line = rolling median of 15 runs of a day, epoch labels at the top. An isolated late
  group of runs (the two E9 re-evaluations at 15:31 on 28 Aug) sits in a narrow segment behind an x-axis break.
- Slow / fast phases (shaded): 30-minute windows of these runs with at least {PHASE_MIN_RUNS} runs; two indicators per window (median write
  latency per request, slow-delete share), each split into two groups by an exact 1-D two-means split over the windows
  (boundaries {thesis["cuts"]}); a window is shaded only if both indicators agree, adjacent windows of one class are
  merged. Result: {thesis["spans"]} (UTC). Numbers behind it: `tables/timeline_phases.csv`.
- Tables: `latex/drift_by_epoch.tex` (main models only), `latex/gold_drift.tex`, `latex/retest_crossday.tex` (flash pairs only).
- Everything else in this folder is kept for the record: the old two-row figures `measurement_timeline_all` /
  `measurement_timeline_success` and the `pipeline_*` LaTeX tables contain flash-lite runs and do not go into the thesis.

## Which model is evidence for what
- The model comparison uses one Gemini model, **gemini-3.5-flash** (flash), next to claude-haiku-5-5 (haiku) and glm-5.3 (glm).
- **gemini-3.5-flash-lite was used only to develop and test the pipeline cheaply.** It is never model evidence here.
  It appears for one reason: its runs were measured on the same SUT on the same days, so they are part of the measurement
  environment (load on the SUT, time-of-day phases, slow-delete phases). Where it appears it is labelled
  "flash-lite (pipeline development)":
  - the timeline figures draw it subdued (light grey dots, grey epoch bars for the flash-lite-only epochs E6/E7/E10/E11);
  - `drift_by_epoch.csv` has a separate flash-lite block with its own Holm family (column `role`; not in the LaTeX table);
  - `retest_noise` and `noise_band` consist only of flash-lite identical-artefact pairs (column `role`): identical code
    measured twice, i.e. measurement noise, not model performance; their LaTeX is `pipeline_retest_noise.tex` /
    `pipeline_noise_band.tex` (record only);
  - `retest_crossday` reports flash (E9->E12, {cross_n["flash E9->E12"]} pairs) and flash-lite (E7->E12, {cross_n["flash-lite E7->E12"]} pairs) as separate groups,
    plus a pooled group ({cross_n["pooled"]} pairs) labelled "measurement noise only"; the LaTeX table shows the flash group only;
  - `timeline_runs.csv` carries a `role` column, `timeline_windows.csv` an `n_pipeline_flash_lite` count.

## Conventions
- Full spectrum: every run that executed and produced a proxy value ("measured", the rule of `thesis_results.sb` /
  `RunRecord.stage_b_value`) is plotted and enters the statistics, including runs that executed but then failed Stage A
  ({view_all["failed"]} in the timeline: {view_all["failed_main"]} main-model runs (glm in E16/E17, workload not conformant) and
  {view_all["failed_pipeline"]} flash-lite run in E6). They are never dropped silently: x markers in the figure, a dagger count in the tables,
  `stage_a_success` in `timeline_runs.csv`.
- Two equal-ranking views, each its own figure: `measurement_timeline_all` (all measured runs) and
  `measurement_timeline_success` (Stage-A successes only). CSVs carry a `view` column (`all` / `success`) or `_success`
  columns; the LaTeX tables add an "A-ok only" row wherever the views differ and say so in the note where they do not.
- Failure stages are kept apart: Stage-A failure = executed but not syntax/executed/workload conformant (kept and marked if measured);
  Stage-B failure = a Stage-A success without a proxy value for an operation ({n_stage_b} in E6-E17, reported in `drift_by_epoch.csv`).
- No metric is primary. Per-request latency, throughput, the slow-delete share and the gold ratios are indicators read together.
- Gold series = `_gold_*` fixtures (E12/E13 on 13 Sep, E16/E17 on 9 Oct), all Stage-A successes.
- Pinned request counts: first 10 write / first 20 read requests from `evaluation/summary.json` (`proxy_stats.requests`, by `started_at`).
  A request-level value is only taken when the run has a proxy value for that operation.
- Per-request latency = median `latency_ms` of requests 2..n (request 1 is a cold start). Delete latency = the first delete request.
- Slow delete = delete latency > 25 ms (= delete throughput < 40 ops/s), only runs with exactly one delete request.

## Files
- `fig/measurement_timeline.pdf/.png`, `fig/measurement_timeline_delete.pdf/.png`, `latex/figures.tex`,
  `tables/timeline_phases.csv` - the thesis timeline, see above.
- Record only: `fig/measurement_timeline_all.pdf/.png` - view "all measured runs": {view_all["n"]} runs ({view_all["main"]} main-model, {view_all["pipeline"]} flash-lite,
  {view_all["gold"]} gold), {view_all["failed"]} of them failed Stage A (x in model colour; grey x for flash-lite). "main-model" counts every
  flash/haiku/glm run, including the flash multi-shot L3 runs of E8/E9 outside the 12 arms (`role` in `timeline_runs.csv`).
- `fig/measurement_timeline_success.pdf/.png` - view "Stage-A successes only": {view_success["n"]} runs ({view_success["main"]} main-model,
  {view_success["pipeline"]} flash-lite, {view_success["gold"]} gold).
  Both: per-run write latency per request (top) and delete latency (bottom) against UTC evaluation time, one column per
  measurement day; colour = main model, light grey = flash-lite (pipeline development, drawn subdued), black diamonds = gold block,
  grey line = rolling median of 15 runs over the runs shown in that figure (all generated runs of the day, flash-lite included,
  because it describes the environment), grey band = E12 L3 block. The view label with the counts is printed in the figure.
- `tables/drift_by_epoch.csv`, `latex/drift_by_epoch.tex` (`tab:drift-by-epoch`, main-model rows only) - per role (main models / flash-lite pipeline),
  epoch and view: Spearman of the cell-residual (model x prompting x level) of log write latency / log write tp against time,
  Kruskal-Wallis over repetition, slow-delete share, Stage-A-failed and Stage-B-failed counts; plus an `E12 without L3`
  sensitivity row (main models). Holm per role and view. The main-model rows of E8/E9 include the measured flash multi-shot L3 runs
  of these epochs (10 generated per epoch) that belong to no main arm (`n_outside_arms`).
- `tables/gold_drift.csv`, `latex/gold_drift.tex` (`tab:gold-drift`) - gold Kestrel E13 vs E16 and Tessera E12 vs E17 per operation:
  medians, Hodges-Lehmann ratio with 95% CI, Mann-Whitney p, Holm over the six latency tests (gold only, one view; no model involved).
- `tables/retest_noise.csv`, `latex/pipeline_retest_noise.tex` (`tab:retest-noise`, record only) - **pipeline development (flash-lite), measurement noise only**:
  same-day pairs (E6->E10, E7->E11 on 28 Aug) from `thesis_results.retest_pairs`, split by level and prompting: counts, identical,
  both measured, of these with a Stage-A failure, write ratio median/IQR/range/sd(log) for all measured pairs and (`_success`)
  both-successful pairs.
- `tables/noise_band.csv`, `latex/pipeline_noise_band.tex` (`tab:noise-band`, record only) - **pipeline development (flash-lite), measurement noise only**:
  single-run noise per operation and metric and the 95% band of a 10-vs-10 median ratio (parametric bootstrap, lognormal with
  single-run sd = sd(log ratio)/sqrt 2), both views.
- `tables/retest_crossday.csv`, `latex/retest_crossday.tex` (`tab:retest-crossday`, flash group only) - byte-identical measured artefacts from 28 Aug
  measured again on 13 Sep (E12), groups flash / flash-lite / pooled in the CSV (column `group`, `role`): ratios per metric, Wilcoxon p,
  Holm within each group and view over write/read/delete tp and read p50.
- `tables/timeline_runs.csv` - one row per plotted run (`role`, `stage_a_success`, `in_view_all`, `in_view_success`);
  `tables/timeline_windows.csv` - 30-min windows per day (median write latency, slow-delete share; all measured and `_success`;
  every run measured in the window, with the flash-lite count).

## Where the two views differ
- Timeline: {view_all["n"]} vs. {view_success["n"]} runs shown ({view_all["failed"]} Stage-A failures dropped in the success view).
- Drift per epoch (all measured vs. Stage-A successes only):
{chr(10).join(drift_lines)}
{pair_line}
- Slow-delete share per day (delete > 25 ms):
{slow_lines}

## Caveats
- Identical-artefact pairs measure measurement noise, not replication of the generation step. All same-day pairs are flash-lite on one day,
  so the band is a lower bound; it is not a statement about flash-lite or any other model.
- The E12 L3 block (flash and flash-lite runs; 10 and 9 measured) ran after the gold block (18:02-18:12 UTC); cross-day shifts for E12 L3 mix day and time of day.
- On 9 Oct the arms ran as sequential blocks (haiku-K, glm-K, gold K, haiku-T, glm-T, gold T), so time-of-day phases are confounded with arm.
- The timeline does not compare models: models ran in different blocks and on different days.
"""
    (out / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
