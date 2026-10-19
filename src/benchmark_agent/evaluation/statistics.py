from __future__ import annotations

import math
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from statistics import median
from typing import Any

from pydantic import BaseModel, Field

from benchmark_agent.evaluation.coverage import REQUIRED_OPERATIONS
from benchmark_agent.evaluation.failure_class import failure_class_counts
from benchmark_agent.evaluation.runs import SOURCE_PROXY, SOURCE_SERVER, RunRecord
from benchmark_agent.models import GuidanceLevel, PromptingStrategy

LEVEL_ORDER: tuple[str, ...] = tuple(level.value for level in GuidanceLevel)
PROMPTING_ORDER: tuple[str, ...] = tuple(strategy.value for strategy in PromptingStrategy)
MIN_GROUP_N = 2
MIN_PAIRS = 3


class GroupSummary(BaseModel):
    key: str
    label: str
    guidance_level: str | None = None
    model: str | None = None
    prompting: str | None = None
    n: int = 0
    n_syntax_ok: int = 0
    n_executed_ok: int = 0
    n_semantics_ok: int = 0
    n_success: int = 0
    success_rate: float | None = None
    syntax_rate: float | None = None
    executed_rate: float | None = None
    semantics_rate: float | None = None
    mean_coverage: float | None = None
    mean_feedback_iterations: float | None = None
    n_stage_b: int = 0
    median_throughput: float | None = None
    iqr_throughput: float | None = None
    min_throughput: float | None = None
    max_throughput: float | None = None
    naive_mean_throughput: float | None = None
    median_latency_p50: float | None = None
    median_latency_p95: float | None = None
    median_latency_p99: float | None = None
    request_error_rate: float | None = None
    throughput_values: list[float] = Field(default_factory=list)
    failure_classes: dict[str, int] = Field(default_factory=dict)


class TestResult(BaseModel):
    name: str
    description: str
    groups: list[str] = Field(default_factory=list)
    group_sizes: list[int] = Field(default_factory=list)
    n: int = 0
    statistic_name: str = ""
    statistic: float | None = None
    p_value: float | None = None
    effect_name: str | None = None
    effect: float | None = None
    applicable: bool = False
    note: str = ""
    family: str = ""
    p_adjusted: float | None = None
    adjust_method: str = ""


PERMUTATIONS = 20_000
PERMUTATION_SEED = 20260715
CONTROL_LEVEL = "L1"
REFERENCE_LEVEL = "L2"


class StatisticsReport(BaseModel):
    operation: str
    source: str
    n_runs: int = 0
    n_measured: int = 0
    funnel: dict[str, int] = Field(default_factory=dict)
    overall: GroupSummary
    cells: list[GroupSummary] = Field(default_factory=list)
    by_level: list[GroupSummary] = Field(default_factory=list)
    by_model: list[GroupSummary] = Field(default_factory=list)
    by_prompting: list[GroupSummary] = Field(default_factory=list)
    by_operation: list[GroupSummary] = Field(default_factory=list)
    tests: list[TestResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def compute_statistics(
    runs: Sequence[RunRecord],
    *,
    operation: str = "write",
    source: str = SOURCE_PROXY,
    include_fixtures: bool = False,
) -> StatisticsReport:
    considered = [run for run in runs if include_fixtures or not run.is_fixture]
    measured = [run for run in considered if run.status != "not_evaluated"]
    report = StatisticsReport(
        operation=operation,
        source=source,
        n_runs=len(considered),
        n_measured=len(measured),
        funnel=funnel_counts(measured),
        overall=summarize_group(
            measured, key="all", label="alle Runs", operation=operation, source=source
        ),
    )
    if not measured:
        report.notes.append("Keine ausgewerteten Runs, Statistik leer.")
        return report

    report.cells = _cells(measured, operation, source)
    report.by_level = _grouped(
        measured, "guidance_level", LEVEL_ORDER, operation, source, lambda run: run.guidance_level
    )
    report.by_model = _grouped(measured, "model", (), operation, source, lambda run: run.model)
    report.by_prompting = _grouped(
        measured, "prompting", PROMPTING_ORDER, operation, source, lambda run: run.prompting
    )
    report.by_operation = [
        summarize_group(measured, key=op, label=op, operation=op, source=source)
        for op in REQUIRED_OPERATIONS
    ]
    report.tests = [
        kruskal_by_level(measured, operation, source),
        spearman_level_vs_throughput(measured, operation, source),
        mann_whitney_prompting(measured, operation, source),
        chi_square_success_by_level(measured),
        fisher_control_vs_rest(measured),
        permutation_success_by_level(measured),
        *pairwise_fisher_vs_reference(measured),
        proxy_server_agreement(measured, operation),
    ]
    apply_holm(report.tests)
    if len(measured) < 5:
        report.notes.append(
            f"Nur {len(measured)} ausgewertete Runs; Kennzahlen sind Momentaufnahmen, "
            "keine belastbare Statistik (Exposé: N >= 5 pro Kondition)."
        )
    return report


def funnel_counts(runs: Iterable[RunRecord]) -> dict[str, int]:
    listed = list(runs)
    return {
        "evaluated": len(listed),
        "syntax_ok": sum(1 for run in listed if run.syntax_ok),
        "executed_ok": sum(1 for run in listed if run.syntax_ok and run.executed_ok),
        "semantics_ok": sum(1 for run in listed if run.success),
    }


def summarize_group(
    runs: Sequence[RunRecord],
    *,
    key: str,
    label: str,
    operation: str,
    source: str,
    guidance_level: str | None = None,
    model: str | None = None,
    prompting: str | None = None,
) -> GroupSummary:
    n = len(runs)
    summary = GroupSummary(
        key=key,
        label=label,
        guidance_level=guidance_level,
        model=model,
        prompting=prompting,
        n=n,
        n_syntax_ok=sum(1 for run in runs if run.syntax_ok),
        n_executed_ok=sum(1 for run in runs if run.executed_ok),
        n_semantics_ok=sum(1 for run in runs if run.semantics_ok),
        n_success=sum(1 for run in runs if run.success),
    )
    if n == 0:
        return summary
    summary.success_rate = summary.n_success / n
    summary.syntax_rate = summary.n_syntax_ok / n
    summary.executed_rate = summary.n_executed_ok / n
    summary.semantics_rate = summary.n_semantics_ok / n
    coverages = [run.coverage_ratio for run in runs if run.coverage_ratio is not None]
    summary.mean_coverage = sum(coverages) / len(coverages) if coverages else None
    summary.mean_feedback_iterations = sum(run.feedback_iterations for run in runs) / n
    summary.failure_classes = failure_class_counts(list(runs))

    throughput = stage_b_values(runs, operation, source, "throughput_ops_s")
    summary.n_stage_b = len(throughput)
    summary.throughput_values = throughput
    if throughput:
        summary.median_throughput = median(throughput)
        summary.iqr_throughput = iqr(throughput)
        summary.min_throughput = min(throughput)
        summary.max_throughput = max(throughput)
    summary.naive_mean_throughput = sum(throughput) / n if n else None
    for percentile in ("p50", "p95", "p99"):
        values = stage_b_values(runs, operation, source, f"latency_{percentile}_ms")
        setattr(summary, f"median_latency_{percentile}", median(values) if values else None)
    requests = errors = 0
    for run in runs:
        stats = run.op(source, operation)
        if stats is not None:
            requests += stats.n_requests
            errors += stats.error_count
    summary.request_error_rate = errors / requests if requests else None
    return summary


def stage_b_values(
    runs: Iterable[RunRecord], operation: str, source: str, metric: str
) -> list[float]:
    values = [run.stage_b_value(source, operation, metric) for run in runs]
    return [value for value in values if value is not None]


def iqr(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    ordered = sorted(values)
    return _percentile(ordered, 0.75) - _percentile(ordered, 0.25)


def _percentile(ordered: Sequence[float], p: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    rank = p * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _cells(runs: Sequence[RunRecord], operation: str, source: str) -> list[GroupSummary]:
    grouped: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[(run.guidance_level or "?", run.model or "?", run.prompting or "?")].append(run)
    cells: list[GroupSummary] = []
    for level, model, prompting in sorted(grouped, key=_cell_sort_key):
        members = grouped[(level, model, prompting)]
        cells.append(
            summarize_group(
                members,
                key=f"{level}|{model}|{prompting}",
                label=f"{level} · {model} · {prompting}",
                operation=operation,
                source=source,
                guidance_level=level,
                model=model,
                prompting=prompting,
            )
        )
    return cells


def _cell_sort_key(cell: tuple[str, str, str]) -> tuple[int, str, int]:
    level, model, prompting = cell
    return (_order_index(level, LEVEL_ORDER), model, _order_index(prompting, PROMPTING_ORDER))


def _order_index(value: str, order: Sequence[str]) -> int:
    return order.index(value) if value in order else len(order)


def _sorted_by_order(values: Iterable[str], order: Sequence[str]) -> list[str]:
    return sorted(values, key=lambda value: (_order_index(value, order), value))


def _grouped(
    runs: Sequence[RunRecord],
    key: str,
    order: Sequence[str],
    operation: str,
    source: str,
    getter: Callable[[RunRecord], str | None],
) -> list[GroupSummary]:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for run in runs:
        grouped[getter(run) or "?"].append(run)
    summaries: list[GroupSummary] = []
    for value in _sorted_by_order(grouped, order):
        fields = {key: value} if key in {"guidance_level", "model", "prompting"} else {}
        summaries.append(
            summarize_group(
                grouped[value],
                key=value,
                label=value,
                operation=operation,
                source=source,
                **fields,
            )
        )
    return summaries


def _values_by_group(
    runs: Iterable[RunRecord],
    operation: str,
    source: str,
    getter: Callable[[RunRecord], str | None],
    order: Sequence[str],
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        value = run.stage_b_value(source, operation, "throughput_ops_s")
        if value is not None:
            grouped[getter(run) or "?"].append(value)
    return {key: grouped[key] for key in _sorted_by_order(grouped, order)}


def kruskal_by_level(runs: Sequence[RunRecord], operation: str, source: str) -> TestResult:
    result = TestResult(
        name="Kruskal-Wallis",
        description=(
            f"Throughput ({operation}, {source}) ~ Guidance-Level; nur Stufe-B-Runs "
            "(ausgeführt, >= 1 fehlerfreier Request)."
        ),
        statistic_name="H",
        effect_name="ε² (epsilon squared)",
        family="stage_b",
    )
    groups = {
        key: values
        for key, values in _values_by_group(
            runs, operation, source, lambda run: run.guidance_level, LEVEL_ORDER
        ).items()
        if len(values) >= MIN_GROUP_N
    }
    result.groups = list(groups)
    result.group_sizes = [len(values) for values in groups.values()]
    result.n = sum(result.group_sizes)
    if len(groups) < 2:
        result.note = (
            f"Mindestens 2 Level mit je >= {MIN_GROUP_N} Stufe-B-Werten nötig "
            f"(vorhanden: {len(groups)})."
        )
        return result
    from scipy import stats

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outcome = stats.kruskal(*groups.values())
    result.statistic = float(outcome.statistic)
    result.p_value = float(outcome.pvalue)
    k = len(groups)
    result.effect = result.statistic / (result.n - 1) if result.n > 1 else None
    result.applicable = True
    result.note = f"{k} Gruppen; Post-hoc (paarweise Mann-Whitney mit Korrektur) bei p < 0.05."
    return result


def spearman_level_vs_throughput(
    runs: Sequence[RunRecord], operation: str, source: str
) -> TestResult:
    result = TestResult(
        name="Spearman",
        description=(
            f"Guidance-Index (L1=1 … L5=5) vs. Throughput je Run ({operation}, {source})."
        ),
        statistic_name="ρ",
        effect_name="ρ (Rangkorrelation)",
        family="stage_b",
    )
    pairs: list[tuple[int, float]] = []
    for run in runs:
        value = run.stage_b_value(source, operation, "throughput_ops_s")
        if value is None or run.guidance_level not in LEVEL_ORDER:
            continue
        pairs.append((LEVEL_ORDER.index(run.guidance_level) + 1, value))
    result.n = len(pairs)
    result.groups = sorted({f"L{level}" for level, _ in pairs})
    if len(pairs) < MIN_PAIRS or len({level for level, _ in pairs}) < 2:
        result.note = f"Mindestens {MIN_PAIRS} Stufe-B-Runs über >= 2 Level nötig."
        return result
    from scipy import stats

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outcome = stats.spearmanr([level for level, _ in pairs], [value for _, value in pairs])
    rho = float(outcome.statistic)
    result.statistic = rho
    result.effect = rho
    result.p_value = float(outcome.pvalue)
    result.applicable = not math.isnan(rho)
    if not result.applicable:
        result.note = "Konstante Werte, Korrelation undefiniert."
    return result


def mann_whitney_prompting(runs: Sequence[RunRecord], operation: str, source: str) -> TestResult:
    zero_shot, multi_shot = PROMPTING_ORDER
    result = TestResult(
        name="Mann-Whitney U",
        description=(
            f"Throughput ({operation}, {source}): {zero_shot} vs. {multi_shot}, alle Level/Modelle."
        ),
        statistic_name="U",
        effect_name="r (rang-biserial; positiv = zero_shot > multi_shot)",
        family="stage_b",
    )
    groups = _values_by_group(runs, operation, source, lambda run: run.prompting, PROMPTING_ORDER)
    zero_shot_values = groups.get(zero_shot, [])
    multi_shot_values = groups.get(multi_shot, [])
    result.groups = [zero_shot, multi_shot]
    result.group_sizes = [len(zero_shot_values), len(multi_shot_values)]
    result.n = len(zero_shot_values) + len(multi_shot_values)
    if len(zero_shot_values) < MIN_GROUP_N or len(multi_shot_values) < MIN_GROUP_N:
        result.note = f"Beide Strategien brauchen >= {MIN_GROUP_N} Stufe-B-Werte."
        return result
    from scipy import stats

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outcome = stats.mannwhitneyu(zero_shot_values, multi_shot_values, alternative="two-sided")
    result.statistic = float(outcome.statistic)
    result.p_value = float(outcome.pvalue)
    result.effect = rank_biserial(result.statistic, len(zero_shot_values), len(multi_shot_values))
    result.applicable = True
    return result


def rank_biserial(u: float, n1: int, n2: int) -> float:
    return (2.0 * u) / (n1 * n2) - 1.0


def chi_square_success_by_level(runs: Sequence[RunRecord]) -> TestResult:
    result = TestResult(
        name="Chi-Quadrat",
        description="Erfolg (syntax ∧ executed ∧ semantics) × Guidance-Level, Stufe A.",
        statistic_name="χ²",
        effect_name="Cramér's V",
    )
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for run in runs:
        level = run.guidance_level or "?"
        counts[level][0 if run.success else 1] += 1
    levels = _sorted_by_order(
        (level for level in counts if sum(counts[level]) >= MIN_GROUP_N), LEVEL_ORDER
    )
    result.groups = levels
    result.group_sizes = [sum(counts[level]) for level in levels]
    result.n = sum(result.group_sizes)
    table = [counts[level] for level in levels]
    if len(levels) < 2 or any(sum(col) == 0 for col in zip(*table, strict=True)):
        result.note = (
            f"Mindestens 2 Level mit je >= {MIN_GROUP_N} Runs sowie Erfolge und "
            "Misserfolge insgesamt nötig."
        )
        return result
    from scipy import stats

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outcome = stats.chi2_contingency(table, correction=False)
    result.statistic = float(outcome.statistic)
    result.p_value = float(outcome.pvalue)
    k = min(len(levels), 2)
    result.effect = (result.statistic / (result.n * (k - 1))) ** 0.5 if result.n else None
    result.applicable = True
    expected = outcome.expected_freq
    if float(expected.min()) < 5:
        result.note = (
            "Erwartete Zellhäufigkeit < 5; Ergebnis nur orientierend "
            "(exakter Test bei finaler Auswertung)."
        )
    return result


def proxy_server_agreement(runs: Sequence[RunRecord], operation: str) -> TestResult:
    result = TestResult(
        name="Proxy vs. /metrics",
        description=(
            f"Cross-Check der Request-Zählung ({operation}): Proxy-Zeilen gegen die "
            "Server-Counter derselben Runs."
        ),
        statistic_name="ρ (Spearman)",
        effect_name="Median |Δ n_requests| relativ",
    )
    pairs: list[tuple[float, float]] = []
    for run in runs:
        proxy = run.op(SOURCE_PROXY, operation)
        server = run.op(SOURCE_SERVER, operation)
        if proxy is None or server is None or proxy.n_requests == 0:
            continue
        pairs.append((float(proxy.n_requests), float(server.n_requests)))
    result.n = len(pairs)
    if not pairs:
        result.note = "Keine Runs mit Proxy- und Server-Zeilen."
        return result
    deltas = [abs(proxy_count - server_count) / proxy_count for proxy_count, server_count in pairs]
    result.effect = median(deltas)
    if len(pairs) >= MIN_PAIRS:
        from scipy import stats

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outcome = stats.spearmanr(
                [proxy_count for proxy_count, _ in pairs],
                [server_count for _, server_count in pairs],
            )
        rho = float(outcome.statistic)
        if not math.isnan(rho):
            result.statistic = rho
            result.p_value = float(outcome.pvalue)
    result.applicable = True
    result.note = f"{len(pairs)} Run(s) mit beiden Quellen."
    return result


def _success_table(
    runs: Sequence[RunRecord], levels: Sequence[str]
) -> tuple[list[str], list[list[int]]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for run in runs:
        level = run.guidance_level
        if level is not None and level in levels:
            counts[level][0 if run.success else 1] += 1
    present = [level for level in levels if sum(counts[level]) >= MIN_GROUP_N]
    return present, [counts[level] for level in present]


def _success_row(runs: Sequence[RunRecord]) -> list[int]:
    successes = sum(1 for run in runs if run.success)
    failures = sum(1 for run in runs if not run.success)
    return [successes, failures]


def fisher_control_vs_rest(runs: Sequence[RunRecord]) -> TestResult:
    rest = [level for level in LEVEL_ORDER if level != CONTROL_LEVEL]
    result = TestResult(
        name="Fisher exakt (Kontrolle)",
        description=(
            f"Erfolg: {CONTROL_LEVEL} (Kontrolle, keine Spezifikation) vs. "
            f"{rest[0]}–{rest[-1]} gepoolt; exakter Test, keine Zellhäufigkeits-Annahme."
        ),
        statistic_name="Odds-Ratio",
        effect_name="Erfolgsquote Kontrolle − Rest",
    )
    control = [run for run in runs if run.guidance_level == CONTROL_LEVEL]
    others = [run for run in runs if run.guidance_level in rest]
    result.groups = [CONTROL_LEVEL, f"{rest[0]}–{rest[-1]}"]
    result.group_sizes = [len(control), len(others)]
    result.n = len(control) + len(others)
    if len(control) < MIN_GROUP_N or len(others) < MIN_GROUP_N:
        result.note = f"Beide Gruppen brauchen >= {MIN_GROUP_N} Runs."
        return result
    table = [_success_row(control), _success_row(others)]
    from scipy import stats

    odds, p_value = stats.fisher_exact(table)
    result.statistic = None if math.isnan(odds) or odds == float("inf") else float(odds)
    result.p_value = float(p_value)
    result.effect = table[0][0] / len(control) - table[1][0] / len(others)
    result.applicable = True
    result.note = f"{table[0][0]}/{len(control)} vs. {table[1][0]}/{len(others)}" + (
        "; Odds-Ratio 0 oder ∞ (leere Zelle)" if result.statistic is None else ""
    )
    return result


LADDER_LEVELS: tuple[str, ...] = tuple(level for level in LEVEL_ORDER if level != CONTROL_LEVEL)


def permutation_success_by_level(
    runs: Sequence[RunRecord],
    *,
    levels: Sequence[str] = LADDER_LEVELS,
    permutations: int = PERMUTATIONS,
    seed: int = PERMUTATION_SEED,
) -> TestResult:
    result = TestResult(
        name="Permutationstest L2–L5",
        description=(
            "Erfolg × Guidance-Level ohne Kontrolle (L2–L5): χ²-Statistik, p aus "
            f"{permutations} Permutationen der Erfolge über die Level (Seed {seed})."
        ),
        statistic_name="χ²",
        effect_name="Cramér's V",
    )
    present, table = _success_table(runs, levels)
    result.groups = present
    result.group_sizes = [sum(row) for row in table]
    result.n = sum(result.group_sizes)
    if len(present) < 2 or any(sum(col) == 0 for col in zip(*table, strict=True)):
        result.note = (
            f"Mindestens 2 Level mit je >= {MIN_GROUP_N} Runs sowie Erfolge und "
            "Misserfolge insgesamt nötig."
        )
        return result
    import numpy as np

    level_blocks = [np.full(sum(row), index) for index, row in enumerate(table)]
    outcome_blocks = [np.array([1] * row[0] + [0] * row[1]) for row in table]
    labels = np.concatenate(level_blocks).astype(int)
    outcomes = np.concatenate(outcome_blocks).astype(int)
    observed = _chi2_statistic(labels, outcomes, len(table))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        permuted = rng.permutation(outcomes)
        if _chi2_statistic(labels, permuted, len(table)) >= observed - 1e-12:
            exceed += 1
    result.statistic = float(observed)
    result.p_value = (exceed + 1) / (permutations + 1)
    result.effect = (observed / result.n) ** 0.5 if result.n else None
    result.applicable = True
    result.note = ", ".join(
        f"{level} {row[0]}/{sum(row)}" for level, row in zip(present, table, strict=True)
    )
    return result


def _chi2_statistic(labels: Any, outcomes: Any, k: int) -> float:
    import numpy as np

    n = labels.shape[0]
    successes = np.bincount(labels, weights=outcomes, minlength=k)
    sizes = np.bincount(labels, minlength=k).astype(float)
    p_success = successes.sum() / n
    if p_success in (0.0, 1.0):
        return 0.0
    expected_success = sizes * p_success
    expected_failure = sizes * (1 - p_success)
    failures = sizes - successes
    return float(
        ((successes - expected_success) ** 2 / expected_success).sum()
        + ((failures - expected_failure) ** 2 / expected_failure).sum()
    )


def pairwise_fisher_vs_reference(
    runs: Sequence[RunRecord], *, reference: str = REFERENCE_LEVEL
) -> list[TestResult]:
    results: list[TestResult] = []
    ref_runs = [run for run in runs if run.guidance_level == reference]
    for level in LEVEL_ORDER:
        if level in (reference, CONTROL_LEVEL):
            continue
        other = [run for run in runs if run.guidance_level == level]
        result = TestResult(
            name=f"Fisher exakt {reference} vs. {level}",
            description=(
                f"Erfolg: {reference} (nur Spezifikation) vs. {level}; paarweise, "
                "Holm-korrigiert über alle Paare."
            ),
            statistic_name="Odds-Ratio",
            effect_name=f"Erfolgsquote {level} − {reference}",
            family="stage_a_pairwise",
            groups=[reference, level],
            group_sizes=[len(ref_runs), len(other)],
            n=len(ref_runs) + len(other),
        )
        if len(ref_runs) < MIN_GROUP_N or len(other) < MIN_GROUP_N:
            result.note = f"Beide Level brauchen >= {MIN_GROUP_N} Runs."
            results.append(result)
            continue
        table = [_success_row(ref_runs), _success_row(other)]
        from scipy import stats

        odds, p_value = stats.fisher_exact(table)
        result.statistic = None if math.isnan(odds) or odds == float("inf") else float(odds)
        result.p_value = float(p_value)
        result.effect = table[1][0] / len(other) - table[0][0] / len(ref_runs)
        result.applicable = True
        result.note = f"{table[0][0]}/{len(ref_runs)} vs. {table[1][0]}/{len(other)}"
        results.append(result)
    return results


def apply_holm(tests: Sequence[TestResult]) -> None:
    families: dict[str, list[TestResult]] = defaultdict(list)
    for test in tests:
        if test.family and test.applicable and test.p_value is not None:
            families[test.family].append(test)
        elif test.p_value is not None:
            test.p_adjusted = test.p_value
    for members in families.values():
        m = len(members)
        ordered = sorted(members, key=lambda member: member.p_value or 0.0)
        running_max = 0.0
        for rank, test in enumerate(ordered):
            adjusted = min(1.0, (m - rank) * (test.p_value or 0.0))
            running_max = max(running_max, adjusted)
            test.p_adjusted = running_max
            test.adjust_method = f"Holm (m={m})"
