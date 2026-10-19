import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(sys.argv[1]) / "scripts"))
import thesis_results as tr

from benchmark_agent.evaluation import statistics as st
from benchmark_agent.evaluation.runs import RunRecord

ROOT = Path(sys.argv[1])
config, runs = tr.load_runs()
ARMS = [
    ("haiku", "claude_haiku", "Kestrel", 14),
    ("haiku", "claude_haiku", "Tessera", 15),
    ("glm", "glm", "Kestrel", 16),
    ("glm", "glm", "Tessera", 17),
]
GOLD_EPOCH = {"Kestrel": 16, "Tessera": 17}
TODAY = "2026-10-09"


def gold(dialect: str) -> list[RunRecord]:
    return [
        run
        for run in runs
        if run.is_fixture
        and run.condition_name == f"_gold_{dialect.lower()}"
        and run.epoch == GOLD_EPOCH[dialect]
        and (run.evaluated_at or "").startswith(TODAY)
    ]


for dialect in ("Kestrel", "Tessera"):
    gold_runs = gold(dialect)
    for op in ("write", "read"):
        values = tr.sb(gold_runs, op, "throughput_ops_s")
        if values:
            q1, median, q3 = tr.quart(values)
            print(f"GOLD {dialect} {op}: n={len(values)} median={median:.1f} IQR {q1:.1f}-{q3:.1f}")

for label, model, dialect, epoch in ARMS:
    for prompting in ("zero_shot", "multi_shot"):
        arm = [
            run
            for run in runs
            if run.epoch == epoch
            and run.model == model
            and run.prompting == prompting
            and not run.is_fixture
        ]
        if not arm:
            continue
        runs_by_level = tr.by_level(arm)
        success_counts = []
        for level in tr.LEVELS:
            members = runs_by_level[level]
            success_counts.append(f"{level} {sum(run.success for run in members)}/{len(members)}")
        success_text = " ".join(success_counts)
        feedback_run_count = sum(
            run.feedback_iterations > 0 for level in ("L4", "L5") for run in runs_by_level[level]
        )
        permutation = st.permutation_success_by_level(arm)
        print(
            f"\n== {label} {prompting} {dialect} (E{epoch}) ==  {success_text}  | exec-feedback L4/L5: {feedback_run_count}"
        )
        print(f"   permutation L2-L5 p={tr.fmt_p_plain(permutation.p_value)}")
        gold_runs = gold(dialect)
        for op in ("write", "read"):
            gold_values = tr.sb(gold_runs, op, "throughput_ops_s")
            gold_median = float(np.median(gold_values)) if gold_values else None
            cells = []
            for level in tr.LEVELS[1:]:
                values = tr.sb(runs_by_level[level], op, "throughput_ops_s")
                if values:
                    median = float(np.median(values))
                    delta = ""
                    if gold_median:
                        delta = f" ({(median - gold_median) / gold_median * 100:+.0f}%)"
                    cells.append(f"{level} {median:.1f}{delta} n={len(values)}")
            kruskal = st.kruskal_by_level(arm, op, "proxy")
            spearman = st.spearman_level_vs_throughput(arm, op, "proxy")
            rho = spearman.statistic if spearman.statistic is None else round(spearman.statistic, 2)
            print(
                f"   {op}: "
                + " | ".join(cells)
                + f"   KW p={tr.fmt_p_plain(kruskal.p_value)}  Spearman rho={rho} p={tr.fmt_p_plain(spearman.p_value)}"
            )
