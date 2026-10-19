from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from benchmark_agent.config import load_app_config
from benchmark_agent.evaluation.cost import collect_costs
from benchmark_agent.evaluation.runs import RunRecord, collect_runs
from benchmark_agent.evaluation.statistics import LEVEL_ORDER

CONFIG_FILES = (
    "config/default.yaml",
    "config/guidance.yaml",
    "config/guidance.kestrel.yaml",
    "config/guidance.tessera.yaml",
    "config/models.yaml",
    "config/conditions.yaml",
    "config/workload.yaml",
    "config/workload_v2.yaml",
    "config/workload_v1.yaml",
)


def main() -> None:
    config = load_app_config()
    root = config.paths.project_root
    collection = collect_runs(
        csv_path=config.paths.results_raw_dir / "metrics.csv",
        artifacts_dir=config.paths.results_artifacts_dir,
        project_root=root,
    )
    costs = collect_costs(config)
    cost_by_run = {record.run_id: record for record in costs.runs}

    runs_by_epoch: dict[int, list[RunRecord]] = defaultdict(list)
    for run in collection.runs:
        if run.is_fixture:
            continue
        runs_by_epoch[run.epoch or 1].append(run)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = config.paths.results_reports_dir / "config_snapshots" / stamp
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name in CONFIG_FILES:
        path = root / name
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
            shutil.copy2(path, snapshot_dir / Path(name).name)

    payload: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "runs_total": sum(len(epoch_runs) for epoch_runs in runs_by_epoch.values()),
        "quarantined": len(collection.quarantined),
        "config_hashes": hashes,
        "config_snapshot": str(snapshot_dir.relative_to(root)).replace("\\", "/"),
        "spend_usd_total_estimate": round(costs.total_usd, 2),
        "epochs": [],
    }
    lines = [
        "# Dataset manifest: Benchmark-Agent",
        "",
        f"Generated {payload['generated_at']}; {payload['runs_total']} runs in the dataset, "
        f"{payload['quarantined']} quarantined (excluded). Estimated spend "
        f"{payload['spend_usd_total_estimate']} USD. Config snapshot: "
        f"`{payload['config_snapshot']}`.",
        "",
        "| Epoche | Runs | Modell(e) | API | Kriterien | Erfolg je Stufe (L1…L5) | L2–L5 | "
        "Kosten (USD, gezählt) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for epoch in sorted(runs_by_epoch):
        runs = runs_by_epoch[epoch]
        models = sorted({run.model or "?" for run in runs})
        apis = sorted({run.sut_api or "influx" for run in runs})
        criteria = sorted({run.criteria_version or 1 for run in runs})
        per_level = []
        for level in LEVEL_ORDER:
            members = [run for run in runs if run.guidance_level == level]
            if members:
                per_level.append(
                    f"{level} {sum(1 for run in members if run.success)}/{len(members)}"
                )
        ladder = [run for run in runs if run.guidance_level != "L1"]
        ladder_text = f"{sum(1 for run in ladder if run.success)}/{len(ladder)}" if ladder else "–"
        cost = sum(
            cost_by_run[run.run_id].cost_usd or 0.0
            for run in runs
            if run.run_id in cost_by_run and cost_by_run[run.run_id].cost_usd is not None
        )
        complete = all(
            run.run_id in cost_by_run and cost_by_run[run.run_id].complete for run in runs
        )
        cost_text = f"{cost:.2f}" + ("" if complete else " (Endturn-Usage, Schätzung)")
        payload["epochs"].append(
            {
                "epoch": epoch,
                "runs": len(runs),
                "models": models,
                "apis": apis,
                "criteria_versions": criteria,
                "success_by_level": per_level,
                "ladder_success": ladder_text,
                "cost_usd": round(cost, 3),
                "cost_complete": complete,
                "failure_classes": _class_counts(runs),
            }
        )
        criteria_text = "/".join(str(version) for version in criteria)
        lines.append(
            f"| {epoch} | {len(runs)} | {', '.join(models)} | {', '.join(apis)} | "
            f"v{criteria_text} | {', '.join(per_level)} | {ladder_text} | "
            f"{cost_text} |"
        )
    lines += [
        "",
        "Regeln: Epochen nie poolen (Prompt, Workload, Kriterien oder Feedback-Regime "
        "unterscheiden sich); Quarantäne-Runs sind je Run begründet "
        "(`_quarantine/**/QUARANTINE.json`); "
        "Epochen 1–9 hatten Feedback nur bei Syntaxfehlern, ab Epoche 10 Ausführungs-Feedback.",
        "",
        "Config-Hashes (SHA-256, 16 Zeichen) zum Zeitpunkt des Einfrierens:",
        "",
    ]
    lines += [f"- `{name}`: `{digest}`" for name, digest in hashes.items()]
    out_md = config.paths.results_reports_dir / "dataset_manifest.md"
    out_json = config.paths.results_reports_dir / "dataset_manifest.json"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWritten: {out_md}\n         {out_json}")


def _class_counts(runs: list[RunRecord]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for run in runs:
        if run.failure_class and run.failure_class != "none":
            counts[run.failure_class] += 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


if __name__ == "__main__":
    main()
