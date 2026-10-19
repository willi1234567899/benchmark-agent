from __future__ import annotations

import argparse
import json
import shutil

from benchmark_agent.config import load_app_config
from benchmark_agent.orchestration.evaluation_runner import EvaluationRunner
from benchmark_agent.orchestration.workspace import WorkspaceManager


def main() -> None:
    args = _parse_args()
    app_config = load_app_config()
    if app_config.sut.api != args.dialect:
        raise SystemExit(
            f"config/default.yaml has sut.api={app_config.sut.api!r}, but --dialect="
            f"{args.dialect!r} was requested. Switch guidance.yaml/default.yaml to the "
            f"{args.dialect} dialect first (see README 'Epochen 4-5' dialect switch), same as "
            "for a real matrix run."
        )

    project_root = app_config.paths.project_root
    fixture_dir = project_root / "tests" / "fixtures" / f"gold_{args.dialect}"
    feature_src = fixture_dir / "benchmark.feature"
    steps_src = fixture_dir / "steps.py"
    metadata_src = fixture_dir / "metadata.json"
    for path in (feature_src, steps_src, metadata_src):
        if not path.is_file():
            raise SystemExit(f"Missing gold fixture file: {path}")

    condition_name = f"_gold_{args.dialect}"
    base_metadata = json.loads(metadata_src.read_text(encoding="utf-8-sig"))
    if base_metadata.get("condition", {}).get("name") != condition_name:
        raise SystemExit(
            f"{metadata_src} condition.name != {condition_name!r}; refusing to mislabel runs."
        )
    base_metadata["condition"]["guidance_level"] = None
    base_metadata["epoch"] = app_config.defaults.epoch

    workspace = WorkspaceManager(app_config.paths.results_artifacts_dir)
    evaluator = EvaluationRunner(project_root)

    print(
        f"Gold ceiling: {args.dialect}, epoch {app_config.defaults.epoch}, {args.repetitions} reps"
    )
    for rep in range(1, args.repetitions + 1):
        run_directory = workspace.create_run_directory(condition_name)
        shutil.copyfile(feature_src, run_directory / "benchmark.feature")
        shutil.copyfile(steps_src, run_directory / "steps.py")
        (run_directory / "metadata.json").write_text(
            json.dumps(base_metadata, indent=2), encoding="utf-8"
        )

        result = evaluator.run_artifacts(
            run_directory,
            start_sut=True,
            stop_sut=(args.stop_sut_after and rep == args.repetitions),
            enable_measurement=True,
        )
        success = "-" if result.success is None else str(result.success.success)
        throughput = "-"
        if result.proxy_stats is not None:
            write_stats = result.proxy_stats.by_operation.get("write")
            if write_stats is not None:
                throughput = f"{write_stats.throughput_ops_s:.1f} ops/s (write)"
        print(
            f"  rep {rep:>2}/{args.repetitions}  {run_directory.parent.name}/{run_directory.name}  "
            f"status={result.status}  success={success}  {throughput}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the hand-written gold reference through the measurement core."
    )
    parser.add_argument("--dialect", required=True, choices=("tessera", "kestrel"))
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument(
        "--stop-sut-after",
        action="store_true",
        help="Stop the SUT after the last repetition (default: leave it running).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
