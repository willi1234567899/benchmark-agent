from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from benchmark_agent.errors import GenerationError
from benchmark_agent.execution import SutManager
from benchmark_agent.models import (
    STATUS_GENERATION_FAILED,
    STATUS_VALIDATION_FAILED,
    ConditionConfig,
)
from benchmark_agent.orchestration import EvaluationRunner, GenerationRunner

app = typer.Typer(add_completion=False, help="Benchmark-Agent CLI (Generation + Evaluation).")
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
SEED_REPORT_DIR = PACKAGE_ROOT / "results" / "reports" / "seed"
_FAILED_GENERATION_STATUSES = frozenset({STATUS_GENERATION_FAILED, STATUS_VALIDATION_FAILED})


@app.callback()
def main() -> None:
    pass


def _configuration_error(error: ValidationError) -> typer.Exit:
    typer.echo(f"Configuration error (run from the project root?): {error}", err=True)
    return typer.Exit(code=2)


def _ensure_sut_for_probing(condition_name: str, *, start_sut: bool) -> None:
    from benchmark_agent.config import load_app_config

    app_config = load_app_config()
    condition = app_config.resolve_condition(condition_name)
    level = app_config.resolve_guidance_level(condition.guidance_level.value)
    if not level.probing_enabled and not (level.feedback_enabled and level.execution_feedback):
        return
    sut = SutManager(app_config.paths.project_root, app_config.sut)
    if start_sut:
        sut.up()
        return
    if not sut.is_reachable():
        typer.echo(
            f"L1 probing needs a reachable SUT at {app_config.sut.influxdb_direct_url}; "
            "start it with --start-sut or `benchmark-agent sut up`.",
            err=True,
        )
        raise typer.Exit(code=1)


@app.command(help="Generate benchmark artifacts for one configured condition.")
def generate(
    condition: Annotated[str, typer.Option(help="Condition name, for example L2_gemini_zeroshot.")],
    start_sut: Annotated[
        bool, typer.Option(help="Start InfluxDB via docker compose (needed for L1 probing).")
    ] = False,
) -> None:
    try:
        _ensure_sut_for_probing(condition, start_sut=start_sut)
        runner = GenerationRunner()
        result = runner.run_condition(condition)
    except ValidationError as error:
        raise _configuration_error(error) from error
    except FileNotFoundError as error:
        raise typer.BadParameter(f"Missing required input file: {error}") from error
    except KeyError as error:
        raise typer.BadParameter(str(error.args[0]) if error.args else str(error)) from error
    except GenerationError as error:
        typer.echo(str(error), err=True)
        if error.run_directory is not None:
            typer.echo(f"Run directory: {error.run_directory}", err=True)
        raise typer.Exit(code=1) from error
    except (ValueError, TimeoutError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Condition: {result.condition.name}")
    typer.echo(f"Run directory: {result.artifacts.run_directory}")
    typer.echo(f"Feature file: {result.artifacts.feature_path}")
    typer.echo(f"Steps file: {result.artifacts.steps_path}")


@app.command(
    "llm-check", help="Ping one configured model with the API key from `.env` or the environment."
)
def llm_check(
    model: Annotated[str, typer.Option(help="Model key from config/models.yaml.")] = "gemini",
) -> None:
    from benchmark_agent.config import load_app_config
    from benchmark_agent.llm import create_llm_client
    from benchmark_agent.models import LLMRequest

    try:
        app_config = load_app_config()
        model_config = app_config.resolve_model(model)
    except ValidationError as error:
        raise _configuration_error(error) from error
    except KeyError as error:
        raise typer.BadParameter(str(error.args[0]) if error.args else str(error)) from error
    try:
        client = create_llm_client(model_config, **app_config.llm_client_kwargs)
        response = client.complete(
            LLMRequest(
                condition_name="_llm_check",
                model_name=model_config.model_name,
                prompt="Reply with the single word pong.",
            )
        )
    except (ValueError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error

    backend = getattr(client, "backend", "api_key")
    typer.echo(f"Model: {model_config.model_name}")
    typer.echo(f"Backend: {backend}")
    if backend == "api_key":
        typer.echo("API key: configured")
    elif backend == "claude_cli_subscription":
        typer.echo("Auth: Claude Code CLI subscription login")
    else:
        typer.echo("Auth: Application Default Credentials")
    if response.usage:
        typer.echo(f"Usage: {response.usage}")
    typer.echo(f"Reply: {response.content.strip()[:200]}")


@app.command(help="Validate and execute generated artifacts against the InfluxDB SUT.")
def evaluate(
    run: Annotated[
        Path, typer.Option(exists=True, file_okay=False, help="Artifact run directory.")
    ],
    start_sut: Annotated[
        bool, typer.Option(help="Start a fresh InfluxDB via docker compose.")
    ] = True,
    stop_sut: Annotated[bool, typer.Option(help="Stop InfluxDB after evaluation.")] = False,
    measure: Annotated[
        bool, typer.Option(help="Enable proxy + /metrics measurement window.")
    ] = True,
) -> None:
    from benchmark_agent.orchestration.matrix import read_metadata_status

    if read_metadata_status(run) not in _FAILED_GENERATION_STATUSES:
        for name in ("benchmark.feature", "steps.py"):
            if not (run / name).is_file():
                raise typer.BadParameter(f"{run} does not contain {name}")
    try:
        result = EvaluationRunner().run_artifacts(
            run,
            start_sut=start_sut,
            stop_sut=stop_sut,
            enable_measurement=measure,
        )
    except ValidationError as error:
        raise _configuration_error(error) from error
    except FileNotFoundError as error:
        raise typer.BadParameter(f"Missing required input file: {error}") from error
    except (TimeoutError, RuntimeError, ValueError, OSError) as error:
        typer.echo(f"Evaluation failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(f"Run directory: {result.run_directory}")
    typer.echo(f"Status: {result.status}")
    typer.echo(f"Syntax OK: {result.validation.syntax_ok}")
    if result.skipped_execution:
        typer.echo(f"Execution skipped: {result.skip_reason}")
        raise typer.Exit(code=1)

    assert result.behave is not None
    assert result.success is not None
    typer.echo(f"Behave exit code: {result.behave.exit_code}")
    typer.echo(f"Executed OK: {result.success.executed_ok}")
    typer.echo(f"Semantics OK: {result.success.semantics_ok}")
    if result.coverage is not None:
        typer.echo(
            "Coverage: "
            f"{result.coverage.coverage_ratio:.2f} "
            f"({', '.join(result.coverage.covered_operations) or 'none'})"
        )
    else:
        typer.echo("Coverage: not measured (--no-measure)")
    if result.csv_path is not None:
        typer.echo(f"Metrics CSV: {result.csv_path}")

    if not result.success.success:
        raise typer.Exit(code=1)


@app.command(
    help=("Generate and evaluate the selected conditions N times (single-switch matrix run).")
)
def run(
    condition: Annotated[
        list[str] | None, typer.Option("--condition", help="Condition name (repeatable).")
    ] = None,
    condition_id: Annotated[
        list[int] | None,
        typer.Option(
            "--condition-id", help="Condition id from config/conditions.yaml (repeatable)."
        ),
    ] = None,
    all_conditions: Annotated[
        bool, typer.Option("--all", help="Run every condition in config/conditions.yaml.")
    ] = False,
    expand: Annotated[
        bool,
        typer.Option(
            "--expand",
            help="With --all: run the full level x model x strategy cross product (30).",
        ),
    ] = False,
    repetitions: Annotated[
        int | None, typer.Option(help="Repetitions per condition (default: defaults.repetitions).")
    ] = None,
    first_repetition: Annotated[
        int,
        typer.Option(
            "--first-repetition",
            help=(
                "Number of the first repetition (seed = seed + repetition - 1); continue an "
                "epoch with 6..10 instead of regenerating 1..5."
            ),
            min=1,
        ),
    ] = 1,
    interleave: Annotated[
        bool,
        typer.Option(
            "--interleave",
            help=(
                "Repetition-major order (rep 1 of every condition, then rep 2, ...) so time "
                "drift hits all levels alike; default is condition-major blocks."
            ),
        ),
    ] = False,
    start_sut: Annotated[bool, typer.Option(help="Start InfluxDB (fresh SUT per run).")] = True,
    stop_sut: Annotated[bool, typer.Option(help="Stop InfluxDB after the matrix.")] = False,
    measure: Annotated[
        bool, typer.Option(help="Enable proxy + /metrics measurement window.")
    ] = True,
) -> None:
    from benchmark_agent.orchestration.matrix import (
        MatrixRunner,
        expand_conditions,
        format_outcome_table,
    )

    if expand and not all_conditions:
        raise typer.BadParameter("--expand only makes sense together with --all.")
    try:
        matrix = MatrixRunner()
        app_config = matrix.app_config
        selected: list[ConditionConfig] = []
        if all_conditions:
            selected = (
                expand_conditions(app_config.models.models)
                if expand
                else list(app_config.conditions.conditions)
            )
        for name in condition or []:
            selected.append(app_config.resolve_condition(name))
        for identifier in condition_id or []:
            selected.append(app_config.resolve_condition_by_id(identifier))
    except ValidationError as error:
        raise _configuration_error(error) from error
    except KeyError as error:
        raise typer.BadParameter(str(error.args[0]) if error.args else str(error)) from error

    unique = list({cond.name: cond for cond in selected}.values())
    if not unique:
        raise typer.BadParameter("Select conditions with --condition, --condition-id or --all.")

    try:
        outcomes = matrix.run(
            unique,
            repetitions,
            start_sut=start_sut,
            stop_sut=stop_sut,
            measure=measure,
            first_repetition=first_repetition,
            interleave=interleave,
        )
    except (TimeoutError, RuntimeError, FileNotFoundError, ValueError) as error:
        typer.echo(f"Matrix run failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    report_path = matrix.write_outcomes(outcomes)
    typer.echo(format_outcome_table(outcomes))
    typer.echo(f"Outcomes: {report_path}")
    _refresh_dashboard(app_config)
    failed = [outcome for outcome in outcomes if not outcome.ok]
    typer.echo(f"Runs: {len(outcomes)}, failed: {len(failed)}")
    if failed:
        raise typer.Exit(code=1)


@app.command(
    help=(
        "List the configured conditions (config/conditions.yaml is the smoke subset).\n\n"
        "Ids are the deterministic matrix ids (level-major cross product), so a condition "
        "carries the same id in both listings and in metrics.csv."
    )
)
def conditions(
    expand: Annotated[
        bool,
        typer.Option("--expand", help="Print the full level x model x strategy cross product."),
    ] = False,
) -> None:
    from benchmark_agent.config import load_app_config
    from benchmark_agent.orchestration.matrix import expand_conditions

    try:
        app_config = load_app_config()
    except ValidationError as error:
        raise _configuration_error(error) from error

    if expand:
        listed = expand_conditions(app_config.models.models)
        typer.echo(f"Full matrix ({len(listed)} conditions):")
    else:
        listed = app_config.conditions.conditions
        typer.echo(f"Configured conditions ({len(listed)}, config/conditions.yaml):")
    for condition in listed:
        typer.echo(
            f"{condition.id:>3}  {condition.name:<32} {condition.guidance_level.value}  "
            f"{condition.model:<10} {condition.prompting_strategy.value}"
        )


sut_app = typer.Typer(
    help=(
        "Manage the InfluxDB SUT container. The compose project name is derived from this "
        "checkout, so use these commands instead of calling docker compose by hand."
    ),
)
app.add_typer(sut_app, name="sut")


def _sut_manager() -> tuple[str, str, SutManager]:
    from benchmark_agent.config import load_app_config
    from benchmark_agent.execution.sut import compose_project_name

    app_config = load_app_config()
    manager = SutManager(app_config.paths.project_root, app_config.sut)
    return (
        app_config.sut.influxdb_direct_url,
        compose_project_name(app_config.paths.project_root),
        manager,
    )


def _run_sut_action(action: str) -> None:
    try:
        direct_url, project, manager = _sut_manager()
        getattr(manager, action)()
    except ValidationError as error:
        raise _configuration_error(error) from error
    except (TimeoutError, RuntimeError) as error:
        typer.echo(f"SUT {action} failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"SUT {action}: ok (compose project {project}, direct URL {direct_url})")


@sut_app.command("up", help="Start InfluxDB via docker compose and wait until /health answers.")
def sut_up() -> None:
    _run_sut_action("up")


@sut_app.command("down", help="Stop InfluxDB and drop its data volume.")
def sut_down() -> None:
    _run_sut_action("down")


@sut_app.command("reset", help="Fresh SUT: down (with volume) followed by up.")
def sut_reset() -> None:
    _run_sut_action("reset")


@sut_app.command(
    "status", help="Show the compose project name and whether the direct URL answers /health."
)
def sut_status() -> None:
    try:
        direct_url, project, manager = _sut_manager()
    except ValidationError as error:
        raise _configuration_error(error) from error
    reachable = manager.is_reachable()
    typer.echo(f"Compose project: {project}")
    typer.echo(f"Direct URL: {direct_url}")
    typer.echo(f"Reachable: {reachable}")
    if not reachable:
        raise typer.Exit(code=1)


def _refresh_dashboard(app_config: object) -> None:
    from benchmark_agent.config import AppConfig
    from benchmark_agent.dashboard import build_dashboard

    assert isinstance(app_config, AppConfig)
    try:
        built = build_dashboard(app_config)
    except Exception as error:
        typer.echo(f"Dashboard not updated: {type(error).__name__}: {error}", err=True)
        return
    typer.echo(f"Dashboard: {built.html_path}")


@app.command(
    help=(
        "Build the run dashboard (HTML + tables) from results/raw and results/artifacts.\n\n"
        "The page is self-contained (no CDN) and shows Stage-A quotas, Stage-B throughput "
        "/ latency, coverage, proxy-vs-/metrics cross-check, a run table with drill-down "
        "and the non-parametric test battery from evaluation/statistics.py."
    )
)
def dashboard(
    output: Annotated[
        Path | None,
        typer.Option(help="Output directory (default: results/reports/dashboard)."),
    ] = None,
    serve: Annotated[
        bool, typer.Option("--serve", help="Serve a live view that rebuilds on every request.")
    ] = False,
    host: Annotated[str, typer.Option(help="Bind address for --serve.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port for --serve.")] = 8050,
    refresh: Annotated[
        int, typer.Option(help="Auto-refresh interval in seconds for the served page (0 = off).")
    ] = 30,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the dashboard in the default browser.")
    ] = False,
    seed_demo: Annotated[
        bool,
        typer.Option(
            "--seed-demo",
            help="Render the SYNTHETIC seed matrix instead of results/ (layout preview).",
        ),
    ] = False,
    tables: Annotated[
        bool, typer.Option(help="Also write runs.csv, cells.csv and statistics.json.")
    ] = True,
) -> None:
    import webbrowser

    from benchmark_agent.config import load_app_config
    from benchmark_agent.dashboard import build_dashboard, serve_dashboard
    from benchmark_agent.dashboard.build import render_current_html

    try:
        app_config = load_app_config()
    except ValidationError as error:
        raise _configuration_error(error) from error

    try:
        built = build_dashboard(
            app_config,
            output_dir=output,
            seed_demo=seed_demo,
            write_tables=tables,
        )
    except (OSError, ValueError) as error:
        typer.echo(f"Dashboard build failed: {error}", err=True)
        raise typer.Exit(code=1) from error

    label = " (SYNTHETIC seed data, not measured)" if seed_demo else ""
    typer.echo(f"Dashboard: {built.html_path}{label}")
    typer.echo(f"Runs: {built.n_runs} ({built.n_measured} evaluated)")
    for path in built.table_paths:
        typer.echo(f"  {path}")
    for issue in built.issues:
        typer.echo(f"  data issue: {issue}", err=True)

    if not serve:
        if open_browser:
            webbrowser.open(built.html_path.resolve().as_uri())
        return

    url = f"http://{host}:{port}/"
    typer.echo(f"Serving live dashboard at {url} (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        serve_dashboard(
            lambda: render_current_html(
                app_config, seed_demo=seed_demo, refresh_seconds=refresh or None
            ),
            host=host,
            port=port,
        )
    except OSError as error:
        typer.echo(f"Could not bind {url}: {error}", err=True)
        raise typer.Exit(code=1) from error


@app.command(
    help=(
        "Move harness-caused runs out of the dataset, with a written reason.\n\n"
        "The run directory moves intact to "
        "results/artifacts/_quarantine/<condition>/run-NNN together with QUARANTINE.json "
        "and its former metrics.csv rows (metrics_rows.csv); dashboard and statistics "
        "stop counting it and the run number is never reused. Only for outcomes the "
        "harness caused (parser bug, API quota, seed collision); genuine model failures "
        "are results and stay."
    )
)
def quarantine(
    run: Annotated[
        list[Path],
        typer.Option("--run", help="Artifact run directory <condition>/run-NNN (repeatable)."),
    ],
    reason: Annotated[
        str,
        typer.Option(help="Why the outcome is a harness artefact, not a model result."),
    ],
) -> None:
    from benchmark_agent.config import load_app_config
    from benchmark_agent.orchestration.quarantine import quarantine_run

    try:
        app_config = load_app_config()
    except ValidationError as error:
        raise _configuration_error(error) from error

    csv_path = app_config.paths.results_raw_dir / "metrics.csv"
    for run_directory in run:
        try:
            moved = quarantine_run(
                run_directory,
                artifacts_dir=app_config.paths.results_artifacts_dir,
                csv_path=csv_path,
                reason=reason,
            )
        except (ValueError, OSError) as error:
            typer.echo(f"Quarantine failed for {run_directory}: {error}", err=True)
            raise typer.Exit(code=1) from error
        typer.echo(
            f"Quarantined {moved.run_id} -> {moved.target} "
            f"({moved.rows_removed} metrics.csv rows archived)"
        )
    _refresh_dashboard(app_config)


@app.command(
    help=(
        "Estimated Gemini spend from metadata.json token counts x config/models.yaml "
        "prices.\n\n"
        "Runs since budget.tracked_since carry usage_total (all model turns); earlier "
        "runs are covered by budget.spent_before_usd. The same numbers drive the guard of "
        "`run`."
    )
)
def cost(
    runs: Annotated[
        bool, typer.Option("--runs", help="List every priced run (newest last).")
    ] = False,
    epoch: Annotated[
        int | None, typer.Option(help="Only list runs of this epoch (with --runs).")
    ] = None,
) -> None:
    from benchmark_agent.config import load_app_config
    from benchmark_agent.evaluation.cost import collect_costs, format_cost_report

    try:
        app_config = load_app_config()
    except ValidationError as error:
        raise _configuration_error(error) from error
    report = collect_costs(app_config)
    if runs:
        for record in report.runs:
            if epoch is not None and record.epoch != epoch:
                continue
            price = "unpriced" if record.cost_usd is None else f"{record.cost_usd:8.4f} USD"
            flags = ("" if record.complete else " final-turn-only") + (
                " quarantined" if record.quarantined else ""
            )
            typer.echo(
                f"{record.run_id:<36} epoch {record.epoch!s:>2}  {record.model or '?':<12} "
                f"calls {record.llm_calls!s:>3}  in {record.prompt_tokens:>7}  "
                f"out {record.output_tokens:>6}  {price}{flags}"
            )
    typer.echo(format_cost_report(report))


@app.command(
    help="Model vs. harness failures per matrix (results/raw/matrix_*.json), newest first."
)
def stability(
    required: Annotated[
        int, typer.Option(help="Consecutive clean matrices required for 'STABLE'.")
    ] = 2,
) -> None:
    from benchmark_agent.config import load_app_config
    from benchmark_agent.orchestration.stability import format_stability, load_stability

    try:
        app_config = load_app_config()
    except ValidationError as error:
        raise _configuration_error(error) from error
    reports = load_stability(app_config.paths.results_raw_dir)
    typer.echo(format_stability(reports, required=required))


@app.command(
    "plot-seed",
    help=(
        "Render the SYNTHETIC seed statistics set (matplotlib).\n\n"
        "Synthetic seed data, not measured: a layout prototype for the real metrics.csv."
    ),
)
def plot_seed(
    output: Annotated[
        Path | None, typer.Option(help=f"Output directory (default: {SEED_REPORT_DIR}).")
    ] = None,
) -> None:
    from benchmark_agent.evaluation.seed_report import render_all

    target = output or SEED_REPORT_DIR
    figures = render_all(target)
    typer.echo(f"Wrote {len(figures)} figures to {target} (synthetic seed data, not measured)")
    for path in figures:
        typer.echo(f"  {path}")


if __name__ == "__main__":
    app()
