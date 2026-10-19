from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from benchmark_agent.config import AppConfig, load_app_config
from benchmark_agent.evaluation import evaluate_coverage, evaluate_semantics, evaluate_success
from benchmark_agent.evaluation.adherence import evaluate_adherence
from benchmark_agent.evaluation.conformance import CRITERIA_VERSION, evaluate_conformance
from benchmark_agent.evaluation.contamination import scan_artifacts
from benchmark_agent.evaluation.coverage import REQUIRED_OPERATIONS
from benchmark_agent.execution import BehaveRunner, SutManager, validate_artifacts
from benchmark_agent.execution.behave_runner import sanitized_environment
from benchmark_agent.execution.sut import sut_environment
from benchmark_agent.measurement import MeasurementProxy, ServerMetricsCollector
from benchmark_agent.models import (
    STATUS_EXECUTION_FAILED,
    STATUS_GENERATION_FAILED,
    STATUS_INFRASTRUCTURE_FAILED,
    STATUS_NOT_MEASURED,
    STATUS_OK,
    STATUS_VALIDATION_FAILED,
    BehaveRunResult,
    CoverageResult,
    EvaluationResult,
    MetricRow,
    MetricsSnapshot,
    OperationWindowStats,
    ProxyWindowStats,
    ServerMetricsDelta,
    SuccessResult,
    ValidationIssue,
    ValidationResult,
)
from benchmark_agent.results_io import replace_metric_rows, utc_timestamp

_EVALUATION_DIR_NAME = "evaluation"
SOURCE_NONE = "none"


class EvaluationRunner:
    def __init__(self, project_root: Path | None = None) -> None:
        self._app_config: AppConfig = load_app_config(project_root)
        self._project_root = self._app_config.paths.project_root

    def run_artifacts(
        self,
        run_directory: Path,
        *,
        start_sut: bool = True,
        stop_sut: bool = False,
        enable_measurement: bool = True,
    ) -> EvaluationResult:
        run_directory = run_directory.resolve()
        feature_path = run_directory / "benchmark.feature"
        steps_path = run_directory / "steps.py"
        evaluation_dir = _fresh_evaluation_dir(run_directory)
        csv_path = self._app_config.paths.results_raw_dir / "metrics.csv"

        metadata = _load_metadata(run_directory / "metadata.json")
        seed = _as_int(metadata.get("seed"))
        run_id = _run_id(run_directory)
        evaluation_id = utc_timestamp()

        if metadata.get("status") == STATUS_GENERATION_FAILED:
            validation = ValidationResult(
                syntax_ok=False,
                issues=[
                    ValidationIssue(
                        code="GENERATION_FAILED",
                        message=str(metadata.get("error") or "generation failed"),
                        path=str(run_directory),
                    )
                ],
            )
            return self._finish_without_execution(
                run_directory=run_directory,
                evaluation_dir=evaluation_dir,
                csv_path=csv_path,
                metadata=metadata,
                validation=validation,
                status=STATUS_GENERATION_FAILED,
                skip_reason="generation failed",
                run_id=run_id,
                evaluation_id=evaluation_id,
                seed=seed,
            )

        if not enable_measurement and self._app_config.sut.api != "influx":
            raise ValueError(
                f"sut.api={self._app_config.sut.api} needs the measurement proxy (it is the API "
                "facade); evaluate with measurement enabled"
            )
        influx_url = (
            self._app_config.sut.influxdb_url
            if enable_measurement
            else self._app_config.sut.influxdb_direct_url
        )
        sut_env = sut_environment(self._app_config.sut, influx_url=influx_url)
        validation = validate_artifacts(feature_path, steps_path, env=sut_env)
        (evaluation_dir / "validation.json").write_text(
            validation.model_dump_json(indent=2),
            encoding="utf-8",
        )
        if not validation.syntax_ok:
            return self._finish_without_execution(
                run_directory=run_directory,
                evaluation_dir=evaluation_dir,
                csv_path=csv_path,
                metadata=metadata,
                validation=validation,
                status=STATUS_VALIDATION_FAILED,
                skip_reason="validation failed",
                run_id=run_id,
                evaluation_id=evaluation_id,
                seed=seed,
            )

        sut = SutManager(self._project_root, self._app_config.sut)
        proxy: MeasurementProxy | None = None
        behave_result = None
        proxy_stats: ProxyWindowStats | None = None
        server_delta: ServerMetricsDelta | None = None

        try:
            if start_sut:
                sut.reset()
                sut.seed()
            elif sut.seed_files() and sut.is_reachable():
                sut.seed()

            if enable_measurement:
                proxy = self._create_proxy()
                proxy.start()

            env = {**sanitized_environment(os.environ), **sut_env}

            metrics_collector = ServerMetricsCollector()
            metrics_url = (
                f"{self._app_config.sut.influxdb_direct_url.rstrip('/')}"
                f"{self._app_config.measurement.metrics_path}"
            )
            start_snapshot = None
            if proxy is not None:
                start_snapshot = _try_metrics_snapshot(
                    metrics_collector, metrics_url, evaluation_dir
                )
                proxy.open_window()

            behave_result = BehaveRunner(self._app_config.execution.behave_timeout_seconds).run(
                feature_path, steps_path, env=env
            )

            if proxy is not None:
                proxy_stats = proxy.close_window()
                if start_snapshot is not None:
                    end_snapshot = _try_metrics_snapshot(
                        metrics_collector, metrics_url, evaluation_dir
                    )
                    if end_snapshot is not None:
                        server_delta = metrics_collector.delta(start_snapshot, end_snapshot)
                        server_delta = _normalize_server_throughput(server_delta, proxy_stats)

            _write_behave_outputs(evaluation_dir, behave_result)
        except (RuntimeError, TimeoutError, OSError) as error:
            self._finish_without_execution(
                run_directory=run_directory,
                evaluation_dir=evaluation_dir,
                csv_path=csv_path,
                metadata=metadata,
                validation=validation,
                status=STATUS_INFRASTRUCTURE_FAILED,
                skip_reason=f"infrastructure failure: {error}",
                run_id=run_id,
                evaluation_id=evaluation_id,
                seed=seed,
            )
            raise
        finally:
            if proxy is not None:
                proxy.stop()
            if stop_sut:
                try:
                    sut.down()
                except Exception as teardown_error:
                    (evaluation_dir / "sut_teardown_error.txt").write_text(
                        str(teardown_error),
                        encoding="utf-8",
                    )

        conformance = evaluate_conformance(self._app_config.workload, proxy_stats)
        adherence = evaluate_adherence(proxy_stats, api=self._app_config.sut.api)
        vocabulary = scan_artifacts(feature_path, steps_path)
        semantics_ok = evaluate_semantics(proxy_stats, behave_result, conformance=conformance)
        success = evaluate_success(validation, behave_result, semantics_ok=semantics_ok)
        coverage: CoverageResult | None = None
        if proxy_stats is not None:
            coverage = evaluate_coverage(proxy_stats.by_operation, executed_ok=success.executed_ok)

        if proxy_stats is None:
            status = STATUS_NOT_MEASURED
        elif not success.executed_ok:
            status = STATUS_EXECUTION_FAILED
        else:
            status = STATUS_OK

        rows = _build_metric_rows(
            metadata=metadata,
            run_directory=run_directory,
            success=success,
            coverage=coverage,
            proxy_stats=proxy_stats,
            server_delta=server_delta,
            seed=seed,
            run_id=run_id,
            evaluation_id=evaluation_id,
            status=status,
        )
        replace_metric_rows(csv_path, run_id, rows)

        result = EvaluationResult(
            run_directory=run_directory,
            validation=validation,
            behave=behave_result,
            success=success,
            coverage=coverage,
            proxy_stats=proxy_stats,
            server_delta=server_delta,
            csv_path=csv_path,
            skip_reason=None if proxy_stats is not None else "semantics not measured",
            status=status,
            conformance=conformance,
            adherence=adherence,
            vocabulary=vocabulary,
        )
        _write_evaluation_summary(evaluation_dir, result)
        return result

    def _create_proxy(self) -> MeasurementProxy:
        return MeasurementProxy(
            listen_host=self._app_config.measurement.proxy_listen_host,
            listen_port=self._app_config.measurement.proxy_listen_port,
            upstream_url=self._app_config.sut.influxdb_direct_url,
            upstream_timeout_s=self._app_config.execution.behave_timeout_seconds + 5,
            api=self._app_config.sut.api,
            org=self._app_config.sut.org,
            bucket=self._app_config.sut.bucket,
        )

    def _finish_without_execution(
        self,
        *,
        run_directory: Path,
        evaluation_dir: Path,
        csv_path: Path,
        metadata: dict[str, object],
        validation: ValidationResult,
        status: str,
        skip_reason: str,
        run_id: str,
        evaluation_id: str,
        seed: int | None,
    ) -> EvaluationResult:
        success = evaluate_success(validation, None, semantics_ok=False)
        rows = _build_failure_rows(
            metadata=metadata,
            run_directory=run_directory,
            success=success,
            seed=seed,
            run_id=run_id,
            evaluation_id=evaluation_id,
            status=status,
        )
        replace_metric_rows(csv_path, run_id, rows)
        result = EvaluationResult(
            run_directory=run_directory,
            validation=validation,
            success=success,
            csv_path=csv_path,
            skipped_execution=True,
            skip_reason=skip_reason,
            status=status,
        )
        _write_evaluation_summary(evaluation_dir, result)
        return result


def _fresh_evaluation_dir(run_directory: Path) -> Path:
    evaluation_dir = run_directory / _EVALUATION_DIR_NAME
    if evaluation_dir.is_dir():
        shutil.rmtree(evaluation_dir, ignore_errors=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    return evaluation_dir


def _run_id(run_directory: Path) -> str:
    return f"{run_directory.parent.name}/{run_directory.name}"


def _try_metrics_snapshot(
    collector: ServerMetricsCollector,
    metrics_url: str,
    evaluation_dir: Path,
) -> MetricsSnapshot | None:
    try:
        return collector.snapshot(metrics_url)
    except Exception as error:
        (evaluation_dir / "metrics_error.txt").write_text(str(error), encoding="utf-8")
        return None


def _normalize_server_throughput(
    server_delta: ServerMetricsDelta,
    proxy_stats: ProxyWindowStats,
) -> ServerMetricsDelta:
    normalized: dict[str, OperationWindowStats] = {}
    for operation, stats in server_delta.by_operation.items():
        proxy_operation = proxy_stats.by_operation.get(operation)
        if proxy_operation is not None and proxy_operation.active_duration_s > 0:
            duration = proxy_operation.active_duration_s
        else:
            duration = proxy_stats.window_duration_s
        duration = max(duration, 1e-9)
        normalized[operation] = stats.model_copy(
            update={
                "throughput_ops_s": stats.n_requests / duration,
                "active_duration_s": duration,
            }
        )
    return ServerMetricsDelta(by_operation=normalized)


def _load_metadata(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: metadata.json must contain a JSON object")
    return payload


class _RunIdentity:
    def __init__(self, metadata: dict[str, object], run_directory: Path) -> None:
        raw_condition = metadata.get("condition")
        condition = raw_condition if isinstance(raw_condition, dict) else {}
        raw_model = metadata.get("model")
        model = raw_model if isinstance(raw_model, dict) else {}
        self.condition_id = _as_int(condition.get("id"))
        self.condition_name = str(condition.get("name") or run_directory.parent.name)
        self.guidance_level = _as_optional_str(condition.get("guidance_level"))
        self.model = _as_optional_str(condition.get("model"))
        self.prompting = _as_optional_str(condition.get("prompting_strategy"))
        self.model_version = _as_optional_str(model.get("model_version"))
        self.repetition = _as_int(metadata.get("repetition")) or _repetition_from_name(
            run_directory.name
        )
        self.feedback_iterations = _as_int(metadata.get("feedback_iterations")) or 0


def _repetition_from_name(name: str) -> int:
    suffix = name.removeprefix("run-")
    if name.startswith("run-") and suffix.isdigit():
        return int(suffix)
    return 1


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _as_optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _build_metric_rows(
    *,
    metadata: dict[str, object],
    run_directory: Path,
    success: SuccessResult,
    coverage: CoverageResult | None,
    proxy_stats: ProxyWindowStats | None,
    server_delta: ServerMetricsDelta | None,
    seed: int | None,
    run_id: str,
    evaluation_id: str,
    status: str,
) -> list[MetricRow]:
    identity = _RunIdentity(metadata, run_directory)
    covered_operations = set(coverage.covered_operations) if coverage is not None else set()

    layers: list[tuple[str, dict[str, OperationWindowStats]]] = []
    if proxy_stats is not None:
        layers.append(("proxy", proxy_stats.by_operation))
    if server_delta is not None:
        layers.append(("server", server_delta.by_operation))
    if not layers:
        empty_stats = {operation: OperationWindowStats() for operation in REQUIRED_OPERATIONS}
        layers.append((SOURCE_NONE, empty_stats))

    rows: list[MetricRow] = []
    for source, by_operation in layers:
        for operation, stats in by_operation.items():
            rows.append(
                _row(
                    identity,
                    success,
                    operation=operation,
                    source=source,
                    stats=stats,
                    covered=operation in covered_operations,
                    seed=seed,
                    timestamp=evaluation_id,
                    run_id=run_id,
                    evaluation_id=evaluation_id,
                    status=status,
                )
            )
    return rows


def _build_failure_rows(
    *,
    metadata: dict[str, object],
    run_directory: Path,
    success: SuccessResult,
    seed: int | None,
    run_id: str,
    evaluation_id: str,
    status: str,
) -> list[MetricRow]:
    identity = _RunIdentity(metadata, run_directory)
    return [
        _row(
            identity,
            success,
            operation=operation,
            source=SOURCE_NONE,
            stats=OperationWindowStats(),
            covered=False,
            seed=seed,
            timestamp=evaluation_id,
            run_id=run_id,
            evaluation_id=evaluation_id,
            status=status,
        )
        for operation in REQUIRED_OPERATIONS
    ]


def _row(
    identity: _RunIdentity,
    success: SuccessResult,
    *,
    operation: str,
    source: str,
    stats: OperationWindowStats,
    covered: bool,
    seed: int | None,
    timestamp: str,
    run_id: str,
    evaluation_id: str,
    status: str,
) -> MetricRow:
    return MetricRow(
        condition_id=identity.condition_id,
        condition_name=identity.condition_name,
        guidance_level=identity.guidance_level,
        model=identity.model,
        prompting=identity.prompting,
        repetition=identity.repetition,
        operation=operation,
        source=source,
        executed_ok=success.executed_ok,
        syntax_ok=success.syntax_ok,
        semantics_ok=success.semantics_ok,
        throughput_ops_s=stats.throughput_ops_s,
        latency_p50_ms=stats.latency_p50_ms,
        latency_p95_ms=stats.latency_p95_ms,
        latency_p99_ms=stats.latency_p99_ms,
        n_requests=stats.n_requests,
        error_count=stats.error_count,
        covered=covered,
        feedback_iterations=identity.feedback_iterations,
        model_version=identity.model_version,
        seed=seed,
        timestamp=timestamp,
        run_id=run_id,
        evaluation_id=evaluation_id,
        status=status,
        n_points=stats.n_points,
    )


def _write_behave_outputs(evaluation_dir: Path, behave_result: BehaveRunResult) -> None:
    (evaluation_dir / "behave.json").write_text(
        behave_result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (evaluation_dir / "behave.stdout.txt").write_text(
        behave_result.stdout,
        encoding="utf-8",
    )
    (evaluation_dir / "behave.stderr.txt").write_text(
        behave_result.stderr,
        encoding="utf-8",
    )


def _write_evaluation_summary(evaluation_dir: Path, result: EvaluationResult) -> None:
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    result.criteria_version = CRITERIA_VERSION
    (evaluation_dir / "summary.json").write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
