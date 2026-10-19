from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from benchmark_agent.models import ConditionConfig, ModelConfig, VertexSettings, WorkloadSpec


class RawPathsConfig(BaseModel):
    config_dir: str
    inputs_dir: str
    openapi_dir: str
    workload_dir: str
    templates_dir: str
    results_artifacts_dir: str
    results_raw_dir: str = "results/raw"
    results_reports_dir: str = "results/reports"


class FeedbackSettings(BaseModel):
    max_iterations: int = Field(ge=0)


class SutSettings(BaseModel):
    compose_file: str = "sut/docker-compose.yaml"
    influxdb_url: str = "http://127.0.0.1:8086"
    influxdb_direct_url: str = "http://127.0.0.1:8087"
    org: str = "benchmark"
    bucket: str = "benchmark"
    username: str = "benchmark"
    password: str = "benchmarkpassword"
    token: str = "benchmark-token-change-me"
    ready_timeout_seconds: int = Field(default=90, gt=0)
    api: Literal["influx", "tessera", "kestrel"] = "influx"
    display_name: str = "InfluxDB v2"


class ExecutionSettings(BaseModel):
    behave_timeout_seconds: int = Field(default=120, gt=0)


class MeasurementSettings(BaseModel):
    proxy_listen_host: str = "127.0.0.1"
    proxy_listen_port: int = Field(default=8086, gt=0)
    metrics_path: str = "/metrics"


class ProbingSettings(BaseModel):
    max_calls: int = Field(default=8, ge=1)
    timeout_seconds: float = Field(default=5.0, gt=0)
    max_response_chars: int = Field(default=4000, ge=100)


class BudgetSettings(BaseModel):
    limit_usd: float = Field(default=0.0, ge=0)
    stop_at_usd: float | None = Field(default=None, ge=0)
    spent_before_usd: float = Field(default=0.0, ge=0)
    tracked_since: str | None = None
    enabled: bool = False


class DefaultSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BENCHMARK_AGENT_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file="config/default.yaml",
        extra="ignore",
    )

    seed: int
    epoch: int = Field(default=1, ge=1)
    seed_per_repetition: bool = False
    repetitions: int = Field(gt=0)
    feedback: FeedbackSettings
    paths: RawPathsConfig
    sut: SutSettings = Field(default_factory=SutSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    measurement: MeasurementSettings = Field(default_factory=MeasurementSettings)
    probing: ProbingSettings = Field(default_factory=ProbingSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    gemini_api_key: str | None = Field(default=None, repr=False)
    gemini_backend: Literal["api_key", "vertex"] = "api_key"
    vertex_project: str | None = None
    vertex_location: str = "global"
    openai_api_key: str | None = Field(default=None, repr=False)
    deepseek_api_key: str | None = Field(default=None, repr=False)
    openrouter_api_key: str | None = Field(default=None, repr=False)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


class GuidanceLevelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_workload: bool
    workload_file: str | None = None
    openapi_files: list[str]
    use_case_files: list[str]
    feedback_mode: Literal["auto", "scripted", "hybrid"] | None = None
    feedback_files: list[str] = Field(default_factory=list)
    feedback_enabled: bool
    reference_files: list[str] = Field(default_factory=list)
    reference_enabled: bool
    probing_enabled: bool
    execution_feedback: bool = False


class GuidanceFile(BaseModel):
    levels: dict[str, GuidanceLevelConfig]


class ModelsFile(BaseModel):
    models: dict[str, ModelConfig]


class ConditionsFile(BaseModel):
    conditions: list[ConditionConfig]


class ResolvedPaths(BaseModel):
    project_root: Path
    config_dir: Path
    inputs_dir: Path
    openapi_dir: Path
    workload_dir: Path
    templates_dir: Path
    results_artifacts_dir: Path
    results_raw_dir: Path
    results_reports_dir: Path


class AppConfig(BaseModel):
    defaults: DefaultSettings
    paths: ResolvedPaths
    guidance: GuidanceFile
    models: ModelsFile
    conditions: ConditionsFile
    workload: WorkloadSpec

    @property
    def sut(self) -> SutSettings:
        return self.defaults.sut

    @property
    def execution(self) -> ExecutionSettings:
        return self.defaults.execution

    @property
    def measurement(self) -> MeasurementSettings:
        return self.defaults.measurement

    @property
    def probing(self) -> ProbingSettings:
        return self.defaults.probing

    @model_validator(mode="after")
    def _check_condition_references(self) -> AppConfig:
        seen_ids: set[int] = set()
        seen_names: set[str] = set()
        for condition in self.conditions.conditions:
            if condition.id in seen_ids:
                raise ValueError(
                    f"Duplicate condition id {condition.id} in conditions.yaml "
                    f"(condition {condition.name!r})"
                )
            if condition.name in seen_names:
                raise ValueError(f"Duplicate condition name {condition.name!r} in conditions.yaml")
            seen_ids.add(condition.id)
            seen_names.add(condition.name)
            if condition.model not in self.models.models:
                raise ValueError(
                    f"Condition {condition.name!r} references unknown model "
                    f"{condition.model!r}; known models: {sorted(self.models.models)}"
                )
            if condition.guidance_level.value not in self.guidance.levels:
                raise ValueError(
                    f"Condition {condition.name!r} references unknown guidance level "
                    f"{condition.guidance_level.value!r}; known levels: "
                    f"{sorted(self.guidance.levels)}"
                )
        self._check_conditions_against_matrix()
        return self

    def _check_conditions_against_matrix(self) -> None:
        from benchmark_agent.conditions import matrix_by_name

        matrix = matrix_by_name(self.models.models)
        for condition in self.conditions.conditions:
            expected = matrix.get(condition.name)
            if expected is None:
                raise ValueError(
                    f"Condition {condition.name!r} is not part of the level x model x strategy "
                    "matrix (see `benchmark-agent conditions --expand`)"
                )
            if expected != condition:
                raise ValueError(
                    f"Condition {condition.name!r} must match its matrix entry "
                    f"(id {expected.id}, {expected.guidance_level.value}, {expected.model}, "
                    f"{expected.prompting_strategy.value}); got id {condition.id}, "
                    f"{condition.guidance_level.value}, {condition.model}, "
                    f"{condition.prompting_strategy.value}"
                )

    @model_validator(mode="after")
    def _check_proxy_wiring(self) -> AppConfig:
        proxy = f"{self.measurement.proxy_listen_host}:{self.measurement.proxy_listen_port}"
        if urlsplit(self.sut.influxdb_url).netloc != proxy:
            raise ValueError(
                f"sut.influxdb_url ({self.sut.influxdb_url}) must point at the measurement "
                f"proxy address {proxy} (measurement.proxy_listen_host/port)"
            )
        if urlsplit(self.sut.influxdb_direct_url).netloc == proxy:
            raise ValueError(
                f"sut.influxdb_direct_url ({self.sut.influxdb_direct_url}) must not equal the "
                f"measurement proxy address {proxy}"
            )
        return self

    def resolve_condition(self, condition_name: str) -> ConditionConfig:
        for condition in self.conditions.conditions:
            if condition.name == condition_name:
                return condition
        raise KeyError(f"Unknown condition: {condition_name}")

    def resolve_condition_by_id(self, condition_id: int) -> ConditionConfig:
        for condition in self.conditions.conditions:
            if condition.id == condition_id:
                return condition
        raise KeyError(f"Unknown condition id: {condition_id}")

    def resolve_model(self, model_key: str) -> ModelConfig:
        try:
            return self.models.models[model_key]
        except KeyError as error:
            raise KeyError(f"Unknown model: {model_key}") from error

    def resolve_guidance_level(self, level: str) -> GuidanceLevelConfig:
        try:
            return self.guidance.levels[level]
        except KeyError as error:
            raise KeyError(f"Unknown guidance level: {level}") from error

    @property
    def gemini_api_key(self) -> str | None:
        from benchmark_agent.llm.base import resolve_gemini_api_key

        return resolve_gemini_api_key(self.defaults.gemini_api_key)

    @property
    def vertex_settings(self) -> VertexSettings | None:
        if self.defaults.gemini_backend != "vertex":
            return None
        project = self.defaults.vertex_project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise ValueError(
                "BENCHMARK_AGENT_GEMINI_BACKEND=vertex needs BENCHMARK_AGENT_VERTEX_PROJECT "
                "(or GOOGLE_CLOUD_PROJECT) set to the Google Cloud project id."
            )
        return VertexSettings(project=project, location=self.defaults.vertex_location)

    @property
    def llm_client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "gemini_api_key": self.gemini_api_key,
            "provider_api_keys": {
                "openai": self.defaults.openai_api_key,
                "deepseek": self.defaults.deepseek_api_key,
                "openrouter": self.defaults.openrouter_api_key,
            },
        }
        vertex = self.vertex_settings
        if vertex is not None:
            kwargs["vertex"] = vertex
        return kwargs


def load_app_config(project_root: Path | None = None) -> AppConfig:
    root = (project_root or Path.cwd()).resolve()
    defaults = _load_default_settings(root)
    paths = _resolve_paths(root, defaults.paths)
    guidance = _load_yaml_model(paths.config_dir / "guidance.yaml", GuidanceFile)
    models = _load_yaml_model(paths.config_dir / "models.yaml", ModelsFile)
    conditions = _load_yaml_model(paths.config_dir / "conditions.yaml", ConditionsFile)
    workload = _load_yaml_model(paths.config_dir / "workload.yaml", WorkloadSpec)
    return AppConfig(
        defaults=defaults,
        paths=paths,
        guidance=guidance,
        models=models,
        conditions=conditions,
        workload=workload,
    )


def _load_default_settings(root: Path) -> DefaultSettings:
    class _RootedSettings(DefaultSettings):
        model_config = SettingsConfigDict(
            {
                **DefaultSettings.model_config,
                "env_file": str(root / ".env"),
                "yaml_file": str(root / "config" / "default.yaml"),
            }
        )

    load_settings = cast(Callable[[], DefaultSettings], _RootedSettings)
    return load_settings()


def _resolve_paths(project_root: Path, raw_paths: RawPathsConfig) -> ResolvedPaths:
    return ResolvedPaths(
        project_root=project_root,
        config_dir=project_root / raw_paths.config_dir,
        inputs_dir=project_root / raw_paths.inputs_dir,
        openapi_dir=project_root / raw_paths.openapi_dir,
        workload_dir=project_root / raw_paths.workload_dir,
        templates_dir=project_root / raw_paths.templates_dir,
        results_artifacts_dir=project_root / raw_paths.results_artifacts_dir,
        results_raw_dir=project_root / raw_paths.results_raw_dir,
        results_reports_dir=project_root / raw_paths.results_reports_dir,
    )


def _load_yaml_model[ModelT: BaseModel](path: Path, model_type: type[ModelT]) -> ModelT:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TypeAdapter(model_type).validate_python(payload)
