from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GuidanceLevel(StrEnum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"


class PromptingStrategy(StrEnum):
    ZERO_SHOT = "zero_shot"
    MULTI_SHOT = "multi_shot"


STATUS_OK = "ok"
STATUS_GENERATION_FAILED = "generation_failed"
STATUS_VALIDATION_FAILED = "validation_failed"
STATUS_EXECUTION_FAILED = "execution_failed"
STATUS_INFRASTRUCTURE_FAILED = "infrastructure_failed"
STATUS_NOT_MEASURED = "not_measured"


class ContextDocument(BaseModel):
    title: str
    source_path: str
    content: str
    media_type: str = "text/plain"


class GuidanceContext(BaseModel):
    level: GuidanceLevel
    workload_name: str
    workload_prompt: str
    openapi_documents: list[ContextDocument] = Field(default_factory=list)
    use_case_documents: list[ContextDocument] = Field(default_factory=list)
    probe_documents: list[ContextDocument] = Field(default_factory=list)
    feedback_documents: list[ContextDocument] = Field(default_factory=list)
    example_documents: list[ContextDocument] = Field(default_factory=list)
    reference_documents: list[ContextDocument] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    probing_enabled: bool = False
    probe_budget: int | None = None


class ProbeToolCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ConditionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    name: str
    guidance_level: GuidanceLevel
    model: str
    prompting_strategy: PromptingStrategy


class ModelPricing(BaseModel):
    input_usd_per_m: float = Field(ge=0)
    output_usd_per_m: float = Field(ge=0)
    cached_usd_per_m: float = Field(default=0.0, ge=0)
    verified_on: str | None = None


class ModelConfig(BaseModel):
    provider: str
    model_name: str
    model_version: str
    client_adapter: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    pricing: ModelPricing | None = None


class OperationSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = True

    def settings_dict(self) -> dict[str, Any]:
        return {key: value for key, value in self.model_dump().items() if key != "enabled"}


class WorkloadSpec(BaseModel):
    name: str
    description: str
    measurement_window_seconds: int = Field(gt=0)
    operations: dict[str, OperationSpec]


class PromptBuildResult(BaseModel):
    template_name: str
    system_prompt: str
    user_prompt: str
    full_prompt: str


class VertexSettings(BaseModel):
    project: str = Field(min_length=1)
    location: str = Field(default="global", min_length=1)


class LLMRequest(BaseModel):
    condition_name: str
    model_name: str
    prompt: str
    seed: int | None = None


class LLMResponse(BaseModel):
    provider: str
    model_name: str
    model_version: str
    content: str
    finish_reason: str | None = None
    response_id: str | None = None
    usage: dict[str, Any] | None = None


class GeneratedArtifactPayload(BaseModel):
    feature: str = Field(min_length=1)
    steps_py: str = Field(min_length=1)
    summary: str | None = None


class GeneratedArtifacts(BaseModel):
    run_directory: Path
    feature_path: Path
    steps_path: Path
    prompt_path: Path
    response_path: Path
    metadata_path: Path


class GenerationResult(BaseModel):
    condition: ConditionConfig
    prompt: PromptBuildResult
    response: LLMResponse
    payload: GeneratedArtifactPayload
    artifacts: GeneratedArtifacts
    feedback_iterations: int = 0
    status: str = STATUS_OK


class ValidationIssue(BaseModel):
    code: str
    message: str
    path: str | None = None


class ValidationResult(BaseModel):
    syntax_ok: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)


class BehaveRunResult(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    scenarios_passed: int | None = None
    scenarios_failed: int | None = None
    scenarios_error: int | None = None


class RequestSample(BaseModel):
    operation: str | None
    method: str
    path: str
    status_code: int
    latency_ms: float
    error: bool
    started_at: float = 0.0
    ended_at: float = 0.0
    n_points: int = 0
    upstream_path: str | None = None
    body_excerpt: str | None = None
    api: str = "influx"
    rejection: str | None = None


class OperationWindowStats(BaseModel):
    n_requests: int = 0
    error_count: int = 0
    n_points: int = 0
    throughput_ops_s: float = 0.0
    active_duration_s: float = 0.0
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None


class ProxyWindowStats(BaseModel):
    window_duration_s: float
    by_operation: dict[str, OperationWindowStats]
    unmapped_requests: int = 0
    unmapped_paths: list[str] = Field(default_factory=list)
    requests: list[RequestSample] = Field(default_factory=list)


class MetricsSnapshot(BaseModel):
    samples: dict[str, float] = Field(default_factory=dict)


class ServerMetricsDelta(BaseModel):
    by_operation: dict[str, OperationWindowStats]


class SuccessResult(BaseModel):
    executed_ok: bool
    syntax_ok: bool
    semantics_ok: bool

    @property
    def success(self) -> bool:
        return self.executed_ok and self.syntax_ok and self.semantics_ok


class CoverageResult(BaseModel):
    covered_operations: list[str]
    coverage_ratio: float


class ConformanceCheck(BaseModel):
    operation: str
    name: str
    ok: bool
    detail: str = ""


class ConformanceResult(BaseModel):
    ok: bool
    checks: list[ConformanceCheck] = Field(default_factory=list)
    deviations: list[str] = Field(default_factory=list)
    workload_name: str | None = None
    criteria_version: int = 1
    spec_coverage: float | None = None
    spec_checks: list[str] = Field(default_factory=list)


class VocabularyScan(BaseModel):
    hits: dict[str, int] = Field(default_factory=dict)
    total: int = 0


class AdherenceResult(BaseModel):
    api: str
    requests: int = 0
    accepted: int = 0
    rejected: int = 0
    legacy: int = 0
    rejections: list[str] = Field(default_factory=list)
    ok: bool = True


class MetricRow(BaseModel):
    condition_id: int | None = None
    condition_name: str
    guidance_level: str | None = None
    model: str | None = None
    prompting: str | None = None
    repetition: int = 1
    operation: str
    source: str
    executed_ok: bool
    syntax_ok: bool
    semantics_ok: bool
    throughput_ops_s: float
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    latency_p99_ms: float | None = None
    n_requests: int
    error_count: int
    covered: bool
    feedback_iterations: int = 0
    model_version: str | None = None
    seed: int | None = None
    timestamp: str
    run_id: str = ""
    evaluation_id: str = ""
    status: str = STATUS_OK
    n_points: int = 0


class EvaluationResult(BaseModel):
    run_directory: Path
    validation: ValidationResult
    behave: BehaveRunResult | None = None
    success: SuccessResult | None = None
    coverage: CoverageResult | None = None
    proxy_stats: ProxyWindowStats | None = None
    server_delta: ServerMetricsDelta | None = None
    csv_path: Path | None = None
    skipped_execution: bool = False
    skip_reason: str | None = None
    status: str = STATUS_OK
    conformance: ConformanceResult | None = None
    adherence: AdherenceResult | None = None
    vocabulary: VocabularyScan | None = None
    criteria_version: int = 1
