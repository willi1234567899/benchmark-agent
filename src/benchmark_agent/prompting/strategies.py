from __future__ import annotations

import json
from pathlib import Path

from benchmark_agent.guidance.common import format_document_sections
from benchmark_agent.models import ConditionConfig, GeneratedArtifactPayload, GuidanceContext

_PROBE_PROTOCOL = """
You may probe the reachable system before writing artifacts.
The harness pins the SUT base URL; you cannot choose another host.
{auth_note}

Return exactly one JSON object per turn:

1. A tool call:
{{"tool": "sut_health", "args": {{}}}}
or
{{"tool": "http_request",
 "args": {{"method": "GET", "path": "/health", "headers": {{}}, "body": null}}}}

2. Or the final artifacts that validate against the schema (feature + steps_py).

Available tools:
- sut_health: GET {health_path} on the configured SUT
- http_request: HTTP request on the configured SUT (method, path, optional headers, optional body).
  Allowed methods: GET, HEAD, and POST; {post_note} Mutating requests are rejected (they still
  consume budget).

Probe budget: {budget} tool calls in total (unknown tools and rejected requests also count).
After each tool result you are told how many calls remain. Once the budget is exhausted you get
exactly one final turn, which must contain the artifact JSON.

Stay within the probe budget. After probing, return the artifact JSON.
""".strip()

_PROBE_VARIANTS: dict[str, dict[str, str]] = {
    "influx": {
        "auth_note": (
            "Every probe request is sent with a valid `Authorization: Token ...` header added by "
            "the harness, so you do not need to supply credentials. The organisation is named "
            "`{org}` and the bucket is named `{bucket}`; both exist already."
        ),
        "health_path": "/health",
        "post_note": "POST is accepted only for the read-only query and sign-in endpoints.",
    },
    "tessera": {
        "auth_note": (
            "Every probe request is sent with a valid `X-Tessera-Key` header added by the "
            "harness, so you do not need to supply credentials. The space is named `{org}` and "
            "the store is named `{bucket}`; both exist already."
        ),
        "health_path": "/tessera/v1/health",
        "post_note": "POST is accepted only for the read-only fetch endpoint.",
    },
    "kestrel": {
        "auth_note": (
            "Every probe request is sent with a valid `X-Api-Token` header added by the "
            "harness, so you do not need to supply credentials. The tenant is named `{org}` and "
            "the store is named `{bucket}`; both exist already."
        ),
        "health_path": "/v1/status",
        "post_note": "POST is not needed for probing (reads are GET requests) and is rejected.",
    },
}


def build_zero_shot_prompt(
    templates_dir: Path,
    condition: ConditionConfig,
    context: GuidanceContext,
    *,
    sut_org: str = "benchmark",
    sut_bucket: str = "benchmark",
    sut_api: str = "influx",
    sut_name: str = "InfluxDB v2",
) -> tuple[str, str]:
    return _render_prompt(
        templates_dir=templates_dir,
        template_name="zero_shot_v1.md",
        condition=condition,
        context=context,
        include_examples=False,
        sut_org=sut_org,
        sut_bucket=sut_bucket,
        sut_api=sut_api,
        sut_name=sut_name,
    )


def build_multi_shot_prompt(
    templates_dir: Path,
    condition: ConditionConfig,
    context: GuidanceContext,
    *,
    sut_org: str = "benchmark",
    sut_bucket: str = "benchmark",
    sut_api: str = "influx",
    sut_name: str = "InfluxDB v2",
) -> tuple[str, str]:
    return _render_prompt(
        templates_dir=templates_dir,
        template_name="multi_shot_v1.md",
        condition=condition,
        context=context,
        include_examples=True,
        sut_org=sut_org,
        sut_bucket=sut_bucket,
        sut_api=sut_api,
        sut_name=sut_name,
    )


def _render_prompt(
    *,
    templates_dir: Path,
    template_name: str,
    condition: ConditionConfig,
    context: GuidanceContext,
    include_examples: bool,
    sut_org: str,
    sut_bucket: str,
    sut_api: str = "influx",
    sut_name: str = "InfluxDB v2",
) -> tuple[str, str]:
    template = (templates_dir / template_name).read_text(encoding="utf-8")
    schema_json = json.dumps(GeneratedArtifactPayload.model_json_schema(), indent=2)
    extra_sections = _extra_sections(context, include_examples=include_examples)
    user_prompt = template.format(
        sut_name=sut_name,
        output_contract=_output_contract(
            context, sut_org=sut_org, sut_bucket=sut_bucket, sut_api=sut_api
        ),
        probe_requirement=_probe_requirement(context),
        schema_json=schema_json,
        condition_name=condition.name,
        guidance_level=condition.guidance_level.value,
        prompting_strategy=condition.prompting_strategy.value,
        workload_name=context.workload_name,
        workload_prompt=context.workload_prompt,
        extra_sections=extra_sections,
    ).strip()
    return template_name, user_prompt


def _output_contract(
    context: GuidanceContext, *, sut_org: str, sut_bucket: str, sut_api: str = "influx"
) -> str:
    if context.probing_enabled:
        budget = (
            "a limited number of" if context.probe_budget is None else str(context.probe_budget)
        )
        variant = _PROBE_VARIANTS.get(sut_api, _PROBE_VARIANTS["influx"])
        protocol = _PROBE_PROTOCOL.format(
            budget=budget,
            auth_note=variant["auth_note"].format(org=sut_org, bucket=sut_bucket),
            health_path=variant["health_path"],
            post_note=variant["post_note"],
        )
        return (
            f"{protocol}\n\n"
            "After probing, the final artifact JSON must validate against this Pydantic schema:"
        )
    return "Return exactly one JSON object that validates against this Pydantic schema:"


def _probe_requirement(context: GuidanceContext) -> str:
    if not context.probing_enabled:
        return ""
    return "- You may first return tool-call JSON as specified in the probe protocol above.\n"


def _extra_sections(context: GuidanceContext, *, include_examples: bool) -> str:
    parts: list[str] = []
    openapi = format_document_sections(context.openapi_documents)
    if openapi:
        parts.append(f"API specification (OpenAPI):\n{openapi}")
    use_cases = format_document_sections(context.use_case_documents)
    if use_cases:
        parts.append(f"Use-case documents:\n{use_cases}")
    if include_examples:
        examples = format_document_sections(context.example_documents)
        if examples:
            parts.append("Multi-shot examples (different domain, contamination-safe):\n" + examples)
    reference = format_document_sections(context.reference_documents)
    if reference:
        parts.append(
            "Manually written reference (ceiling input, not the measurement harness):\n" + reference
        )
    if not parts:
        return ""
    return "\n\n" + "\n\n".join(parts)
