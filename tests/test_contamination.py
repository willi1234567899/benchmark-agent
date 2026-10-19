from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from benchmark_agent.config import AppConfig, load_app_config
from benchmark_agent.evaluation.contamination import scan_artifacts, scan_text
from benchmark_agent.guidance import create_guidance_provider
from benchmark_agent.measurement.facade import (
    LocalResponse,
    TesseraFacade,
    UpstreamRequest,
    index_document,
    wire_code,
)
from benchmark_agent.prompting import PromptBuilder

FORBIDDEN = re.compile(
    r"influx|\bflux\b|/api/v2|line[\s_-]?protocol|\bprecision\b|\bbuckets?\b|\borg\b|"
    r"_measurement|\b_field\b|\b_value\b|\b_time\b|\bpredicate\b|authorization: token",
    re.IGNORECASE,
)


def _offending(text: str) -> list[str]:
    return sorted({match.group(0).lower() for match in FORBIDDEN.finditer(text)})


@pytest.fixture
def tessera_config(project_root: Path, monkeypatch: pytest.MonkeyPatch) -> AppConfig:
    monkeypatch.setenv("BENCHMARK_AGENT_SUT__API", "tessera")
    return load_app_config(project_root)


def test_every_condition_prompt_is_free_of_influx_vocabulary(tessera_config: AppConfig) -> None:
    app_config = tessera_config
    builder = PromptBuilder(app_config.paths.templates_dir)
    checked = 0
    for condition in app_config.conditions.conditions:
        context = create_guidance_provider(condition.guidance_level).build_context(
            app_config, condition
        )
        prompt = builder.build(condition, context, app_config=app_config)
        offending = _offending(prompt.full_prompt)
        assert not offending, f"{condition.name}: {offending}"
        checked += 1
    assert checked == 60


def test_facade_wire_surface_is_free_of_influx_vocabulary() -> None:
    facade = TesseraFacade(org="benchmark", bucket="benchmark")
    surfaces = [json.dumps(index_document())]
    probes = [
        ("POST", "/tessera/v1/ingest", {"org": "b", "bucket": "b", "precision": "ns"}, {}, b"x"),
        (
            "POST",
            "/tessera/v1/ingest",
            {"space": "b", "store": "b"},
            {"Authorization": "Token x"},
            b"x",
        ),
        ("POST", "/tessera/v1/fetch", {"space": "b"}, {"X-Tessera-Key": "k"}, b'{"query": "x"}'),
        (
            "POST",
            "/tessera/v1/purge",
            {"space": "b", "store": "b"},
            {"X-Tessera-Key": "k"},
            b'{"start": 1}',
        ),
        ("POST", "/api/v2/write", {"org": "b"}, {}, b"m v=1"),
        ("GET", "/health", {}, {}, b""),
    ]
    for method, path, query, headers, body in probes:
        outcome = facade.handle(method, path, query, headers, body)
        assert isinstance(outcome, LocalResponse), path
        surfaces.append(outcome.body.decode())
    read = UpstreamRequest("read", "POST", "/api/v2/query", {}, b"", series="benchmark")
    csv = ",result,table,_start,_stop,_time,_value,_field,_measurement,host\n"
    csv += ",_result,0,a,b,2026-08-27T18:00:00Z,1.5,value,benchmark,h1\n"
    _, body, _ = facade.translate_response(read, 200, csv.encode())
    surfaces.append(body.decode())
    assert json.loads(body)["rows"][0] == {
        "ts": "2026-08-27T18:00:00Z",
        "value": 1.5,
        "metric": "value",
        "series": "benchmark",
        "host": "h1",
    }
    for surface in surfaces:
        assert not _offending(surface), surface[:200]


def test_wire_codes_hide_the_canary_names() -> None:
    assert wire_code("influx_param_precision") == "unsupported_parameter"
    assert wire_code("influx_field_limit") == "unknown_field"
    assert wire_code("influx_auth_header") == "unsupported_auth_header"
    assert wire_code("legacy_endpoint") == "unknown_endpoint"
    assert wire_code("missing_field_series") == "missing_field_series"


def test_tessera_material_files_are_free_of_influx_vocabulary(project_root: Path) -> None:
    files = [
        project_root / "inputs" / "openapi" / "tessera-v1.yaml",
        project_root / "inputs" / "workload" / "tessera_task.md",
        project_root / "inputs" / "workload" / "tessera_task_l1.md",
        *sorted((project_root / "inputs" / "usecases").glob("tessera_*.md")),
        *sorted((project_root / "reference" / "features").glob("tessera_*.feature")),
        project_root / "inputs" / "openapi" / "kestrel-v1.yaml",
        *sorted((project_root / "inputs" / "workload").glob("kestrel_*.md")),
        *sorted((project_root / "inputs" / "usecases").glob("kestrel_*.md")),
        *sorted((project_root / "reference" / "features").glob("kestrel_*.feature")),
        *sorted((project_root / "tests" / "fixtures" / "gold_kestrel").glob("*.py")),
        *sorted((project_root / "tests" / "fixtures" / "gold_tessera").glob("*.py")),
        *sorted((project_root / "inputs" / "examples").glob("*")),
    ]
    assert len(files) >= 10
    for path in files:
        if path.is_file() and path.suffix in {".yaml", ".md", ".feature", ".py"}:
            assert not _offending(path.read_text(encoding="utf-8")), path.name


def test_vocabulary_scan_counts_influx_terms(tmp_path: Path) -> None:
    feature = tmp_path / "benchmark.feature"
    steps = tmp_path / "steps.py"
    feature.write_text("Feature: fetch rows from series benchmark\n", encoding="utf-8")
    steps.write_text(
        'url = f"{base}/api/v2/write?org={org}&bucket={bucket}&precision=ns"\n'
        'headers = {"Authorization": f"Token {key}"}\n'
        "flux = 'from(bucket: \"b\") |> range(start: -1h)'\n",
        encoding="utf-8",
    )
    scan = scan_artifacts(feature, steps)
    assert scan.hits["/api/v2"] == 1
    assert scan.hits["bucket"] >= 2
    assert scan.hits["precision"] == 1
    assert scan.hits["flux"] >= 1
    assert scan.total == sum(scan.hits.values())
    assert scan_text("ingest records into series benchmark with labels").total == 0


def test_vocabulary_scan_ignores_the_english_word_precision() -> None:
    assert scan_text("compare with floating-point precision issues").total == 0
    assert scan_text('params = {"precision": "ms"}').hits["precision"] == 1
    assert scan_text("url += '&precision=ns'").hits["precision"] == 1
