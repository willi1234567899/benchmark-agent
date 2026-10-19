from __future__ import annotations

import difflib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from benchmark_agent.config import load_app_config
from benchmark_agent.llm.base import create_llm_client
from benchmark_agent.models import LLMRequest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = PROJECT_ROOT / "reference" / "features" / "influx_write_benchmark.feature"
SEED_LINES = 8

PROMPTS: dict[str, str] = {
    "A_tessera_known": (
        "What is the 'Tessera time-series store' and how does its HTTP API v1 work? Name its "
        "endpoints and request fields if you know them. If you have never heard of it, say "
        "exactly: I do not know this system."
    ),
    "B_influx_known": (
        "For InfluxDB 2.x (OSS): (1) name the HTTP endpoint and the query parameters used to "
        "write line protocol, (2) name the JSON body fields of the delete endpoint, (3) write the "
        "Flux query that reads the last hour of measurement 'benchmark' limited to 10 rows. "
        "Answer briefly."
    ),
    "D_bddbench_repo": (
        "Do you know the GitHub repository DPS25/bddbench? If yes, describe its purpose, its "
        "structure and quote one of its Gherkin feature file names. If you do not know it, say "
        "exactly: I do not know this repository."
    ),
}


def main() -> int:
    app_config = load_app_config(PROJECT_ROOT)
    model_config = app_config.resolve_model("gemini")
    probe_model = model_config.model_copy(
        update={"parameters": {**model_config.parameters, "temperature": 0}}
    )
    client = create_llm_client(probe_model, **app_config.llm_client_kwargs)

    reference_text = REFERENCE.read_text(encoding="utf-8")
    reference_lines = reference_text.splitlines()
    seed = "\n".join(reference_lines[:SEED_LINES])
    remainder = "\n".join(reference_lines[SEED_LINES:])
    prompts = dict(PROMPTS)
    prompts["C_bddbench_recall"] = (
        "The following is the beginning of a Gherkin feature file from a public GitHub "
        "repository. Continue the file exactly as it appears in that repository (same scenario "
        "outline, same table columns and rows). Output only the continuation.\n\n" + seed
    )

    results: dict[str, dict[str, object]] = {}
    for key in ("A_tessera_known", "B_influx_known", "C_bddbench_recall", "D_bddbench_repo"):
        response = client.complete(
            LLMRequest(
                condition_name="knowledge_probe",
                model_name=probe_model.model_name,
                prompt=prompts[key],
                seed=app_config.defaults.seed,
            )
        )
        entry: dict[str, object] = {
            "prompt": prompts[key],
            "answer": response.content,
            "model_version": response.model_version,
            "usage": response.usage,
        }
        if key == "C_bddbench_recall":
            entry["recall"] = _recall_metrics(response.content, remainder)
        results[key] = entry
        print(f"[{key}] {response.content[:300].replace(chr(10), ' | ')}")

    stamp = datetime.now(UTC)
    out_dir = PROJECT_ROOT / "results" / "reports" / "knowledge_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": stamp.isoformat(timespec="seconds"),
        "model": probe_model.model_dump(mode="json"),
        "backend": getattr(client, "backend", "api_key"),
        "reference_file": REFERENCE.relative_to(PROJECT_ROOT).as_posix(),
        "seed_lines": SEED_LINES,
        "results": results,
    }
    json_path = out_dir / f"{stamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path = json_path.with_suffix(".md")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    print(f"\nwritten: {json_path}\n         {md_path}")
    recall = results["C_bddbench_recall"]["recall"]
    print(f"bddbench recall: {recall}")
    return 0


def _recall_metrics(answer: str, remainder: str) -> dict[str, float | int]:
    original = [line.strip() for line in remainder.splitlines() if line.strip()]
    generated = [line.strip() for line in answer.splitlines() if line.strip()]
    original_set = set(original)
    verbatim = sum(1 for line in generated if line in original_set)
    ratio = difflib.SequenceMatcher(None, "\n".join(generated), "\n".join(original)).ratio()
    return {
        "generated_lines": len(generated),
        "verbatim_lines": verbatim,
        "verbatim_share": round(verbatim / len(generated), 3) if generated else 0.0,
        "sequence_ratio": round(ratio, 3),
        "original_lines": len(original),
    }


def _markdown(payload: dict[str, object]) -> str:
    results = payload["results"]
    assert isinstance(results, dict)
    lines = [
        "# Knowledge probe",
        "",
        f"- timestamp: {payload['timestamp']}",
        f"- backend: {payload['backend']}",
        f"- reference: {payload['reference_file']} (first {payload['seed_lines']} lines as seed)",
        "",
    ]
    for key, entry in results.items():
        assert isinstance(entry, dict)
        lines += [f"## {key}", "", f"model_version: {entry.get('model_version')}", ""]
        if "recall" in entry:
            lines += [f"recall: `{json.dumps(entry['recall'])}`", ""]
        lines += ["Prompt:", "", "```", str(entry["prompt"]), "```", "", "Answer:", "", "```"]
        lines += [str(entry["answer"]), "```", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
