from __future__ import annotations

import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt
import thesis_results as tr
from matplotlib.patches import Patch
from scipy import stats as sps

from benchmark_agent.evaluation.contamination import scan_text
from benchmark_agent.evaluation.runs import SOURCE_PROXY, RunRecord

OPS: tuple[str, ...] = ("write", "read", "delete")
DIALECTS: tuple[str, ...] = ("Kestrel", "Tessera")
MODEL_ORDER = {m: i for i, m in enumerate(tr.MODELS)}
PROMPT_ORDER = {"zero_shot": 0, "multi_shot": 1}
PROBE_BUDGET = 8

STRONG_LABELS: tuple[str, ...] = (
    "influx",
    "flux",
    "/api/v2",
    "line protocol",
    "precision",
    "bucket",
    "org",
    "predicate",
    "Authorization: Token",
)
STRONG_RUNTIME_SUFFIXES: tuple[str, ...] = (
    "org",
    "orgID",
    "bucket",
    "precision",
    "db",
    "predicate",
    "query",
    "flux",
    "lines",
    "line_protocol",
    "measurement",
    "auth_header",
)
WEAK_TERMS: dict[str, dict[str, tuple[str, ...] | None]] = {
    "start": {"Kestrel": ("read", "delete"), "Tessera": ("read", "delete")},
    "limit": {"Kestrel": None, "Tessera": ("read",)},
    "points": {"Kestrel": None, "Tessera": ("write",)},
}
_RESULT_RE = re.compile(r"^(\w+) (\S+) -> HTTP (\d+)")


def _weak_static_re(term: str) -> re.Pattern[str]:
    return re.compile(rf"""(['"]){term}\1\s*:|[?&]{term}=""")


WEAK_STATIC = {term: _weak_static_re(term) for term in WEAK_TERMS}


def _run_dir(run: RunRecord, project_root: Path) -> Path:
    directory = Path(run.run_directory or "")
    return directory if directory.is_absolute() else project_root / directory


def _read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None


def _first_object_is_tool_call(raw: str) -> bool:
    start = raw.find("{")
    if start < 0:
        return False
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except ValueError:
        return False
    return isinstance(obj, dict) and "tool" in obj


def _operation_outcomes(requests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ops: dict[str, dict[str, Any]] = {}
    for op in OPS:
        op_requests = [request for request in requests if request.get("operation") == op]
        accepted = sum(
            1
            for request in op_requests
            if not request.get("rejection") and (request.get("status_code") or 0) < 400
        )
        codes = Counter(
            str(request.get("rejection") or f"http{request.get('status_code')}")
            for request in op_requests
            if request.get("rejection") or (request.get("status_code") or 0) >= 400
        )
        if not op_requests:
            state = "not_reached"
        elif accepted == len(op_requests):
            state = "accepted"
        elif accepted:
            state = "partial"
        else:
            state = "rejected"
        ops[op] = {"state": state, "n": len(op_requests), "accepted": accepted, "codes": codes}
    return ops


def run_facts(r: RunRecord, project_root: Path) -> dict[str, Any]:
    run_dir = _run_dir(r, project_root)
    dialect = "Kestrel" if (r.sut_api or "").lower() == "kestrel" else "Tessera"
    probes_raw = _read_text(run_dir / "probe_rounds.json")
    probes: list[dict[str, Any]] = json.loads(probes_raw) if probes_raw else []

    index_seen = data_200 = False
    masked = dict_nonempty = dict_any = 0
    paths: list[str] = []
    for probe in probes:
        args = probe.get("args") or {}
        result = probe.get("result") or ""
        match = _RESULT_RE.match(result)
        method = (match.group(1) if match else str(args.get("method") or "GET")).upper()
        path = match.group(2) if match else str(args.get("path") or "")
        status = int(match.group(3)) if match else None
        paths.append(f"{method} {path.split('?')[0]} -> {status if status else result[:24]}")
        if status == 200 and '"endpoints"' in result:
            index_seen = True
        if status == 200 and dialect == "Kestrel" and method == "GET" and "/points" in path:
            data_200 = True
        if status == 200 and dialect == "Tessera" and method == "POST" and "/fetch" in path:
            data_200 = True
        body = args.get("body")
        dict_any += isinstance(body, (dict, list))
        if isinstance(body, (dict, list)) and body:
            dict_nonempty += 1
            if "non_json_body" in result:
                masked += 1

    invalid_turns = rescuable_turns = repairs = 0
    turns_raw = _read_text(run_dir / "turns.jsonl")
    for line in (turns_raw or "").splitlines():
        if not line.strip():
            continue
        turn = json.loads(line)
        if turn.get("json_repair"):
            repairs += 1
        if turn.get("parsed_kind") == "invalid":
            invalid_turns += 1
            rescuable_turns += _first_object_is_tool_call(turn.get("raw_response") or "")

    feature = _read_text(run_dir / "benchmark.feature")
    steps = _read_text(run_dir / "steps.py")
    has_artefact = bool(steps)
    text = (feature or "") + "\n" + (steps or "")
    scan = scan_text(text) if has_artefact else None
    strong_hits = sum(
        count for label, count in (scan.hits if scan else {}).items() if label in STRONG_LABELS
    )
    weak_static = {term: bool(steps and WEAK_STATIC[term].search(steps)) for term in WEAK_TERMS}

    summary_raw = _read_text(run_dir / "evaluation" / "summary.json")
    requests: list[dict[str, Any]] = []
    if summary_raw:
        requests = (json.loads(summary_raw).get("proxy_stats") or {}).get("requests") or []
    ops = _operation_outcomes(requests)
    rejection_codes = [str(request.get("rejection") or "") for request in requests]
    influx_codes = sorted({code for code in rejection_codes if code.startswith("influx")})
    weak_fired = {
        term: any(
            code.startswith("influx") and code.endswith(f"_{term}") for code in rejection_codes
        )
        for term in WEAK_TERMS
    }
    strong_runtime = any(
        code.startswith("influx") and code.endswith(STRONG_RUNTIME_SUFFIXES)
        for code in rejection_codes
    )
    return {
        "run": f"{run_dir.parent.name}/{run_dir.name}",
        "epoch": r.epoch,
        "model": r.model,
        "prompting": r.prompting,
        "dialect": dialect,
        "success": r.success,
        "failure_class": r.failure_class,
        "probe_calls": len(probes),
        "probe_calls_metrics": r.probe_calls,
        "budget_full": len(probes) >= PROBE_BUDGET,
        "index_seen": index_seen,
        "data_200": data_200,
        "dict_nonempty": dict_nonempty,
        "dict_any": dict_any,
        "masked": masked,
        "invalid_turns": invalid_turns,
        "rescuable_turns": rescuable_turns,
        "repairs": repairs,
        "has_artefact": has_artefact,
        "strong_hits": strong_hits,
        "weak_static": weak_static,
        "weak_fired": weak_fired,
        "strong_runtime": strong_runtime,
        "influx_codes": influx_codes,
        "ops": ops,
        "has_eval": summary_raw is not None,
        "paths": paths,
        "steps_records": bool(steps and re.search(r"""['"]records['"]""", steps)),
        "columnar": bool(
            steps
            and re.search(r"""['"]columns['"]""", steps)
            and re.search(r"""['"]rows['"]""", steps)
        ),
    }


def ordered_arms() -> list[dict[str, Any]]:
    return sorted(
        tr.ARMS,
        key=lambda arm: (
            DIALECTS.index(arm["dialect"]),
            MODEL_ORDER[arm["model"]],
            PROMPT_ORDER[arm["prompting"]],
        ),
    )


def arm_label(arm: dict[str, Any]) -> str:
    strategy = "zs" if arm["prompting"] == "zero_shot" else "ms"
    return f"{tr.MODEL_LABEL[arm['model']]} · {strategy} · E{arm['epoch']}"


def progress_rows(facts: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for arm in ordered_arms():
        arm_facts = facts[arm["key"]]
        for op in OPS:
            states = Counter(fact["ops"][op]["state"] for fact in arm_facts)
            codes: Counter[str] = Counter()
            for fact in arm_facts:
                op_codes = fact["ops"][op]["codes"]
                if op_codes:
                    codes[op_codes.most_common(1)[0][0]] += 1
            rows.append(
                {
                    "arm": arm["key"],
                    "dialect": arm["dialect"],
                    "model": arm["model"],
                    "prompting": arm["prompting"],
                    "epoch": arm["epoch"],
                    "operation": op,
                    "n": len(arm_facts),
                    "accepted": states["accepted"],
                    "partial": states["partial"],
                    "rejected": states["rejected"],
                    "not_reached": states["not_reached"],
                    "not_reached_no_evaluation": sum(
                        1
                        for fact in arm_facts
                        if fact["ops"][op]["state"] == "not_reached" and not fact["has_eval"]
                    ),
                    "top_rejection_codes": "; ".join(
                        f"{code} x{count}" for code, count in codes.most_common(3)
                    ),
                }
            )
    return rows


def probing_rows(facts: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for arm in ordered_arms():
        arm_facts = facts[arm["key"]]
        no_violation = [fact for fact in arm_facts if fact["invalid_turns"] == 0]
        rows.append(
            {
                "arm": arm["key"],
                "dialect": arm["dialect"],
                "model": arm["model"],
                "epoch": arm["epoch"],
                "n": len(arm_facts),
                "probe_calls_mean": sum(fact["probe_calls"] for fact in arm_facts) / len(arm_facts),
                "probe_calls_mean_no_contract_violation": (
                    sum(fact["probe_calls"] for fact in no_violation) / len(no_violation)
                    if no_violation
                    else None
                ),
                "budget_exhausted_runs": sum(fact["budget_full"] for fact in arm_facts),
                "index_seen_runs": sum(fact["index_seen"] for fact in arm_facts),
                "data_endpoint_200_runs": sum(fact["data_200"] for fact in arm_facts),
                "dict_body_nonempty_calls": sum(fact["dict_nonempty"] for fact in arm_facts),
                "dict_body_masked_calls": sum(fact["masked"] for fact in arm_facts),
                "dict_body_masked_runs": sum(fact["masked"] > 0 for fact in arm_facts),
                "dict_body_runs": sum(fact["dict_any"] > 0 for fact in arm_facts),
                "generation_failed_runs": sum(
                    fact["failure_class"] == "generation_failed" for fact in arm_facts
                ),
                "contract_violation_runs": sum(fact["invalid_turns"] > 0 for fact in arm_facts),
                "contract_violation_turns": sum(fact["invalid_turns"] for fact in arm_facts),
                "first_object_rescuable_turns": sum(fact["rescuable_turns"] for fact in arm_facts),
                "first_object_rescuable_runs": sum(
                    fact["invalid_turns"] > 0 and fact["rescuable_turns"] == fact["invalid_turns"]
                    for fact in arm_facts
                ),
                "tool_call_repairs": sum(fact["repairs"] for fact in arm_facts),
                "tool_call_repair_runs": sum(fact["repairs"] > 0 for fact in arm_facts),
            }
        )
    return rows


def canary_rows(facts: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for arm in ordered_arms():
        arm_facts = facts[arm["key"]]
        with_artefact = [fact for fact in arm_facts if fact["has_artefact"]]
        row: dict[str, Any] = {
            "arm": arm["key"],
            "dialect": arm["dialect"],
            "model": arm["model"],
            "epoch": arm["epoch"],
            "n": len(arm_facts),
            "artefacts": len(with_artefact),
            "strong_static_runs": sum(fact["strong_hits"] > 0 for fact in with_artefact),
            "strong_static_hits": sum(fact["strong_hits"] for fact in with_artefact),
            "strong_runtime_runs": sum(fact["strong_runtime"] for fact in arm_facts),
        }
        for term, by_dialect in WEAK_TERMS.items():
            checked = by_dialect[arm["dialect"]]
            row[f"{term}_static"] = sum(fact["weak_static"][term] for fact in with_artefact)
            if checked is None:
                row[f"{term}_op_reached"] = None
                row[f"{term}_fired"] = None
                row[f"{term}_runtime_rate"] = None
                continue
            reached = [fact for fact in arm_facts if any(fact["ops"][op]["n"] for op in checked)]
            fired = sum(fact["weak_fired"][term] for fact in arm_facts)
            row[f"{term}_op_reached"] = len(reached)
            row[f"{term}_fired"] = fired
            row[f"{term}_runtime_rate"] = fired / len(reached) if reached else None
        row["weak_fired_runs"] = sum(any(fact["weak_fired"].values()) for fact in arm_facts)
        influx_codes = Counter(code for fact in arm_facts for code in fact["influx_codes"])
        row["influx_codes"] = "; ".join(
            f"{code} x{count}" for code, count in influx_codes.most_common()
        )
        rows.append(row)
    return rows


def cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    return 1.0 if k >= n else float(sps.beta.ppf(1 - (1 - conf) / 2, k + 1, n - k))


def strong_static(r: RunRecord, project_root: Path) -> tuple[bool, int]:
    run_dir = _run_dir(r, project_root)
    steps = _read_text(run_dir / "steps.py")
    if not steps:
        return False, 0
    scan = scan_text((_read_text(run_dir / "benchmark.feature") or "") + "\n" + steps)
    return True, sum(count for label, count in scan.hits.items() if label in STRONG_LABELS)


def _level1(runs: Sequence[RunRecord], arms: Sequence[dict[str, Any]]) -> list[RunRecord]:
    return [run for arm in arms for run in tr.by_level(tr.arm_runs(runs, arm))["L1"]]


def _stage_b_measured(members: Sequence[RunRecord]) -> int:
    return sum(
        1
        for run in members
        if any(run.stage_b_value(SOURCE_PROXY, op, "throughput_ops_s") is not None for op in OPS)
    )


def _epoch_label(arms: Sequence[dict[str, Any]]) -> str:
    epochs = sorted({int(arm["epoch"]) for arm in arms})
    blocks: list[list[int]] = []
    for epoch in epochs:
        if blocks and epoch == blocks[-1][-1] + 1:
            blocks[-1].append(epoch)
        else:
            blocks.append([epoch])
    parts = []
    for block in blocks:
        if len(block) == 1:
            parts.append(f"E{block[0]}")
        elif len(block) == 2:
            parts.append(f"E{block[0]}/E{block[1]}")
        else:
            parts.append(f"E{block[0]}-E{block[-1]}")
    return ", ".join(parts)


def denominator_rows(runs: Sequence[RunRecord], project_root: Path) -> list[dict[str, Any]]:
    l1 = [run for run in runs if run.guidance_level == "L1" and not run.is_fixture]
    main = _level1(runs, tr.ARMS)
    syntax = _level1(runs, tr.SYNTAX_ONLY_ARMS)
    pipe = _level1(runs, tr.PIPELINE_ARMS)
    pilots = [run for run in l1 if run.epoch is not None and 3 <= run.epoch <= 5]
    pre = [run for run in l1 if run.epoch is None or run.epoch <= 2]
    facade_all = [run for run in l1 if run.epoch is not None and run.epoch >= 3]
    n_lite = len(syntax) + len(pipe)
    main_role, pipe_role = tr.ROLE_MAIN, f"{tr.ROLE_PIPELINE} (flash-lite)"
    main_epochs = _epoch_label(tr.ARMS)
    groups: list[tuple[str, str, str, list[RunRecord], str]] = [
        (
            "Main matrix",
            main_epochs,
            main_role,
            main,
            f"{len(tr.ARMS)} arms x 10 runs; flash, haiku, glm; no flash-lite",
        ),
        *[
            (
                f"Main matrix, {tr.MODEL_LABEL[model]}",
                _epoch_label([arm for arm in tr.ARMS if arm["model"] == model]),
                main_role,
                _level1(runs, [arm for arm in tr.ARMS if arm["model"] == model]),
                f"{tr.MODEL_FULL[model]}, 4 arms (zs/ms x Kestrel/Tessera)",
            )
            for model in tr.MODELS
        ],
        ("Facade pilots", "E3-E5", "pilot (flash)", pilots, "gemini-3.5-flash, pilot workload"),
        (
            "Pilots + main matrix",
            f"E3-E5, {main_epochs}",
            "pilot + main matrix",
            pilots + main,
            "pilots + main matrix",
        ),
        (
            "Pipeline dev.: syntax-only feedback",
            "E6/E7",
            pipe_role,
            syntax,
            "gemini-3.5-flash-lite zs, v3 workload; pipeline development only",
        ),
        (
            "Pipeline dev.: execution feedback",
            "E10-E13",
            pipe_role,
            pipe,
            "gemini-3.5-flash-lite zs + ms; pipeline development only",
        ),
        (
            "All facade epochs (incl. flash-lite)",
            "E3-E17",
            "pooled: pilot + main + pipeline dev.",
            pilots + main + syntax + pipe,
            f"includes {n_lite} flash-lite pipeline-development runs; not a model result",
        ),
        (
            "Pre-facade (context only)",
            "E1/E2",
            "pilot (flash)",
            pre,
            "plain InfluxDB API, old workload; not pooled",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, epochs, role, members, note in groups:
        k = sum(1 for run in members if run.success)
        n = len(members)
        scans = [strong_static(run, project_root) for run in members]
        rows.append(
            {
                "group": name,
                "epochs": epochs,
                "role": role,
                "n": n,
                "success": k,
                "cp95_upper": cp_upper(k, n) if n else None,
                "stage_b_measured_runs": _stage_b_measured(members),
                "flash_lite_runs": sum(1 for run in members if run.model == tr.PIPELINE_MODEL),
                "artefacts": sum(1 for has_steps, _ in scans if has_steps),
                "strong_canary_artefacts": sum(
                    1 for has_steps, hits in scans if has_steps and hits
                ),
                "note": note,
            }
        )
        if name == "Main matrix":
            per_arm = [sum(1 for run in _level1(runs, [arm]) if run.success) for arm in tr.ARMS]
            rows.append(
                {
                    "group": "Main matrix, per arm",
                    "epochs": "one epoch",
                    "role": main_role,
                    "n": 10,
                    "success": max(per_arm),
                    "cp95_upper": cp_upper(max(per_arm), 10),
                    "stage_b_measured_runs": None,
                    "flash_lite_runs": 0,
                    "artefacts": None,
                    "strong_canary_artefacts": None,
                    "note": f"max over the {len(tr.ARMS)} arms; arms with 0/10: {per_arm.count(0)}",
                }
            )
    pooled = next(row for row in rows if row["group"].startswith("All facade epochs"))
    assert pooled["n"] == len(facade_all), (pooled["n"], len(facade_all))
    rows.append(
        {
            "group": "check: all non-fixture L1 runs with epoch >= 3",
            "epochs": "E3-E17",
            "role": "consistency check",
            "n": len(facade_all),
            "success": sum(1 for run in facade_all if run.success),
            "cp95_upper": None,
            "stage_b_measured_runs": _stage_b_measured(facade_all),
            "flash_lite_runs": sum(1 for run in facade_all if run.model == tr.PIPELINE_MODEL),
            "artefacts": None,
            "strong_canary_artefacts": None,
            "note": "must equal 'All facade epochs (incl. flash-lite)'",
        }
    )
    return rows


STATE_COLOR = {
    "accepted": "#1c5cab",
    "partial": "#5598e7",
    "rejected": "#86b6ef",
    "not_reached": tr.GRID,
}
STATE_LABEL = {
    "accepted": "all requests accepted",
    "partial": "partly accepted",
    "rejected": "rejected by the facade",
    "not_reached": "operation not reached",
}


def fig_operation_progress(rows: Sequence[dict[str, Any]], out: Path) -> str:
    tr._style()
    states = [
        state for state in STATE_COLOR if state != "partial" or any(row["partial"] for row in rows)
    ]
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(tr.TEXT_WIDTH_IN, 4.3),
        sharex=True,
        gridspec_kw={"hspace": 0.32, "wspace": 0.2},
    )
    arms_by_dialect = {
        dialect: [arm for arm in ordered_arms() if arm["dialect"] == dialect]
        for dialect in DIALECTS
    }
    for i, dialect in enumerate(DIALECTS):
        arms = arms_by_dialect[dialect]
        for j, op in enumerate(OPS):
            ax = axes[i, j]
            ax.grid(axis="x", color=tr.GRID, linewidth=0.6)
            ax.grid(axis="y", visible=False)
            for y, arm in enumerate(arms):
                row = next(
                    candidate
                    for candidate in rows
                    if candidate["arm"] == arm["key"] and candidate["operation"] == op
                )
                left = 0
                for state in states:
                    count = row[state]
                    if count:
                        ax.barh(
                            y, count, left=left, height=0.72, color=STATE_COLOR[state], linewidth=0
                        )
                        if state != "not_reached":
                            ax.text(
                                left + count / 2,
                                y,
                                str(count),
                                ha="center",
                                va="center",
                                fontsize=6.5,
                                color="white" if state == "accepted" else tr.INK,
                            )
                    left += count
            ax.set_ylim(len(arms) - 0.5, -0.5)
            ax.set_xlim(0, 10)
            ax.set_xticks([0, 5, 10])
            ax.set_xticklabels(["0", "5", "10"])
            ax.set_yticks(range(len(arms)))
            if j == 0:
                ax.set_yticklabels([arm_label(arm) for arm in arms])
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
            ax.set_title(f"{dialect} · {op}")
            if i == 1:
                ax.set_xlabel("runs (of 10)")
    handles = [Patch(color=STATE_COLOR[state], label=STATE_LABEL[state]) for state in states]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(bottom=0.15)
    n_runs = sum(row["n"] for row in rows if row["operation"] == OPS[0])
    tr._header(
        fig,
        f"Main matrix, {len({row['arm'] for row in rows})} arms, {n_runs} L1 runs "
        "(flash, haiku, glm): 0 Stage-A successes, no Stage-B values",
    )
    return tr.save(fig, out, "l1_operation_progress")


def _arm_tex(arm_key: str) -> str:
    arm = tr.ARM_BY_KEY[arm_key]
    strategy = "zs" if arm["prompting"] == "zero_shot" else "ms"
    return f"{tr.MODEL_LABEL[arm['model']]} $\\cdot$ {strategy} $\\cdot$ E{arm['epoch']}"


def tex_progress(rows: Sequence[dict[str, Any]]) -> str:
    body = []
    for arm in ordered_arms():
        cells = [_arm_tex(arm["key"])]
        for op in OPS:
            row = next(
                candidate
                for candidate in rows
                if candidate["arm"] == arm["key"] and candidate["operation"] == op
            )
            accepted = f"{row['accepted']}" + (f"+{row['partial']}p" if row["partial"] else "")
            cells.append(f"{accepted}/{row['rejected']}/{row['not_reached']}")
        body.append(cells)
    for dialect in DIALECTS:
        dialect_rows = [row for row in rows if row["dialect"] == dialect]
        cells = [f"\\textit{{{dialect} total}}"]
        for op in OPS:
            op_rows = [row for row in dialect_rows if row["operation"] == op]
            n_accepted = sum(row["accepted"] for row in op_rows)
            reached = sum(row["accepted"] + row["partial"] + row["rejected"] for row in op_rows)
            cells.append(f"{n_accepted} of {reached} reached")
        body.append(cells)
    return tr.latex_table(
        caption="Graded L1 outcome per arm and operation (draft). Cells: runs with all requests of the operation accepted / rejected by the facade / operation not reached (of 10). Totals: accepted among runs that reached the operation.",
        label="tab:l1_operation_progress",
        header=["Arm", "Write", "Read", "Delete"],
        rows=body,
        align="lccc",
        size=r"\footnotesize",
        midrules_after=(5, 11),
        note="Main matrix only (12 arms). Kestrel arms first (rows 1--6), then Tessera. No main-matrix L1 run passed Stage A and none has a Stage-B measurement; the states are read from the proxy request log. Source: proxy\\_stats.requests of evaluation/summary.json. "
        + tr.tex(tr.ARM_LEGEND),
    )


def tex_probing(rows: Sequence[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        violations = row["contract_violation_runs"]
        body.append(
            [
                _arm_tex(row["arm"]),
                tr.fmt(row["probe_calls_mean"]),
                str(row["budget_exhausted_runs"]),
                str(row["index_seen_runs"]),
                str(row["data_endpoint_200_runs"]),
                f"{row['dict_body_masked_calls']} ({row['dict_body_masked_runs']})"
                if row["dict_body_masked_calls"]
                else "0",
                f"{violations} ({row['first_object_rescuable_runs']})" if violations else "0",
                str(row["tool_call_repairs"]),
            ]
        )
    return tr.latex_table(
        caption="Probing behaviour at L1 per arm (draft, N = 10 per arm, budget 8 calls).",
        label="tab:l1_probing",
        header=[
            "Arm",
            "Calls",
            "Budget",
            "Index",
            "Data 200",
            "Masked (runs)",
            "Contract (resc.)",
            "Repairs",
        ],
        rows=body,
        align="lrrrrrrr",
        size=r"\footnotesize",
        midrules_after=(5,),
        tabcolsep="4pt",
        note="Calls: mean probe calls per run. Budget: runs that used all 8 calls. Index: runs that received the endpoint listing. Data 200: runs with a 200 reply from the data endpoint (Kestrel GET .../points, Tessera POST .../fetch). Masked: non-empty dict bodies that the probe tool serialised with str() and the facade rejected as non\\_json\\_body (calls, runs in parentheses). Contract: runs with a turn that broke the JSON-only reply contract (all classified generation\\_failed when no artefact followed); in parentheses the runs whose invalid turns all start with a valid tool-call object (first-object raw\\_decode would have rescued them). Repairs: turns repaired by the harness JSON-repair list. Main matrix only (12 arms). Kestrel arms rows 1--6. "
        + tr.tex(tr.ARM_LEGEND),
    )


def _static_reached_fired(row: dict[str, Any], term: str) -> str:
    if row[f"{term}_op_reached"] is None:
        return f"{row[f'{term}_static']}/--/--"
    return f"{row[f'{term}_static']}/{row[f'{term}_op_reached']}/{row[f'{term}_fired']}"


def tex_canaries(rows: Sequence[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        body.append(
            [
                _arm_tex(row["arm"]),
                str(row["artefacts"]),
                f"{row['strong_static_runs']}/{row['strong_runtime_runs']}",
                _static_reached_fired(row, "start"),
                _static_reached_fired(row, "limit"),
                _static_reached_fired(row, "points"),
            ]
        )
    return tr.latex_table(
        caption="Canary indicators at L1 per arm (draft). Strong canaries are InfluxDB-specific; weak canaries are generic field names that the facades also reject as InfluxDB forms.",
        label="tab:l1_canaries",
        header=["Arm", "Artef.", "Strong s/r", "start s/r/f", "limit s/r/f", "points s/r/f"],
        rows=body,
        align="lrcccc",
        size=r"\footnotesize",
        midrules_after=(5,),
        tabcolsep="4pt",
        note="Strong (InfluxDB-specific: influx, flux, /api/v2, line protocol, precision, bucket, org, predicate, Authorization Token): s = artefacts with a hit of the project scanner, r = runs with a strong runtime rejection code. Weak: s = artefacts (steps.py) using the term as a body key or query parameter; r = runs that reached an operation in which the facade checks the term (start: read or delete; limit: read; points: write); f = runs in which the facade fired the canary. -- = not a canary in this dialect (Kestrel documents limit; points is an ordinary unknown field there). Runtime counts depend on how far a scenario runs, so compare f with r, not with N. Main matrix only (12 arms). Kestrel arms rows 1--6. "
        + tr.tex(tr.ARM_LEGEND),
    )


TEX_ROLES = (tr.ROLE_MAIN, "pilot (flash)", "pilot + main matrix")
ROLE_TEX = {
    tr.ROLE_MAIN: "main",
    "pilot (flash)": "pilot",
    "pilot + main matrix": "pilot + main",
    f"{tr.ROLE_PIPELINE} (flash-lite)": "pipeline dev.",
    "pooled: pilot + main + pipeline dev.": "pooled",
}


def tex_denominator(rows: Sequence[dict[str, Any]]) -> str:
    body = []
    groups = [row for row in rows if row["role"] in TEX_ROLES]
    for row in groups:
        upper = (
            "--"
            if row["cp95_upper"] is None or row["success"]
            else f"{100 * row['cp95_upper']:.2f}\\,\\%"
        )
        strong = (
            "--"
            if row["artefacts"] is None
            else f"{row['strong_canary_artefacts']}/{row['artefacts']}"
        )
        body.append(
            [
                tr.tex(row["group"]),
                row["epochs"].replace("-", "--"),
                ROLE_TEX[row["role"]],
                f"{row['success']}/{row['n']}",
                upper,
                strong,
            ]
        )
    names = [row["group"] for row in groups]
    after = tuple(
        names.index(group)
        for group in (
            f"Main matrix, {tr.MODEL_LABEL[tr.MODELS[-1]]}",
            "Pilots + main matrix",
        )
    )
    main = next(row for row in groups if row["group"] == "Main matrix")
    facade = next(row for row in groups if row["group"] == "Pilots + main matrix")
    facade_measured = facade["stage_b_measured_runs"]
    pre = next(row for row in rows if row["group"].startswith("Pre-facade"))
    return tr.latex_table(
        caption=f"L1 successes (Stage A) per run group with the exact 95\\,\\% Clopper--Pearson upper bound (draft). The model comparison rests on the main matrix: {main['success']}/{main['n']} (flash, haiku, glm; 12 arms).",
        label="tab:l1_facade_denominator",
        header=["Group", "Epochs", "Role", "Success", "CP upper", "Strong canary"],
        rows=body,
        align="lllrrr",
        size=r"\footnotesize",
        midrules_after=after,
        tabcolsep="3pt",
        note="Success = Stage A (syntax, executed, workload conformant). Two-sided 95\\,\\% interval. "
        + f"Stage B: {facade_measured} of the {facade['n']} L1 runs of the pilots and the main matrix carry a proxy-measured Stage-B value"
        + (
            ", so L1 is empty in both Stage-B views (all measured runs; Stage-A successes only). "
            if facade_measured == 0
            else " (Stage-A failures among them belong to the all-measured view only). "
        )
        + f"The pre-facade context row has {pre['stage_b_measured_runs']}/{pre['n']} measured runs under the old workload; they are not comparable and not used. "
        + "Pilots E3--E5 used gemini-3.5-flash with the pilot workload. The pre-facade row (plain InfluxDB API, pilot workload of about one request per operation) is context only and not pooled. Strong canary: artefacts (steps.py present) with at least one InfluxDB-specific term of the project scanner.",
    )


def tex_figure(rows: Sequence[dict[str, Any]]) -> str:
    def tot(dialect: str, op: str) -> str:
        op_rows = [row for row in rows if row["dialect"] == dialect and row["operation"] == op]
        reached = sum(row["accepted"] + row["partial"] + row["rejected"] for row in op_rows)
        return f"{sum(row['accepted'] for row in op_rows)} of {reached}"

    return "\n".join(
        [
            r"\begin{figure}[H]",
            r"    \centering",
            r"    \includegraphics[width=\columnwidth]{fig/l1_operation_progress.pdf}",
            r"    \caption{Graded L1 outcome per arm and operation, main matrix only (12 arms: gemini-3.5-flash, claude-haiku-5-5, glm-5.3; zero- and multi-shot; draft). No main-matrix L1 run passed Stage A and none has a Stage-B measurement; the states come from the proxy request log, not from Stage A or Stage B. Each bar counts the 10 runs of an arm: all requests of the operation accepted, rejected by the facade, or the operation never reached (no artefact, syntax error or an earlier step failed). Accepted among runs that reached the operation: Kestrel, whose endpoint index documents the read and delete query parameters, "
            + f"write {tot('Kestrel', 'write')}, read {tot('Kestrel', 'read')}, delete {tot('Kestrel', 'delete')}; Tessera write {tot('Tessera', 'write')}, read {tot('Tessera', 'read')}, delete {tot('Tessera', 'delete')}."
            + "}",
            r"    \label{fig:l1_operation_progress}",
            r"\end{figure}",
            "",
        ]
    )


def _csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    def cell(v: Any) -> Any:
        return round(v, 4) if isinstance(v, float) else v

    tr.write_csv(
        path, list(rows[0].keys()), [[cell(value) for value in row.values()] for row in rows]
    )


def main() -> None:
    config, runs = tr.load_runs()
    root = config.paths.project_root
    out = root / "results" / "reports" / "thesis_results" / "extra" / "l1"
    fig_dir, tab_dir, tex_dir = out / "fig", out / "tables", out / "latex"
    for directory in (fig_dir, tab_dir, tex_dir):
        directory.mkdir(parents=True, exist_ok=True)

    facts: dict[str, list[dict[str, Any]]] = {}
    for arm in tr.ARMS:
        members = tr.by_level(tr.arm_runs(runs, arm))["L1"]
        facts[arm["key"]] = [run_facts(run, root) for run in members]
        assert len(members) == 10, (arm["key"], len(members))

    progress = progress_rows(facts)
    probing = probing_rows(facts)
    canaries = canary_rows(facts)
    denominators = denominator_rows(runs, root)

    _csv(tab_dir / "l1_operation_progress.csv", progress)
    _csv(tab_dir / "l1_probing.csv", probing)
    _csv(tab_dir / "l1_canaries.csv", canaries)
    _csv(tab_dir / "l1_facade_denominator.csv", denominators)
    per_run = []
    for arm in ordered_arms():
        for fact in facts[arm["key"]]:
            per_run.append(
                {
                    "arm": arm["key"],
                    "run": fact["run"],
                    "failure_class": fact["failure_class"],
                    **{f"{op}_state": fact["ops"][op]["state"] for op in OPS},
                    **{f"{op}_requests": fact["ops"][op]["n"] for op in OPS},
                    **{f"{op}_accepted": fact["ops"][op]["accepted"] for op in OPS},
                    "probe_calls": fact["probe_calls"],
                    "budget_full": fact["budget_full"],
                    "index_seen": fact["index_seen"],
                    "data_200": fact["data_200"],
                    "masked_calls": fact["masked"],
                    "invalid_turns": fact["invalid_turns"],
                    "rescuable_turns": fact["rescuable_turns"],
                    "repairs": fact["repairs"],
                    "strong_hits": fact["strong_hits"],
                    **{f"static_{term}": fact["weak_static"][term] for term in WEAK_TERMS},
                    "influx_codes": " ".join(fact["influx_codes"]),
                }
            )
    _csv(tab_dir / "l1_per_run.csv", per_run)

    kestrel = [fact for arm in tr.ARMS if arm["dialect"] == "Kestrel" for fact in facts[arm["key"]]]
    reached = [fact for fact in kestrel if fact["ops"]["read"]["n"]]
    with_200_accepted = sum(
        1 for fact in reached if fact["data_200"] and fact["ops"]["read"]["state"] == "accepted"
    )
    with_200_other = sum(1 for fact in reached if fact["data_200"]) - with_200_accepted
    without_200_accepted = sum(
        1 for fact in reached if not fact["data_200"] and fact["ops"]["read"]["state"] == "accepted"
    )
    without_200_other = sum(1 for fact in reached if not fact["data_200"]) - without_200_accepted
    no200_models = sorted(tr.MODEL_LABEL[fact["model"]] for fact in reached if not fact["data_200"])
    table = [[with_200_accepted, with_200_other], [without_200_accepted, without_200_other]]
    has_both_groups = (with_200_accepted + with_200_other) and (
        without_200_accepted + without_200_other
    )
    p_read = float(sps.fisher_exact(table)[1]) if has_both_groups else None
    glm = {
        dialect: [
            fact
            for arm_key, arm_facts in facts.items()
            for fact in arm_facts
            if tr.ARM_BY_KEY[arm_key]["model"] == "glm"
            and tr.ARM_BY_KEY[arm_key]["dialect"] == dialect
        ]
        for dialect in DIALECTS
    }
    generation_failed = {
        dialect: sum(fact["failure_class"] == "generation_failed" for fact in glm[dialect])
        for dialect in DIALECTS
    }
    violation_runs = {
        dialect: sum(fact["invalid_turns"] > 0 for fact in glm[dialect]) for dialect in DIALECTS
    }
    p_glm = tr.fisher(
        generation_failed["Tessera"],
        len(glm["Tessera"]),
        generation_failed["Kestrel"],
        len(glm["Kestrel"]),
    )
    tests = [
        {
            "test": "Kestrel read accepted ~ data-endpoint 200 during probing (runs that reached the read)",
            "table": f"[[{with_200_accepted},{with_200_other}],[{without_200_accepted},{without_200_other}]]",
            "p_fisher": p_read,
            "note": (
                f"no-200 runs: {', '.join(no200_models)} (model confounded); exploratory, uncorrected"
                if p_read is not None
                else "not testable: every main-matrix Kestrel run that reached the read had a data-endpoint 200 (no contrast group); descriptive only"
            ),
        },
        {
            "test": "glm generation_failed Tessera vs Kestrel (L1)",
            "table": f"{generation_failed['Tessera']}/{len(glm['Tessera'])} vs {generation_failed['Kestrel']}/{len(glm['Kestrel'])}",
            "p_fisher": p_glm,
            "note": f"runs with an invalid (non-JSON-contract) turn: Tessera {violation_runs['Tessera']}, Kestrel {violation_runs['Kestrel']}; exploratory, uncorrected",
        },
    ]
    _csv(tab_dir / "l1_tests.csv", tests)

    fig_operation_progress(progress, fig_dir)
    (tex_dir / "l1_operation_progress.tex").write_text(tex_progress(progress), encoding="utf-8")
    (tex_dir / "l1_operation_progress_figure.tex").write_text(
        tex_figure(progress), encoding="utf-8"
    )
    (tex_dir / "l1_probing.tex").write_text(tex_probing(probing), encoding="utf-8")
    (tex_dir / "l1_canaries.tex").write_text(tex_canaries(canaries), encoding="utf-8")
    (tex_dir / "l1_facade_denominator.tex").write_text(
        tex_denominator(denominators), encoding="utf-8"
    )

    write_ok = [
        (arm_key, fact)
        for arm_key, arm_facts in facts.items()
        for fact in arm_facts
        if fact["ops"]["write"]["state"] in ("accepted", "partial")
    ]

    kestrel_artefacts = [
        fact
        for arm in tr.ARMS
        if arm["dialect"] == "Kestrel"
        for fact in facts[arm["key"]]
        if fact["has_artefact"]
    ]
    columnar = (sum(fact["columnar"] for fact in kestrel_artefacts), len(kestrel_artefacts))

    headline = _headlines(progress, probing, canaries, denominators, tests, write_ok)
    headline.append(
        f"- Kestrel columnar write body ({{columns, rows}}) in main-arm L1 steps.py: {columnar[0]}/{columnar[1]}"
    )
    print("\n".join(headline))
    _write_readme(out, headline, write_ok, columnar)


def _sum_static_reached_fired(rows: Sequence[dict[str, Any]], term: str) -> str:
    static = sum(row[f"{term}_static"] for row in rows)
    if rows[0][f"{term}_op_reached"] is None:
        return f"{term} {static}/--/-- (no canary in this dialect)"
    reached = sum(row[f"{term}_op_reached"] for row in rows)
    fired = sum(row[f"{term}_fired"] for row in rows)
    return f"{term} {static}/{reached}/{fired}"


def _headlines(
    progress: Sequence[dict[str, Any]],
    probing: Sequence[dict[str, Any]],
    canaries: Sequence[dict[str, Any]],
    denominators: Sequence[dict[str, Any]],
    tests: Sequence[dict[str, Any]],
    write_ok: Sequence[tuple[str, dict[str, Any]]],
) -> list[str]:
    lines = [
        "## Key numbers",
        "",
        "Success denominators (Stage A). Rows with role 'pipeline development (flash-lite)' and the pooled row are not model results.",
        "",
    ]
    for row in denominators:
        upper = "" if row["cp95_upper"] is None else f", CP95 upper {100 * row['cp95_upper']:.2f} %"
        canaries_txt = (
            ""
            if row["artefacts"] is None
            else f", strong canaries in {row['strong_canary_artefacts']}/{row['artefacts']} artefacts"
        )
        lite = f", of which {row['flash_lite_runs']} flash-lite" if row["flash_lite_runs"] else ""
        lines.append(
            f"- {row['group']} ({row['epochs']}; {row['role']}): {row['success']}/{row['n']}{lite}{upper}{canaries_txt}"
        )
    lines.append(
        "- Graded outcome, main matrix only (runs that reached the operation; all requests accepted / reached; rejected; not reached):"
    )
    for dialect in DIALECTS:
        for op in OPS:
            op_rows = [
                row for row in progress if row["dialect"] == dialect and row["operation"] == op
            ]
            accepted, partial, rejected, not_reached = (
                sum(row[state] for row in op_rows)
                for state in ("accepted", "partial", "rejected", "not_reached")
            )
            lines.append(
                f"  - {dialect} {op}: accepted {accepted}/{accepted + partial + rejected}"
                + (f" (+{partial} partly)" if partial else "")
                + f", rejected {rejected}, not reached {not_reached}"
            )
    for test in tests:
        p_txt = "n/a" if test["p_fisher"] is None else f"{test['p_fisher']:.4f}"
        lines.append(f"- {test['test']}: {test['table']}, Fisher p = {p_txt} ({test['note']})")
    lines.append(
        "- Probing per model x dialect (budget exhausted / masked dict-body calls in runs / contract-violation runs, first-object rescuable runs / repairs):"
    )
    for dialect in DIALECTS:
        for model in tr.MODELS:
            cells = [row for row in probing if row["dialect"] == dialect and row["model"] == model]
            n = sum(row["n"] for row in cells)
            lines.append(
                f"  - {dialect} {tr.MODEL_LABEL[model]}: budget {sum(row['budget_exhausted_runs'] for row in cells)}/{n}, "
                f"index {sum(row['index_seen_runs'] for row in cells)}/{n}, data-200 {sum(row['data_endpoint_200_runs'] for row in cells)}/{n}, "
                f"masked {sum(row['dict_body_masked_calls'] for row in cells)} calls in {sum(row['dict_body_masked_runs'] for row in cells)}/{sum(row['dict_body_runs'] for row in cells)} dict-body runs, "
                f"contract {sum(row['contract_violation_runs'] for row in cells)} (rescuable {sum(row['first_object_rescuable_runs'] for row in cells)}; turns {sum(row['first_object_rescuable_turns'] for row in cells)}/{sum(row['contract_violation_turns'] for row in cells)}), "
                f"generation_failed {sum(row['generation_failed_runs'] for row in cells)}, repairs {sum(row['tool_call_repairs'] for row in cells)}"
            )
    lines.append(
        f"- Index listing seen: {sum(row['index_seen_runs'] for row in probing)}/{sum(row['n'] for row in probing)} runs"
    )
    lines.append(
        "- Canaries per model x dialect (strong static runs / artefacts; start static, reached, fired):"
    )
    for dialect in DIALECTS:
        for model in tr.MODELS:
            cells = [row for row in canaries if row["dialect"] == dialect and row["model"] == model]
            lines.append(
                f"  - {dialect} {tr.MODEL_LABEL[model]}: strong {sum(row['strong_static_runs'] for row in cells)}/{sum(row['artefacts'] for row in cells)} (runtime {sum(row['strong_runtime_runs'] for row in cells)}); "
                + "; ".join(_sum_static_reached_fired(cells, term) for term in WEAK_TERMS)
            )
    for arm_key, fact in write_ok:
        ops = fact["ops"]
        lines.append(
            f"- Write accepted: {arm_key} {fact['run']} (E{fact['epoch']}): write {ops['write']['accepted']}/{ops['write']['n']}, "
            f"read {ops['read']['accepted']}/{ops['read']['n']} ({dict(ops['read']['codes'])}), "
            f"delete {ops['delete']['accepted']}/{ops['delete']['n']} ({dict(ops['delete']['codes'])}); "
            f"probes: {' | '.join(fact['paths'])}"
        )
    return lines


def _write_readme(
    out: Path,
    headline: Sequence[str],
    write_ok: Sequence[tuple[str, dict[str, Any]]],
    columnar: tuple[int, int],
) -> None:
    callout = []
    for arm_key, fact in write_ok:
        probed_ingest = any("/ingest" in path for path in fact["paths"])
        callout += [
            f"**Callout: {'the only' if len(write_ok) == 1 else 'an'} accepted L1 write in the main matrix.** `{fact['run']}` ({arm_key}, E{fact['epoch']}, {fact['failure_class']}). "
            f"The steps sent a {'`records` body' if fact['steps_records'] else 'JSON body'} to `/ingest` and {fact['ops']['write']['accepted']}/{fact['ops']['write']['n']} write requests were accepted, "
            f"{'after probing' if probed_ingest else 'without ever probing'} `/ingest`. Its {fact['probe_calls']} probe calls: "
            + "; ".join(f"`{path}`" for path in fact["paths"])
            + f". {fact['masked']} of its dict-body fetch probes were masked by the `str(body)` bug. "
            f"Afterwards all {fact['ops']['read']['n']} reads were rejected ({', '.join(f'{code} x{count}' for code, count in fact['ops']['read']['codes'].items())}) "
            f"and the delete as well ({', '.join(f'{code} x{count}' for code, count in fact['ops']['delete']['codes'].items())}). "
            f"A single run, but it shows that the Tessera write schema can be guessed without probing it, whereas Kestrel's columnar body appears in {columnar[0]}/{columnar[1]} main-arm Kestrel L1 artefacts.",
            "",
        ]
    text = [
        "# L1 / RQ1 detail (thesis_l1.py)",
        "",
        "Generated by `uv run python scripts/thesis_l1.py`. Data: the L1 runs of the run collection (results/raw/metrics.csv + results/artifacts); main matrix: flash zero-shot E8/E9, flash multi-shot E12/E13 (13 Sep 2026), haiku E14/E15 and glm E16/E17 (9 Oct 2026). Descriptive counts; the two Fisher tests are exploratory and uncorrected.",
        "",
        "**Model comparison uses one Gemini model.** The figure and the operation-progress, probing, canary and per-run tables cover only the 12 main-matrix arms (gemini-3.5-flash, claude-haiku-5-5, glm-5.3; zero-/multi-shot x Kestrel/Tessera; 120 L1 runs). gemini-3.5-flash-lite was only used to develop and test the pipeline cheaply; its L1 runs (E6/E7 syntax-only feedback, E10-E13 execution feedback; 60 runs) appear only in `tables/l1_facade_denominator.csv` (record only), in rows whose role is `pipeline development (flash-lite)`, and in the pooled row `All facade epochs (incl. flash-lite)`, which states how many flash-lite runs it contains. They are never a model result, and no figure or LaTeX table shows or names them. Epochs are not pooled except in that success-only table.",
        "",
        "**Stage A vs Stage B.** No facade-epoch L1 run passed Stage A (main matrix 0/120; pilots + main matrix 0/136), and no facade-epoch L1 run carries a proxy-measured Stage-B value (only the pre-facade E1/E2 context runs, old workload, do), so both Stage-B views (all measured runs; Stage-A successes only) are empty at L1 and no Stage-B figure exists here. The graded operation outcome is read from the proxy request log (which requests the facade accepted or rejected); it is a descriptive diagnostic, neither Stage A nor Stage B.",
        "",
        "## Files",
        "",
        "- `fig/l1_operation_progress.{pdf,png}`: main matrix only; small multiples (dialect x operation), one stacked bar per arm: all requests of the operation accepted / rejected by the facade / operation not reached.",
        "- `tables/l1_operation_progress.csv` + `latex/l1_operation_progress.tex` (tab:l1_operation_progress), figure float `latex/l1_operation_progress_figure.tex` (fig:l1_operation_progress): per arm and operation; CSV also counts runs without evaluation and the top rejection codes.",
        "- `tables/l1_probing.csv` + `latex/l1_probing.tex` (tab:l1_probing): mean probe calls, budget exhausted (8), index seen, data-endpoint 200, dict-body calls masked by `str(body)`, JSON-contract violations and first-object rescuable runs/turns, tool-call repairs.",
        "- `tables/l1_canaries.csv` + `latex/l1_canaries.tex` (tab:l1_canaries): strong canaries (static scan + runtime codes), weak canaries start/limit/points as static use, runs reaching the checked operation, runs where the facade fired.",
        "- `tables/l1_facade_denominator.csv` + `latex/l1_facade_denominator.tex` (tab:l1_facade_denominator): L1 Stage-A successes per run group with Clopper-Pearson upper bounds and a `role` column. Main matrix 0/120 (also per model and the per-arm maximum), pilots E3-E5 (flash) 0/16, facade epochs of pilots + main matrix 0/136, pre-facade E1/E2 as context row (LaTeX). The CSV additionally keeps, for the record, pipeline development (flash-lite) E6/E7 0/20 and E10-E13 0/40 and the pooled facade epochs 0/196 including the 60 flash-lite runs. Columns `stage_b_measured_runs` and `flash_lite_runs` per row.",
        "- `tables/l1_tests.csv`: two exploratory, uncorrected Fisher tests over main-matrix runs (Kestrel read acceptance ~ data-endpoint 200, reported as not testable when there is no contrast group; glm generation_failed Tessera vs Kestrel).",
        "- `tables/l1_per_run.csv`: the per-run facts behind the main-matrix tables (120 runs, no flash-lite).",
        "",
        "## Definitions",
        "",
        "- Operation state per run (from `evaluation/summary.json` proxy_stats.requests): *accepted* = every request of the operation passed the facade (no rejection code, status < 400); *partly accepted* = some did; *rejected* = none did; *not reached* = no request of that operation (no artefact, syntax error, or an earlier step failed).",
        "- Probing (from `probe_rounds.json`): data endpoint = Kestrel `GET .../points`, Tessera `POST .../fetch`; index = a 200 reply that lists `endpoints`; masked = non-empty dict/list body answered with `non_json_body` (the probe tool sends `str(body)`).",
        "- JSON contract (from `turns.jsonl`): a turn with `parsed_kind == invalid`; rescuable = `json.JSONDecoder.raw_decode` on the first `{` yields a dict with a `tool` key. Repairs = turns with a `json_repair` entry.",
        "- Strong canaries: labels influx, flux, /api/v2, line protocol, precision, bucket, org, predicate, Authorization Token of `evaluation/contamination.scan_text` over benchmark.feature + steps.py; runtime = facade rejection codes `influx_*` that name an InfluxDB-only form. Weak canaries: `start`, `limit`, `points` used as body key or query parameter in steps.py; runtime counts are normalised by the runs that reached the operation the facade checks (Kestrel documents `limit`, and `points` is no canary there).",
        "- Tests are exploratory and uncorrected (two separate descriptive questions, no family).",
        "",
        *callout,
        *headline,
        "",
    ]
    (out / "README.md").write_text("\n".join(text), encoding="utf-8")


if __name__ == "__main__":
    main()
