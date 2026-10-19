from __future__ import annotations

import csv
import hashlib
import io
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from benchmark_agent.evaluation.runs import SOURCE_PROXY, RunRecord

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thesis_results as tr

OUT_REL = "results/reports/thesis_results/dataset"
METRICS_REL = "results/raw/metrics.csv"
SCRIPT_REL = "scripts/thesis_dataset_export.py"
NA = "NA"
METRICS: tuple[str, ...] = (
    "throughput_ops_s",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
)
EXPECTED_MODEL_ID = {
    "gemini": "gemini-3.5-flash",
    "claude_haiku": "claude-haiku-5-5",
    "glm": "z-ai/glm-5.3",
}
GOLD_MODEL_ID = "gold-fixture"
ARTEFACT_FILES: tuple[tuple[str, str], ...] = (
    ("feature", "benchmark.feature"),
    ("steps", "steps.py"),
    ("metadata", "metadata.json"),
    ("evaluation_summary", "evaluation/summary.json"),
)
CONFIG_PATTERNS: tuple[str, ...] = (
    "default.yaml",
    "models.yaml",
    "conditions.yaml",
    "guidance*.yaml",
    "workload.yaml",
)
STAGE_B_COLUMNS: tuple[str, ...] = tuple(f"{op}_{m}" for op in tr.OPS for m in METRICS)
COLUMNS: tuple[str, ...] = (
    "run_id",
    "row_type",
    "arm",
    "model",
    "model_name",
    "model_id",
    "prompting",
    "dialect",
    "epoch",
    "guidance_level",
    "condition_name",
    "repetition",
    "seed",
    "generated_at",
    "evaluated_at",
    "measurement_day",
    "harness_status",
    "syntax_ok",
    "executed_ok",
    "workload_conformant",
    "stage_a_success",
    "failure_class",
    "failure_detail",
    "feedback_iterations",
    "execution_feedback_rounds",
    "llm_calls",
    "stage_b_measured_stage_a_failed",
    *STAGE_B_COLUMNS,
    "same_day_reference",
    "run_directory",
    *(f"{suffix}_path" for suffix, _ in ARTEFACT_FILES),
)
MONTHS_EN = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def fail(message: str) -> None:
    raise SystemExit(f"dataset export check failed: {message}")


def check(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def cell(value: Any) -> str:
    if value is None or value == "":
        return NA
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([cell(value) for value in row])


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def day_de(day: str) -> str:
    parsed = date.fromisoformat(day)
    return f"{parsed.day:02d}.{parsed.month:02d}.{parsed.year}"


def day_en(day: str) -> str:
    parsed = date.fromisoformat(day)
    return f"{parsed.day} {MONTHS_EN[parsed.month - 1]} {parsed.year}"


def thousands(n: int) -> str:
    return f"{n:,}"


def thousands_de(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def measurement_day(r: RunRecord) -> str | None:
    return r.evaluated_at[:10] if r.evaluated_at else None


def _measurement_days(runs: Iterable[RunRecord]) -> list[str]:
    days = {measurement_day(run) for run in runs}
    return sorted(day for day in days if day)


def short_detail(text: str, root: Path) -> str:
    if not text:
        return ""
    for prefix in (str(root), root.as_posix()):
        text = re.sub(re.escape(prefix) + r"[\\/]?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"results\\[^\s\"']*", lambda m: m.group(0).replace("\\", "/"), text)
    return " ".join(text.split())


def stage_b(r: RunRecord, op: str, metric: str) -> float | None:
    if not r.success:
        return None
    return r.stage_b_value(SOURCE_PROXY, op, metric)


def measured_but_failed(r: RunRecord) -> bool:
    return not r.success and any(
        r.stage_b_value(SOURCE_PROXY, op, metric) is not None for op in tr.OPS for metric in METRICS
    )


def artefact_path(root: Path, run_directory: str, name: str) -> str | None:
    return f"{run_directory}/{name}" if (root / run_directory / name).is_file() else None


def count_files(directory: Path) -> int:
    return sum(1 for path in directory.rglob("*") if path.is_file())


def select_main(runs: Sequence[RunRecord]) -> list[tuple[dict[str, Any], RunRecord]]:
    level_order = {level: i for i, level in enumerate(tr.LEVELS)}
    selected: list[tuple[dict[str, Any], RunRecord]] = []
    for arm in tr.ARMS:
        members = sorted(
            tr.arm_runs(runs, arm),
            key=lambda run: (
                level_order.get(run.guidance_level or "", 99),
                run.repetition,
                run.run_id,
            ),
        )
        check(len(members) == 50, f"{arm['key']}: {len(members)} runs instead of 50")
        per_level = Counter(run.guidance_level for run in members)
        check(
            all(per_level[level] == 10 for level in tr.LEVELS),
            f"{arm['key']}: runs per level {dict(per_level)}",
        )
        for run in members:
            check(
                run.model in tr.MAIN_MODELS,
                f"{run.run_id}: model {run.model} not in the main matrix",
            )
            check(
                run.model_version == EXPECTED_MODEL_ID[arm["model"]],
                f"{run.run_id}: model id {run.model_version} != {EXPECTED_MODEL_ID[arm['model']]}",
            )
            check(run.run_directory is not None, f"{run.run_id}: no run directory")
            selected.append((arm, run))
    check(len(selected) == 600, f"{len(selected)} main-matrix runs instead of 600")
    return selected


def series_key(dialect: str, epoch: int) -> str:
    return f"{dialect} E{epoch}"


def select_gold(runs: Sequence[RunRecord]) -> list[tuple[str, str, RunRecord]]:
    selected: list[tuple[str, str, RunRecord]] = []
    for dialect, epoch in tr.REFERENCE_EPOCHS:
        condition = f"_gold_{dialect.lower()}"
        members = sorted(
            (
                run
                for run in runs
                if run.is_fixture and run.condition_name == condition and run.epoch == epoch
            ),
            key=lambda run: (run.repetition, run.run_id),
        )
        key = series_key(dialect, epoch)
        check(len(members) == 10, f"gold {key}: {len(members)} runs instead of 10")
        for run in members:
            check(run.success, f"gold {run.run_id} failed Stage A")
            check(
                run.model_version == GOLD_MODEL_ID,
                f"gold {run.run_id}: model id {run.model_version}",
            )
            check(run.run_directory is not None, f"gold {run.run_id}: no run directory")
            selected.append((dialect, key, run))
    check(len(selected) == 40, f"{len(selected)} gold runs instead of 40")
    return selected


def common_tail(r: RunRecord, root: Path) -> list[Any]:
    return [
        r.generated_at,
        r.evaluated_at,
        measurement_day(r),
        r.status,
        r.syntax_ok,
        r.executed_ok,
        r.semantics_ok,
        r.success,
        r.failure_class,
        short_detail(r.failure_detail, root),
        r.feedback_iterations,
        r.execution_feedback_rounds,
        r.llm_calls,
        measured_but_failed(r),
        *(stage_b(r, op, metric) for op in tr.OPS for metric in METRICS),
    ]


def main_row(arm: dict[str, Any], r: RunRecord, root: Path) -> list[Any]:
    run_dir = r.run_directory or ""
    return [
        r.run_id,
        "main_matrix",
        arm["key"],
        arm["model"],
        tr.MODEL_FULL[arm["model"]],
        r.model_version,
        arm["prompting"],
        arm["dialect"],
        arm["epoch"],
        r.guidance_level,
        r.condition_name,
        r.repetition,
        r.seed,
        *common_tail(r, root),
        arm["ref"],
        run_dir,
        *(artefact_path(root, run_dir, name) for _, name in ARTEFACT_FILES),
    ]


def gold_row(dialect: str, key: str, r: RunRecord, root: Path) -> list[Any]:
    run_dir = r.run_directory or ""
    return [
        r.run_id,
        "gold_reference",
        "reference",
        r.model,
        "hand-written reference",
        r.model_version,
        None,
        dialect,
        r.epoch,
        None,
        r.condition_name,
        r.repetition,
        r.seed,
        *common_tail(r, root),
        key,
        run_dir,
        *(artefact_path(root, run_dir, name) for _, name in ARTEFACT_FILES),
    ]


def metrics_counts(data: bytes, run_ids: set[str]) -> tuple[int, int, set[str]]:
    reader = csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    header = next(reader)
    run_id_column = header.index("run_id")
    total = exported = 0
    seen: set[str] = set()
    for row in reader:
        if not row:
            continue
        total += 1
        run_id = row[run_id_column] if run_id_column < len(row) else ""
        if run_id in run_ids:
            exported += 1
            seen.add(run_id)
    return total, exported, seen


def config_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in CONFIG_PATTERNS:
        for path in sorted((root / "config").glob(pattern)):
            if path.is_file() and path not in found:
                found.append(path)
    return found


COLUMN_DOC_DE: tuple[tuple[str, str], ...] = (
    (
        "run_id",
        "Lauf-ID `<Bedingung>/run-NNN` (eindeutig; identisch mit `run_id` in `results/raw/metrics.csv`).",
    ),
    (
        "row_type",
        "`main_matrix` (Lauf der Hauptmatrix) oder `gold_reference` (handgeschriebene Referenz).",
    ),
    (
        "arm",
        "Arm-Schlüssel `<Modell>-<zs/ms>-<K/T>` (zs = zero-shot, ms = multi-shot, K = Kestrel, T = Tessera); `reference` bei Gold-Läufen.",
    ),
    (
        "model",
        "Interner Modellschlüssel aus `config/models.yaml` (`gemini`, `claude_haiku`, `glm`; Gold: `manual`).",
    ),
    (
        "model_name",
        "Modellname wie in der Arbeit (gemini-3.5-flash, claude-haiku-5-5, glm-5.3; Gold: hand-written reference).",
    ),
    (
        "model_id",
        "Modell-ID, die der Generierungs-Trace meldet (`model_version`; Gold: `gold-fixture`).",
    ),
    ("prompting", "`zero_shot` oder `multi_shot`; NA bei Gold-Läufen."),
    ("dialect", "API-Fassade vor InfluxDB: `Kestrel` oder `Tessera`."),
    (
        "epoch",
        "Epoche (gleicher Prompt, gleicher Workload, gleiche Kriterien); Epochen werden nie gepoolt.",
    ),
    ("guidance_level", "Guidance-Stufe `L1` bis `L5`; NA bei Gold-Läufen."),
    ("condition_name", "Bedingung = Verzeichnisname unter `results/artifacts/`."),
    (
        "repetition",
        "Wiederholung, wie in `metrics.csv` erfasst: in der Hauptmatrix 1 bis 10 innerhalb der Zelle (Arm × Stufe); bei Gold-Läufen die fortlaufende Laufnummer der Bedingung (E12/E13: 1 bis 10, E16: 12 bis 21, E17: 11 bis 20).",
    ),
    (
        "seed",
        "Seed der Generierung (20260715 bis 20260724, in jeder Epoche dieselben zehn Seeds); NA bei Gold-Läufen.",
    ),
    ("generated_at", "Zeitpunkt der Generierung (ISO 8601, UTC); NA bei Gold-Läufen."),
    ("evaluated_at", "Zeitpunkt der Auswertung/Messung (ISO 8601, UTC)."),
    ("measurement_day", "Messtag = Datum von `evaluated_at` (UTC)."),
    (
        "harness_status",
        "Status der Harness (`ok`, `execution_failed`, `validation_failed`, `generation_failed`).",
    ),
    (
        "syntax_ok",
        "Stage-A-Prüfung 1: Validierung bestanden (Gherkin und Python syntaktisch gültig).",
    ),
    (
        "executed_ok",
        "Stage-A-Prüfung 2: behave-Lauf rechtzeitig beendet, Exit-Code 0, kein fehlgeschlagenes oder abgebrochenes Szenario, mindestens ein bestandenes.",
    ),
    (
        "workload_conformant",
        "Stage-A-Prüfung 3: alle drei Operationen mit mindestens einer fehlerfreien Anfrage am Proxy und Workload konform zur festgelegten Spezifikation (Kriterien v2; Feld `semantics_ok` in `metrics.csv`).",
    ),
    (
        "stage_a_success",
        "Stage-A-Erfolg = `syntax_ok` und `executed_ok` und `workload_conformant`.",
    ),
    ("failure_class", "Fehlerklasse (`evaluation/failure_class.py`); `none` bei Erfolg."),
    (
        "failure_detail",
        "Kurzbeschreibung des Fehlers (max. ca. 160 Zeichen, absoluter Projektpfad durch relativen ersetzt); NA bei Erfolg.",
    ),
    ("feedback_iterations", "Anzahl Feedback-Iterationen der Generierung."),
    ("execution_feedback_rounds", "Davon Runden mit Ausführungs-Feedback."),
    (
        "llm_calls",
        "Anzahl LLM-Aufrufe für diesen Lauf (inkl. API-Probing und Feedback); NA bei Gold-Läufen.",
    ),
    (
        "stage_b_measured_stage_a_failed",
        "`true`, wenn der Lauf Stage A nicht bestanden hat, aber ausgeführt wurde und der Proxy Werte gemessen hat. Diese Werte stehen hier als NA; sie liegen in `results/raw/metrics.csv` und erscheinen in der Arbeit in der Sicht „alle gemessenen Läufe“ (markiert).",
    ),
    (
        "`<op>_throughput_ops_s`",
        "Stage B: Durchsatz in Operationen/s je Operation `<op>` ∈ {write, read, delete}, gemessen am Proxy.",
    ),
    (
        "`<op>_latency_p50_ms`, `_p95_ms`, `_p99_ms`",
        "Stage B: Latenz-Perzentile in ms je Operation, gemessen am Proxy.",
    ),
    (
        "same_day_reference",
        "Gold-Referenzserie desselben Messtags, mit der der Arm verglichen wird (z. B. `Kestrel E16`); bei Gold-Läufen die eigene Serie; NA für E8/E9 (an diesem Tag keine Referenz gemessen).",
    ),
    ("run_directory", "Laufverzeichnis relativ zum Repository-Wurzelverzeichnis."),
    (
        "feature_path",
        "Generiertes Gherkin-Feature `benchmark.feature`; NA, wenn keine Artefakte entstanden sind.",
    ),
    ("steps_path", "Generierte behave-Steps `steps.py`; NA, wenn keine Artefakte entstanden sind."),
    (
        "metadata_path",
        "Generierungs-Metadaten `metadata.json` (Bedingung, Modell, Seed, Prompt-Hash, Token, Feedback).",
    ),
    (
        "evaluation_summary_path",
        "Auswertungs-Zusammenfassung `evaluation/summary.json` (Validierung, Adhärenz, Konformität, Proxy-Statistik).",
    ),
)


def readme(
    main: Sequence[tuple[dict[str, Any], RunRecord]],
    gold: Sequence[tuple[str, str, RunRecord]],
    stats: dict[str, Any],
) -> str:
    lines = [
        "# Datensatz-Export: Hauptmatrix und Gold-Referenz",
        "",
        "Dieser Ordner erschließt die Daten, auf denen die Auswertung der Arbeit beruht: eine Zeile pro Lauf "
        "(Kennzahlen, Prüfergebnisse, Pfade zu den Artefakten) und SHA-256-Prüfsummen der zugehörigen Dateien. "
        "Die Rohdaten selbst liegen unverändert in `results/raw/` und `results/artifacts/`. "
        f"Der Ordner wird vollständig von `{SCRIPT_REL}` erzeugt und nicht von Hand bearbeitet.",
        "",
        "## Inhalt",
        "",
        "| Datei | Inhalt |",
        "|---|---|",
        f"| `main_matrix_runs.csv` | {stats['n_rows']} Zeilen, eine pro Lauf: {stats['n_main']} Läufe der Hauptmatrix und {stats['n_gold']} Gold-Referenzläufe. Spalten siehe unten. |",
        f"| `manifest_sha256.csv` | {thousands_de(stats['n_manifest'])} Zeilen: SHA-256 und Größe jeder exportierten Artefaktdatei (`benchmark.feature`, `steps.py`, `metadata.json`, `evaluation/summary.json`), von `{METRICS_REL}`, der {stats['n_config']} Konfigurationsdateien und von `main_matrix_runs.csv`. |",
        "| `tab_data_sources.tex` | LaTeX-Tabelle *Data sources of the evaluation* (`tab:data-sources`) für die Arbeit. |",
        "| `README.md` | Diese Beschreibung. |",
        "",
        "## Einschlussregel",
        "",
        "Exportiert werden genau zwei Gruppen von Läufen:",
        "",
        "1. **Hauptmatrix** (600 Läufe): die 12 Arme aus `ARMS` in `scripts/thesis_results.py`, also Modell × Prompting × Dialekt mit je genau einer Epoche, je 5 Guidance-Stufen (L1 bis L5) × N = 10. "
        "Ein Lauf gehört zu einem Arm, wenn Epoche, Modellschlüssel und Prompting übereinstimmen und er keine Fixture ist; zusätzlich muss die gemeldete Modell-ID exakt "
        "`gemini-3.5-flash`, `claude-haiku-5-5` bzw. `z-ai/glm-5.3` sein.",
        "2. **Gold-Referenz** (40 Läufe): Fixture-Bedingungen `_gold_kestrel` und `_gold_tessera` in den Epochen 12, 13, 16 und 17, je 10 Läufe pro Serie.",
        "",
        "Läufe außerhalb der Hauptmatrix sind nicht enthalten. Das Skript bricht ab, wenn eine der Zählungen (50 Läufe je Arm, 10 je Arm und Stufe, 10 je Gold-Serie) "
        "oder eine Modell-ID nicht stimmt.",
        "",
        "## Stage-A- und Stage-B-Regel",
        "",
        "- **Stage A** (Ausführungserfolg) = `syntax_ok` ∧ `executed_ok` ∧ `workload_conformant`.",
        "- **Stage B** (Performance) = am Proxy gemessener Durchsatz und Latenz-Perzentile je Operation (write, read, delete). "
        "Ein Wert steht nur bei Stage-A-Erfolgen (alle 40 Gold-Läufe bestehen Stage A); bei jedem anderen Lauf steht **NA**, nie 0. "
        "Innerhalb eines Erfolgs gilt zusätzlich die Harness-Regel aus `evaluation/runs.py` (`stage_b_value`): NA, wenn der Proxy für die Operation keine fehlerfreie Anfrage gesehen hat "
        f"(kommt im Export nicht vor: alle {stats['n_success_main']} Stage-A-Erfolge der Hauptmatrix haben Werte für alle drei Operationen).",
        f"- {stats['n_measured_failed']} Läufe haben Stage A nicht bestanden, wurden aber ausgeführt und gemessen (`stage_b_measured_stage_a_failed = true`). "
        "Ihre Werte stehen hier als NA; sie liegen in `results/raw/metrics.csv` (Quelle `proxy`) und erscheinen in der Arbeit in der Sicht „alle gemessenen Läufe“, dort markiert.",
        "- Fehlende Werte sind in beiden CSV-Dateien einheitlich `NA`; Wahrheitswerte sind `true`/`false`; Zeitstempel sind ISO 8601 in UTC.",
        "",
        "## Messtag und Referenz",
        "",
        "Jeder Arm wird mit der Gold-Serie seines eigenen Messtags verglichen (`same_day_reference`). "
        "Für E8 und E9 wurde am Messtag keine Referenz gemessen; dort steht NA.",
        "",
        "## Läufe je Arm",
        "",
        "| Arm | Modell | Prompting | Dialekt | Epoche | Messtag | Läufe | je Stufe | Stage-A-Erfolge | Referenz (gleicher Tag) |",
        "|---|---|---|---|---|---|---:|---:|---:|---|",
    ]
    by_arm: dict[str, list[RunRecord]] = defaultdict(list)
    for arm, run in main:
        by_arm[arm["key"]].append(run)
    for arm in tr.ARMS:
        members = by_arm[arm["key"]]
        days = _measurement_days(members)
        per_level = sorted(set(Counter(run.guidance_level for run in members).values()))
        lines.append(
            f"| `{arm['key']}` | {tr.MODEL_FULL[arm['model']]} | {tr.PROMPT_LABEL[arm['prompting']]} | {arm['dialect']} | E{arm['epoch']} | "
            f"{', '.join(day_de(day) for day in days)} | {len(members)} | {'/'.join(str(n) for n in per_level)} | "
            f"{sum(run.success for run in members)} | {arm['ref'] or 'NA'} |"
        )
    lines.append(
        f"| **Summe** | | | | | | **{len(main)}** | | **{sum(run.success for _, run in main)}** | |"
    )
    lines += [
        "",
        "## Gold-Referenzserien",
        "",
        "| Serie | Bedingung | Epoche | Messtag | Läufe | Stage-A-Erfolge |",
        "|---|---|---|---|---:|---:|",
    ]
    by_series: dict[str, list[RunRecord]] = defaultdict(list)
    for _, key, run in gold:
        by_series[key].append(run)
    for dialect, epoch in tr.REFERENCE_EPOCHS:
        key = series_key(dialect, epoch)
        members = by_series[key]
        days = _measurement_days(members)
        lines.append(
            f"| {key} | `_gold_{dialect.lower()}` | E{epoch} | {', '.join(day_de(day) for day in days)} | {len(members)} | {sum(run.success for run in members)} |"
        )
    lines += [
        f"| **Summe** | | | | **{len(gold)}** | **{sum(run.success for _, _, run in gold)}** |",
        "",
        "## Spalten von `main_matrix_runs.csv`",
        "",
        "| Spalte | Bedeutung |",
        "|---|---|",
    ]
    for name, meaning in COLUMN_DOC_DE:
        shown = name if name.startswith("`") else f"`{name}`"
        lines.append(f"| {shown} | {meaning} |")
    lines += [
        "",
        f"Reihenfolge der Spalten: {', '.join(f'`{column}`' for column in COLUMNS)}.",
        "",
        "Zeilenreihenfolge: Arme in der Reihenfolge der Tabelle oben, darin Stufe L1 bis L5 und Wiederholung 1 bis 10; danach die Gold-Serien Tessera E12, Kestrel E13, Kestrel E16, Tessera E17.",
        "",
        "## Spalten von `manifest_sha256.csv`",
        "",
        "| Spalte | Bedeutung |",
        "|---|---|",
        "| `scope` | `raw_metrics`, `configuration`, `main_matrix_run`, `gold_reference_run` oder `export`. |",
        "| `run_id` | Lauf-ID bei Artefaktdateien, sonst NA. |",
        "| `file_role` | `metrics_csv`, `config`, `feature`, `steps`, `metadata`, `evaluation_summary` oder `run_table`. |",
        "| `path` | Pfad relativ zum Repository-Wurzelverzeichnis (POSIX-Schreibweise). |",
        "| `size_bytes` | Dateigröße in Byte. |",
        "| `sha256` | SHA-256 des Dateiinhalts (hexadezimal). |",
        "",
        f"Gehasht werden je Lauf nur die vier Kerndateien; fehlen `benchmark.feature` und `steps.py` (Generierung ohne Artefakte, {stats['n_without_artefacts']} Läufe), "
        "entfällt die Zeile und der Pfad in `main_matrix_runs.csv` ist NA. Die Laufverzeichnisse enthalten weitere Dateien (Prompt, Modellantwort, Feedback-Runden, behave-Logs), "
        f"insgesamt {thousands_de(stats['files_main'])} Dateien in den {stats['n_main']} Läufen der Hauptmatrix und {stats['files_gold']} in den {stats['n_gold']} Gold-Läufen. "
        "Von den Konfigurationsdateien wird nur die Prüfsumme festgehalten; `.env` wird weder gelesen noch exportiert.",
        "",
        "## Datenstand",
        "",
        f"- `{METRICS_REL}`: {thousands_de(stats['metrics_rows'])} Datenzeilen, davon {thousands_de(stats['metrics_rows_exported'])} zu exportierten Läufen; SHA-256 `{stats['metrics_sha']}`.",
        f"- Letzte Auswertung eines exportierten Laufs: {day_de(stats['last_day'])}.",
        "",
        "## Reproduktion",
        "",
        "Im Repository-Wurzelverzeichnis:",
        "",
        "```",
        f"uv run python {SCRIPT_REL}",
        "```",
        "",
        "Das Skript liest nur `results/raw/metrics.csv`, `results/artifacts/**` und `config/*.yaml` (über `scripts/thesis_results.py`, dieselbe Datenbasis wie alle Abbildungen und Tabellen der Arbeit). "
        "Es ruft weder das SUT noch Docker noch ein LLM auf. Bei unverändertem Datenstand ist die Ausgabe byte-identisch (keine Zeitstempel). "
        "Eine einzelne Prüfsumme lässt sich z. B. mit `Get-FileHash -Algorithm SHA256 <Pfad>` (PowerShell) oder `sha256sum <Pfad>` (Linux/macOS) gegen `manifest_sha256.csv` prüfen.",
        "",
    ]
    return "\n".join(lines)


def tex_path(path: str) -> str:
    return r"\texttt{" + tr.tex(path).replace("/", r"/\allowbreak{}") + "}"


def tex_table(stats: dict[str, Any]) -> str:
    days: dict[str, list[str]] = defaultdict(list)
    for key, day in stats["gold_days"]:
        days[day].append(key)
    series_text = "; ".join(
        f"{', '.join(keys)} ({day_en(day)})" for day, keys in sorted(days.items())
    )
    rows = [
        (
            "Raw metrics file",
            "One row per run, operation and measurement source (proxy, server; one placeholder row for a run without measurement): Stage-A checks, request counts, throughput and latency percentiles. Covers every evaluated run of the project.",
            f"{thousands(stats['metrics_rows'])} rows, {thousands(stats['metrics_rows_exported'])} of them for exported runs",
            tex_path(METRICS_REL),
        ),
        (
            "Run artefacts",
            f"Per run of the main matrix (12 arms, 5 levels, N = {stats['n_per_cell']}): generated Gherkin feature and Python steps, generation metadata (seed, prompt hash, token usage, feedback rounds), prompt, model response, evaluation summary and behave logs.",
            f"{stats['n_main']} runs, {thousands(stats['files_main'])} files",
            tex_path("results/artifacts/<condition>/run-NNN/"),
        ),
        (
            "Reference runs",
            f"Hand-written reference feature and steps per dialect, 10 runs per series: {series_text}.",
            f"{stats['n_gold']} runs, {thousands(stats['files_gold'])} files",
            tex_path("results/artifacts/_gold_kestrel/")
            + r" \newline "
            + tex_path("results/artifacts/_gold_tessera/"),
        ),
        (
            "Configuration",
            "Harness settings (default.yaml), model ids (models.yaml), condition matrix (conditions.yaml), guidance texts per level and dialect (guidance*.yaml), workload specification (workload.yaml).",
            f"{stats['n_config']} files",
            tex_path("config/"),
        ),
        (
            "Exported run table",
            "One row per exported run: arm, model, level, seed, measurement day, Stage-A checks, failure class, Stage-B values per operation, same-day reference series, artefact paths.",
            f"{stats['n_rows']} rows",
            tex_path(f"{OUT_REL}/main_matrix_runs.csv"),
        ),
        (
            "Checksum manifest",
            "SHA-256 and size of the core files of every exported run (feature, steps, metadata, evaluation summary), the raw metrics file, the configuration files and the run table.",
            f"{thousands(stats['n_manifest'])} rows",
            tex_path(f"{OUT_REL}/manifest_sha256.csv"),
        ),
    ]
    ragged = r">{\raggedright\arraybackslash}"
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Data sources of the evaluation}",
        r"    \label{tab:data-sources}",
        r"    \footnotesize",
        f"    \\begin{{tabularx}}{{\\linewidth}}{{@{{}}{ragged}p{{2.0cm}}{ragged}X{ragged}p{{2.1cm}}{ragged}p{{4.2cm}}@{{}}}}",
        r"        \toprule",
        r"        Source & Content & Size & Path in the repository \\",
        r"        \midrule",
    ]
    for i, (source, content, size, path) in enumerate(rows):
        lines.append(f"        {source} & {tr.tex(content)} & {tr.tex(size)} & {path} \\\\")
        if i < len(rows) - 1:
            lines.append(r"        \addlinespace")
    lines += [
        r"        \bottomrule",
        r"    \end{tabularx}",
        r"    \par\smallskip",
        r"    \begin{minipage}{\linewidth}\footnotesize",
        "    Paths are relative to the repository root; \\texttt{<condition>} is level, model and prompting, e.g.\\ \\texttt{L3\\_glm\\_multishot}. "
        f"The export holds the {stats['n_main']} runs of the main matrix and the {stats['n_gold']} reference runs; runs outside the main matrix are not included. "
        f"Stage-B values are given for Stage-A successes only (NA otherwise). The export is regenerated with \\texttt{{uv run python {tr.tex(SCRIPT_REL)}}}.",
        r"    \end{minipage}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    config, runs = tr.load_runs()
    root: Path = config.paths.project_root
    out = root / OUT_REL
    out.mkdir(parents=True, exist_ok=True)

    main_runs = select_main(runs)
    gold = select_gold(runs)
    run_ids = [run.run_id for _, run in main_runs] + [run.run_id for _, _, run in gold]
    check(len(set(run_ids)) == 640, f"{len(set(run_ids))} distinct run ids instead of 640")

    metrics_path = root / METRICS_REL
    metrics_data = metrics_path.read_bytes()
    metrics_rows, metrics_rows_exported, seen = metrics_counts(metrics_data, set(run_ids))
    missing_rows = sorted(set(run_ids) - seen)
    check(not missing_rows, f"runs without rows in metrics.csv: {missing_rows[:5]}")

    rows = [main_row(arm, run, root) for arm, run in main_runs]
    rows += [gold_row(dialect, key, run, root) for dialect, key, run in gold]
    check(all(len(row) == len(COLUMNS) for row in rows), "row length != header length")
    table_path = out / "main_matrix_runs.csv"
    write_csv(table_path, COLUMNS, rows)

    manifest: list[list[Any]] = [
        [
            "raw_metrics",
            None,
            "metrics_csv",
            METRICS_REL,
            len(metrics_data),
            sha256_bytes(metrics_data),
        ]
    ]
    configs = config_files(root)
    for path in configs:
        rel = path.relative_to(root).as_posix()
        manifest.append(
            ["configuration", None, "config", rel, path.stat().st_size, sha256_file(path)]
        )
    scoped = [("main_matrix_run", run) for _, run in main_runs] + [
        ("gold_reference_run", run) for _, _, run in gold
    ]
    n_without_artefacts = 0
    for scope, run in scoped:
        run_dir = run.run_directory or ""
        if not (root / run_dir / "benchmark.feature").is_file():
            n_without_artefacts += 1
        for role, name in ARTEFACT_FILES:
            path = root / run_dir / name
            if path.is_file():
                manifest.append(
                    [
                        scope,
                        run.run_id,
                        role,
                        f"{run_dir}/{name}",
                        path.stat().st_size,
                        sha256_file(path),
                    ]
                )
    manifest.append(
        [
            "export",
            None,
            "run_table",
            f"{OUT_REL}/main_matrix_runs.csv",
            table_path.stat().st_size,
            sha256_file(table_path),
        ]
    )
    write_csv(
        out / "manifest_sha256.csv",
        ("scope", "run_id", "file_role", "path", "size_bytes", "sha256"),
        manifest,
    )

    gold_days = []
    for dialect, epoch in tr.REFERENCE_EPOCHS:
        key = series_key(dialect, epoch)
        days = {measurement_day(run) for _, series, run in gold if series == key}
        check(len(days) == 1, f"gold {key}: measured on {sorted(day or 'NA' for day in days)}")
        gold_days.append((key, next(iter(days)) or ""))
    stats: dict[str, Any] = {
        "n_main": len(main_runs),
        "n_gold": len(gold),
        "n_rows": len(rows),
        "n_per_cell": 10,
        "n_manifest": len(manifest),
        "n_config": len(configs),
        "n_success_main": sum(run.success for _, run in main_runs),
        "n_measured_failed": sum(measured_but_failed(run) for _, run in main_runs),
        "n_without_artefacts": n_without_artefacts,
        "files_main": sum(count_files(root / (run.run_directory or "")) for _, run in main_runs),
        "files_gold": sum(count_files(root / (run.run_directory or "")) for _, _, run in gold),
        "metrics_rows": metrics_rows,
        "metrics_rows_exported": metrics_rows_exported,
        "metrics_sha": sha256_bytes(metrics_data),
        "last_day": max(_measurement_days(run for _, run in main_runs)),
        "gold_days": gold_days,
    }
    check(
        all(
            run.stage_b_value(SOURCE_PROXY, op, "throughput_ops_s") is not None
            for _, run in main_runs
            if run.success
            for op in tr.OPS
        ),
        "a Stage-A success without a Stage-B value",
    )
    write_text(out / "README.md", readme(main_runs, gold, stats))
    write_text(out / "tab_data_sources.tex", tex_table(stats))

    per_arm = Counter(arm["key"] for arm, _ in main_runs)
    arm_counts = ", ".join(f"{arm_key} {count}" for arm_key, count in per_arm.items())
    print(f"main matrix: {len(main_runs)} runs ({arm_counts})")
    gold_list = ", ".join(f"{key} {day}" for key, day in gold_days)
    print(f"gold reference: {len(gold)} runs ({gold_list})")
    print(
        f"Stage-A successes (main): {stats['n_success_main']}; measured but Stage-A failed: {stats['n_measured_failed']}; without artefacts: {n_without_artefacts}"
    )
    print(f"metrics.csv: {metrics_rows} rows, {metrics_rows_exported} for exported runs")
    print(f"manifest: {len(manifest)} files ({len(configs)} config)")
    print(f"written to {OUT_REL}/")


if __name__ == "__main__":
    main()
