from __future__ import annotations

import re
from pathlib import Path

from benchmark_agent.models import VocabularyScan

INFLUX_VOCABULARY: tuple[tuple[str, str], ...] = (
    ("influx", r"influx"),
    ("flux", r"\bflux\b|\|>"),
    ("/api/v2", r"/api/v2"),
    ("line protocol", r"line[\s_-]?protocol"),
    ("precision", r"\bprecision\s*[=:]|[\"']precision[\"']"),
    ("bucket", r"\bbuckets?\b"),
    ("org", r"\borg\b|\borgs\b|\borg=|\"org\""),
    ("_measurement", r"_measurement|\bmeasurement\b"),
    ("_field", r"\b_field\b|\b_value\b|\b_time\b"),
    ("predicate", r"\bpredicate\b"),
    ("Authorization: Token", r"authorization\s*[\"':]+\s*token|Token\s+\{"),
)
_COMPILED = [(label, re.compile(pattern, re.IGNORECASE)) for label, pattern in INFLUX_VOCABULARY]


def scan_text(text: str) -> VocabularyScan:
    hits: dict[str, int] = {}
    for label, pattern in _COMPILED:
        count = len(pattern.findall(text))
        if count:
            hits[label] = count
    return VocabularyScan(hits=hits, total=sum(hits.values()))


def scan_artifacts(feature_path: Path, steps_path: Path) -> VocabularyScan:
    parts: list[str] = []
    for path in (feature_path, steps_path):
        if path.is_file():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return scan_text("\n".join(parts))
