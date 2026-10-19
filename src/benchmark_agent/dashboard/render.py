from __future__ import annotations

import html
import json
from functools import cache
from pathlib import Path
from typing import Any

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DATA_ELEMENT_ID = "dashboard-data"


@cache
def _asset(name: str) -> str:
    return (ASSETS_DIR / name).read_text(encoding="utf-8")


def embed_json(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def render_html(payload: dict[str, Any], *, refresh_seconds: int | None = None) -> str:
    title = str(payload.get("title") or "Benchmark-Agent · Run-Dashboard")
    refresh = (
        f'<meta http-equiv="refresh" content="{int(refresh_seconds)}">'
        if refresh_seconds and refresh_seconds > 0
        else ""
    )
    synthetic = bool(payload.get("synthetic"))
    banner = (
        '<div class="banner banner-synthetic" role="status">'
        "SYNTHETISCHE SEED-DATEN: Layout-Vorschau, keine Messung. "
        "Nicht als Ergebnis zitieren.</div>"
        if synthetic
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh}
<title>{html.escape(title)}</title>
<style>
{_asset("dashboard.css")}
</style>
</head>
<body>
<div id="app" class="viz-root">
{banner}
<header class="topbar">
  <div>
    <h1>{html.escape(title)}</h1>
    <p class="subtitle" id="subtitle"></p>
  </div>
  <div class="topbar-actions">
    <button type="button" id="theme-toggle" class="btn" title="Farbschema wechseln">
      Theme: <span id="theme-label">auto</span>
    </button>
    <button type="button" id="reload" class="btn" title="Seite neu laden">Neu laden</button>
  </div>
</header>
<section id="issues" class="issues" hidden></section>
<section class="filters" id="filters" aria-label="Filter"></section>
<section id="empty" class="empty" hidden></section>
<section id="kpis" class="kpi-row" aria-label="Kennzahlen"></section>
<main id="charts" class="grid"></main>
<section id="statistics" class="stats"></section>
<section id="runs" class="runs"></section>
<section id="matrices" class="matrices"></section>
<footer id="definitions" class="definitions"></footer>
<div id="tooltip" class="tooltip" role="tooltip" hidden></div>
<aside id="drawer" class="drawer" hidden></aside>
</div>
<script id="{DATA_ELEMENT_ID}" type="application/json">{embed_json(payload)}</script>
<script>
{_asset("dashboard.js")}
</script>
</body>
</html>
"""
