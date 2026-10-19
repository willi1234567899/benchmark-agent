# Reference Steps

bddbench step modules (`influx_write_steps.py`, `influx_query_steps.py`,
`influx_delete_steps.py`) depend on the bddbench harness (`src.utils`,
`context.influxdb`, KPI export to a second InfluxDB). They are **not** copied
here, because this repository measures through its own proxy + `/metrics` path.

L5 therefore sees the Gherkin ceiling from `reference/features/`. The agent must
still emit self-contained `steps.py` for `benchmark-agent evaluate`.
