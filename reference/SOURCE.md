# bddbench reference (ceiling input)

Imported from the prior project [DPS25/bddbench](https://github.com/DPS25/bddbench)
for **L5 guidance only**. These files are the manually written Gherkin for
write / query (read) / delete. They are not the measurement harness: KPI export
to a second InfluxDB and `environment.py` stay in bddbench.

Source:

- repository: `https://github.com/DPS25/bddbench`
- files: `features/influx_write_benchmark.feature`,
  `features/influx_query_benchmark.feature`,
  `features/influx_delete_benchmark.feature`

L5 still has to emit self-contained `benchmark.feature` + `steps.py` for this
repository's evaluate path. The imported features show the ceiling shape
(endpoints, scenario outlines, smoke/load/stress rows).

## bddbench client on the pinned workload (measured 2026-10-11)

`bddbench_pinned/` holds a port of bddbench's own step definitions
(commit `42ddad5`, influxdb-client 1.49.0) that sends exactly the pinned workload
of `tests/fixtures/gold_live` (10 writes x 100 points, ms, host h1..h5; 20 reads
host=h1, -2h, limit 25; 1 delete host=h1). Every change versus the bddbench
source is listed in `bddbench_pinned/PROVENANCE.md` and `changes.diff`.
It was measured in the influx passthrough mode (no facade) as condition
`_bddbench_pinned` (epoch 18), interleaved with `gold_live` as `_ref_live`
(epoch 19), N = 10 each; see the Arbeitsprotokoll entry of 11.10.2026.
Requires the dependency group: `uv run --group bddbench ...`.
