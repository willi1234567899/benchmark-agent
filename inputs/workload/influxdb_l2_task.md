# Pinned InfluxDB benchmark task

Generate executable Gherkin + behave artifacts for InfluxDB v2 that cover write, read, and delete.

The evaluation harness injects connection settings via environment variables:

- `INFLUX_URL` / `INFLUXDB_URL`
- `INFLUX_TOKEN`
- `INFLUX_ORG`
- `INFLUX_BUCKET`

Pinned smoke workload (same numbers for every condition):

- write: 100 simple line-protocol points into measurement `benchmark` (precision `ns`), batch size 100, 1 batch, 1 parallel client, uncompressed, low tag cardinality
- read: Flux query on bucket `$INFLUX_BUCKET`, measurement `benchmark`, range last 1 hour, limit 10, 1 parallel client, small result
- delete: remove measurement `benchmark` (predicate `_measurement="benchmark"`) over the full time range, 1 parallel client

Output:

1. one complete Gherkin `.feature` document
2. one complete Python `steps.py` module compatible with `behave`

The generated Python must be syntactically valid. The generated Gherkin must be parseable by `behave`.
Live HTTP calls in the steps should use the environment variables above.
