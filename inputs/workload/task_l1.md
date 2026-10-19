Create executable Gherkin performance benchmarks for InfluxDB covering the three operations write, read, and delete.

Return one complete `.feature` file and matching `behave` step definitions.

Do not assume extra documentation. You may probe the reachable system with the provided tools before you write the artifacts.

When the artifacts are executed later, the harness injects the connection settings via environment variables:

- `INFLUX_URL` / `INFLUXDB_URL` (base URL of the system under test)
- `INFLUX_TOKEN` (API token; send it as `Authorization: Token <value>`)
- `INFLUX_ORG` (organisation name)
- `INFLUX_BUCKET` (bucket name)

Live HTTP calls in the steps must read these environment variables; do not hard-code hosts, tokens, or names.

Pinned smoke workload (same numbers for every condition):

- write: 100 simple line-protocol points into measurement `benchmark` (precision `ns`), batch size 100, 1 batch, 1 parallel client, uncompressed, low tag cardinality
- read: Flux query on bucket `$INFLUX_BUCKET`, measurement `benchmark`, range last 1 hour, limit 10, 1 parallel client, small result
- delete: remove measurement `benchmark` (predicate `_measurement="benchmark"`) over the full time range, 1 parallel client
The generated Python must be syntactically valid. The generated Gherkin must be parseable by `behave`.
