# Pinned Kestrel benchmark task (workload v2)

Generate executable Gherkin + behave artifacts for the Kestrel metrics service (REST API v1)
that cover the three operations write (append points), read (read points) and delete (remove points).

The evaluation harness injects connection settings via environment variables:

- `KESTREL_ENDPOINT` (base URL of the system under test)
- `KESTREL_TOKEN` (API token)
- `KESTREL_TENANT` (tenant name)
- `KESTREL_STORE` (store name)

Pinned workload v2 (same numbers for every condition):

- write: append 100 points to series `benchmark` in one request; every point carries the label `host` (cycle over `h1`, `h2`, `h3`, `h4`, `h5`, 20 points each), one numeric metric named `value`, and a time given in **milliseconds** since the Unix epoch (the service must know that the times are milliseconds); 1 parallel client, uncompressed
- read: read points of series `benchmark` from store `$KESTREL_STORE` for the last 2 hours, restricted to points whose label `host` equals `h1`, at most 25 points; 1 parallel client
- delete: remove only the points of series `benchmark` whose label `host` equals `h1`, over the full time range (from `1970-01-01T00:00:00Z` to `2100-01-01T00:00:00Z`); 1 parallel client

Output:

1. one complete Gherkin `.feature` document
2. one complete Python `steps.py` module compatible with `behave`

The generated Python must be syntactically valid. The generated Gherkin must be parseable by `behave`.
Live HTTP calls in the steps should use the environment variables above.
