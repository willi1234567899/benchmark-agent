# Pinned Tessera benchmark task (workload v2)

Generate executable Gherkin + behave artifacts for the Tessera time-series store (HTTP API v1)
that cover the three operations write (ingest records), read (fetch rows) and delete (purge records).

The evaluation harness injects connection settings via environment variables:

- `TESSERA_URL` (base URL of the system under test)
- `TESSERA_KEY` (API key)
- `TESSERA_SPACE` (space name)
- `TESSERA_STORE` (store name)

Pinned workload v2 (same numbers for every condition):

- write: ingest 100 records into series `benchmark` in one request; every record carries the label `host` (cycle over `h1`, `h2`, `h3`, `h4`, `h5`, 20 records each), one numeric value named `value`, and a timestamp given in **milliseconds** since the Unix epoch (the store must know that the timestamps are milliseconds); 1 parallel client, uncompressed
- read: fetch rows of series `benchmark` from store `$TESSERA_STORE` for the last 2 hours, restricted to rows whose label `host` equals `h1`, at most 25 rows; 1 parallel client
- delete: purge only the records of series `benchmark` whose label `host` equals `h1`, over the full time range (from `1970-01-01T00:00:00Z` until `2100-01-01T00:00:00Z`); 1 parallel client

Output:

1. one complete Gherkin `.feature` document
2. one complete Python `steps.py` module compatible with `behave`

The generated Python must be syntactically valid. The generated Gherkin must be parseable by `behave`.
Live HTTP calls in the steps should use the environment variables above.
