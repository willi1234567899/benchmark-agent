# Pinned Tessera benchmark task

Generate executable Gherkin + behave artifacts for the Tessera time-series store (HTTP API v1)
that cover the three operations write (ingest records), read (fetch rows) and delete (purge a series).

The evaluation harness injects connection settings via environment variables:

- `TESSERA_URL` (base URL of the system under test)
- `TESSERA_KEY` (API key)
- `TESSERA_SPACE` (space name)
- `TESSERA_STORE` (store name)

Pinned smoke workload (same numbers for every condition):

- write: ingest 100 simple records into series `benchmark` (timestamps in nanoseconds), one request with all 100 records, 1 parallel client, uncompressed, low label cardinality
- read: fetch rows of series `benchmark` from store `$TESSERA_STORE`, window last 1 hour, at most 10 rows, 1 parallel client, small result
- delete: purge series `benchmark` over the full time range (from `1970-01-01T00:00:00Z` until `2100-01-01T00:00:00Z`), 1 parallel client

Output:

1. one complete Gherkin `.feature` document
2. one complete Python `steps.py` module compatible with `behave`

The generated Python must be syntactically valid. The generated Gherkin must be parseable by `behave`.
Live HTTP calls in the steps should use the environment variables above.
