Create executable Gherkin performance benchmarks for the Tessera time-series store covering the three operations write, read, and delete.

Return one complete `.feature` file and matching `behave` step definitions.

Do not assume any documentation of the store's HTTP API. You may probe the reachable system with the provided tools before you write the artifacts.

When the artifacts are executed later, the harness injects the connection settings via environment variables:

- `TESSERA_URL` (base URL of the system under test)
- `TESSERA_KEY` (API key; the store expects it in a request header)
- `TESSERA_SPACE` (space name)
- `TESSERA_STORE` (store name)

Live HTTP calls in the steps must read these environment variables; do not hard-code hosts, keys, or names.
The generated Python must be syntactically valid. The generated Gherkin must be parseable by `behave`.

Pinned smoke workload (same numbers for every condition):

- write: ingest 100 simple records into series `benchmark` (timestamps in nanoseconds), one request with all 100 records, 1 parallel client, uncompressed, low label cardinality
- read: fetch rows of series `benchmark` from store `$TESSERA_STORE`, window last 1 hour, at most 10 rows, 1 parallel client, small result
- delete: purge series `benchmark` over the full time range (from `1970-01-01T00:00:00Z` until `2100-01-01T00:00:00Z`), 1 parallel client
