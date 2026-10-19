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

Pinned workload v3 (same numbers for every condition; the request counts are part of the workload):

- write: ingest 1000 records into series `benchmark` in 10 requests of 100 records each (one request per batch, sent one after another); every record carries the label `host` (cycle over `h1`, `h2`, `h3`, `h4`, `h5`, 20 records per batch each), one numeric value named `value`, and a timestamp given in **milliseconds** since the Unix epoch (the store must know that the timestamps are milliseconds); 1 parallel client, uncompressed
- read: fetch rows of series `benchmark` from store `$TESSERA_STORE` for the last 2 hours, restricted to rows whose label `host` equals `h1`, at most 25 rows; send this identical fetch 20 times in a row (one request each); 1 parallel client
- delete: purge only the records of series `benchmark` whose label `host` equals `h1`, over the full time range (from `1970-01-01T00:00:00Z` until `2100-01-01T00:00:00Z`); one request; 1 parallel client
