Create executable Gherkin performance benchmarks for the Kestrel metrics service covering the three operations write, read, and delete.

Return one complete `.feature` file and matching `behave` step definitions.

Do not assume any documentation of the service's HTTP API. You may probe the reachable system with the provided tools before you write the artifacts.

When the artifacts are executed later, the harness injects the connection settings via environment variables:

- `KESTREL_ENDPOINT` (base URL of the system under test)
- `KESTREL_TOKEN` (API token; the service expects it in a request header)
- `KESTREL_TENANT` (tenant name)
- `KESTREL_STORE` (store name)

Live HTTP calls in the steps must read these environment variables; do not hard-code hosts, tokens, or names.
The generated Python must be syntactically valid. The generated Gherkin must be parseable by `behave`.

Pinned workload v3 (same numbers for every condition; the request counts are part of the workload):

- write: append 1000 points to series `benchmark` in 10 requests of 100 points each (one request per batch, sent one after another); every point carries the label `host` (cycle over `h1`, `h2`, `h3`, `h4`, `h5`, 20 points per batch each), one numeric metric named `value`, and a time given in **milliseconds** since the Unix epoch (the service must know that the times are milliseconds); 1 parallel client, uncompressed
- read: read points of series `benchmark` from store `$KESTREL_STORE` for the last 2 hours, restricted to points whose label `host` equals `h1`, at most 25 points; send this identical read 20 times in a row (one request each); 1 parallel client
- delete: remove only the points of series `benchmark` whose label `host` equals `h1`, over the full time range (from `1970-01-01T00:00:00Z` to `2100-01-01T00:00:00Z`); one request; 1 parallel client
