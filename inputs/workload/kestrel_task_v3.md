# Pinned Kestrel benchmark task (workload v3)

Generate executable Gherkin + behave artifacts for the Kestrel metrics service (REST API v1)
that cover the three operations write (append points), read (read points) and delete (remove points).

The evaluation harness injects connection settings via environment variables:

- `KESTREL_ENDPOINT` (base URL of the system under test)
- `KESTREL_TOKEN` (API token)
- `KESTREL_TENANT` (tenant name)
- `KESTREL_STORE` (store name)

Pinned workload v3 (same numbers for every condition; the request counts are part of the workload):

- write: append 1000 points to series `benchmark` in 10 requests of 100 points each (one request per batch, sent one after another); every point carries the label `host` (cycle over `h1`, `h2`, `h3`, `h4`, `h5`, 20 points per batch each), one numeric metric named `value`, and a time given in **milliseconds** since the Unix epoch (the service must know that the times are milliseconds); 1 parallel client, uncompressed
- read: read points of series `benchmark` from store `$KESTREL_STORE` for the last 2 hours, restricted to points whose label `host` equals `h1`, at most 25 points; send this identical read 20 times in a row (one request each); 1 parallel client
- delete: remove only the points of series `benchmark` whose label `host` equals `h1`, over the full time range (from `1970-01-01T00:00:00Z` to `2100-01-01T00:00:00Z`); one request; 1 parallel client

Output:

1. one complete Gherkin `.feature` document
2. one complete Python `steps.py` module compatible with `behave`

The generated Python must be syntactically valid. The generated Gherkin must be parseable by `behave`.
Live HTTP calls in the steps should use the environment variables above.
