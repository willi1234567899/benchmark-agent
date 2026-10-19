# Write Use Case

The benchmark's `write` operation should represent appending records to a series of the Tessera
time-series store via `POST /tessera/v1/ingest` (JSON body with `records`, query parameters
`space`, `store` and `ts_unit`; API key in the `X-Tessera-Key` header).

Expectations:

- the scenario should describe a write-oriented benchmark action
- the request should target the ingest endpoint, not fetch or purge
- record preparation should be separated from request execution where helpful
- later evaluation should be able to recognize the scenario as a write benchmark
