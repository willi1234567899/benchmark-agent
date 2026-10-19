# Delete Use Case

The benchmark's `delete` operation should represent removing the records of a series inside a time
range via `POST /tessera/v1/purge` (JSON body with `series`, `from` and `until`; query parameters
`space` and `store`; API key in the `X-Tessera-Key` header).

Expectations:

- the scenario should clearly describe a purge/delete action
- the request should target the purge endpoint, not ingest or fetch
- the series and the time range should be part of the logical delete operation
- later evaluation should be able to recognize the scenario as a delete benchmark
