# Read Use Case

The benchmark's `read` operation should represent reading the points of a series inside a time
window via `GET /v1/tenants/{tenant}/stores/{store}/series/{series}/points` (query parameters
`since`, optional `until`, `limit` and `label.<name>`; API token in the `X-Api-Token` header).

Expectations:

- the scenario should clearly describe a read action
- the request should be a `GET` on the points resource, not a write or a removal
- the read should be framed as a benchmarked read workload, not as setup logic
- later evaluation should be able to recognize the scenario as a read benchmark
