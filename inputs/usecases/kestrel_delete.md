# Delete Use Case

The benchmark's `delete` operation should represent removing the points of a series inside a time
range via `DELETE /v1/tenants/{tenant}/stores/{store}/series/{series}/points` (query parameters
`from`, `to` and optional `label.<name>`; API token in the `X-Api-Token` header).

Expectations:

- the scenario should clearly describe a removal action
- the request should be a `DELETE` on the points resource, not a write or a read
- the series, the time range and the label restriction should be part of the logical delete operation
- later evaluation should be able to recognize the scenario as a delete benchmark
