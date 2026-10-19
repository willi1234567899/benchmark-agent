# Delete Use Case

The benchmark's `delete` operation should represent removing data from InfluxDB v2 via
`POST /api/v2/delete`.

Expectations:

- the scenario should clearly describe a delete action
- the request should target the delete endpoint, not write or query
- the predicate and time range should be part of the logical delete operation
- later evaluation should be able to recognize the scenario as a delete benchmark
