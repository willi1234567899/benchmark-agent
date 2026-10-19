# Read Use Case

The benchmark's `read` operation should represent a Flux query against InfluxDB v2 via
`POST /api/v2/query`.

Expectations:

- the scenario should clearly describe a query/read action
- the request should target the query endpoint, not write or delete
- the query should be framed as a benchmarked read workload, not as setup logic
- later evaluation should be able to recognize the scenario as a read benchmark
