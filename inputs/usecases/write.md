# Write Use Case

The benchmark's `write` operation should represent inserting line protocol data into InfluxDB v2 via
`POST /api/v2/write`.

Expectations:

- the scenario should describe a write-oriented benchmark action
- the request should target the write endpoint, not query or delete
- payload preparation should be separated from request execution where helpful
- later evaluation should be able to recognize the scenario as a write benchmark
