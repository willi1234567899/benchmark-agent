# Write Use Case

The benchmark's `write` operation should represent appending points to a series of the Kestrel
metrics service via `PUT /v1/tenants/{tenant}/stores/{store}/series/{series}/points` (columnar
JSON body with `columns`, `rows` and `labels`; query parameter `unit`; API token in the
`X-Api-Token` header; the reply is `201`).

Expectations:

- the scenario should describe a write-oriented benchmark action
- the request should be a `PUT` on the points resource, not a read or a removal
- point preparation should be separated from request execution where helpful
- later evaluation should be able to recognize the scenario as a write benchmark
