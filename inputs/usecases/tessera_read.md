# Read Use Case

The benchmark's `read` operation should represent fetching the rows of a series inside a time
window via `POST /tessera/v1/fetch` (JSON body with `store`, `series`, `window` and `max_rows`;
query parameter `space`; API key in the `X-Tessera-Key` header).

Expectations:

- the scenario should clearly describe a fetch/read action
- the request should target the fetch endpoint, not ingest or purge
- the fetch should be framed as a benchmarked read workload, not as setup logic
- later evaluation should be able to recognize the scenario as a read benchmark
