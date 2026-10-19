# Benchmark-Agent

How does the level of human guidance affect the quality of LLM-generated performance benchmarks compared to an established reference benchmark for a known open-source system?

Code of the bachelor thesis by Willi Leonard Horn (Technische Universität Berlin, Scalable Software Systems group, 2026). The data is in the separate repository `benchmark-agent-dataset`.

## Structure

- `src/benchmark_agent/`: harness for prompting, LLM clients, generation, execution, measurement and evaluation
- `config/`: conditions, models, guidance levels and workloads
- `inputs/`: API specifications, use cases and task descriptions per guidance level
- `reference/`: the pinned reference benchmark
- `sut/`: test setup of the system under test (InfluxDB)
- `scripts/`: analysis and figure scripts used in the thesis
- `tests/`: test suite
