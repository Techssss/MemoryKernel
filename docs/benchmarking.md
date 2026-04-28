# Benchmarking

MemoryKernel has two validation modes:

- Regular tests prove correctness of storage, graph, sync, conflict, service, and
  migration behavior.
- Benchmarks estimate latency, throughput, graph propagation, sync recovery, and
  conflict discovery behavior.

Run the regular suite:

```bash
python -m pytest -q -rs tests
```

Run the offline benchmark suite:

```bash
python -m bench.runner --size 1000
```

The benchmark runner writes:

```text
docs/deep_stress_test_report.md
```

That file is generated output and is ignored by Git.

## Current Scope

Benchmarks use deterministic synthetic data and a lightweight deterministic
embedder. This makes runs repeatable and avoids model downloads in constrained
environments.

Valid claims:

- Relative ingestion and retrieval latency for the current code path.
- Graph propagation and sync recovery behavior under synthetic pressure.
- Conflict discovery behavior for generated scenarios.

Invalid claims:

- Final semantic quality of a production embedding model.
- Real-user recall quality across arbitrary domains.
- Multi-node production performance beyond the tested topology.

## Adding A Benchmark

Add new benchmark logic under `bench/`, then update:

- `bench/runner.py`
- `TESTING_GUIDE.md`
- this document

Keep generated reports out of Git unless the report is intentionally curated as
release evidence.
