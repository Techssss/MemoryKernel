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

## Performance Profiles

MemoryKernel has three runtime profiles:

- `MEMK_PROFILE=lite` is the default. It uses SQLite FTS5 candidate search,
  hashing rerank, smaller SQLite cache settings, lazy background workers, and no
  RAM vector or graph index unless explicitly enabled.
- `MEMK_PROFILE=balanced` keeps the candidate-first path but uses a larger
  candidate set.
- `MEMK_PROFILE=quality` is opt-in for heavier semantic model stacks.

`MEMK_INDEX_MODE=ram` enables the warm RAM vector index path. Leave it unset for
low-RAM local agent workflows.

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
