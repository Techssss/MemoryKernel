# MemoryKernel Testing Guide

This repository now has two validation layers:

1. The regular pytest suite for storage, graph, sync, conflict, service, and migration behavior.
2. The optional benchmark suite in `bench/` for ingestion, retrieval, graph propagation, hybrid sync, and conflict discovery.

## Regular Test Suite

Run:

```powershell
python -m pytest -q -rs tests
```

Expected current result:

```text
128 passed, 25 skipped
```

Skipped tests are intentional unless the required optional dependencies or environment flags are present:

- `tests/test_spacy_extractor.py`: requires the `en_core_web_sm` spaCy model.
- `tests/test_large_scale_graph_sync_stress.py`: requires `MEMK_RUN_LARGE_STRESS=1`.

## Large Stress Test

Run:

```powershell
$env:MEMK_RUN_LARGE_STRESS = "1"
python -m pytest -q -rs tests/test_large_scale_graph_sync_stress.py
```

This covers graph indexing, Merkle delta sync, BLOB propagation, fact reconciliation, checkpoints, archive/unarchive state, and final convergence across two replicas.

## Daemon Soak Test

Run:

```powershell
$env:MEMK_RUN_SOAK = "1"
python -m pytest -q -rs tests/test_daemon_soak.py
```

This repeats daemon health checks and request ID propagation. It is opt-in so
the regular suite stays fast and dependency-light.

## Benchmark Suite

Run a small offline benchmark first:

```powershell
python -m bench.runner --size 1000
```

The runner writes:

```text
docs/deep_stress_test_report.md
```

The benchmark suite uses a deterministic lightweight embedder so it can run without downloading a sentence-transformer model. This validates pipeline behavior and relative bottlenecks, not final production embedding quality.

## Bench Modules

- `bench/dataset.py`: deterministic synthetic memories and fact-like content.
- `bench/metrics.py`: latency, throughput, memory, and error summaries.
- `bench/ingestion.py`: bulk insert pressure.
- `bench/retrieval.py`: semantic-style and multi-hop retrieval checks.
- `bench/graph_bench.py`: graph propagation latency under density pressure.
- `bench/sync_bench.py`: hybrid oplog/Merkle recovery and conflict detection.
- `bench/runner.py`: orchestrates all benchmark modules and writes the report.

## Known Gaps

- Multi-node sync beyond two replicas is not yet covered by the benchmark runner.
- The benchmark report is generated output and is ignored by git.
- Production semantic quality still depends on the configured embedding model.
