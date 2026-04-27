# Changelog

All notable project-facing changes are tracked here.

## 0.1.0-beta - 2026-04-28

### Added

- Versioned REST API documentation and CLI alignment around `/v1`.
- CI workflow for Python tests, Python package build, and Node SDK build.
- Benchmark suite for ingestion, retrieval, graph propagation, hybrid sync, and conflict discovery.
- Large graph/sync stress test gated by `MEMK_RUN_LARGE_STRESS=1`.
- Deterministic hashing embedder fallback for offline tests and constrained environments.

### Changed

- README now describes the project as beta / active development instead of production-ready.
- Package metadata now uses `memk` version `0.1.0`.
- Node SDK package version aligned to `0.1.0`.
- CLI `add`, `search`, and `context` daemon paths now call versioned `/v1` endpoints.
- `memk remember` remains available as an alias for `memk add`.

### Removed

- Local `.memk` workspace state from Git tracking.
- External `neural-memory-main/` reference snapshot from Git tracking. Keep it local only, or archive it in a separate repository if it is needed for comparison work.

### Validation

- Regular suite: `128 passed, 25 skipped`.
- Expected skips: spaCy model-dependent tests and opt-in large stress test.
