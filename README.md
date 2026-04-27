# MemoryKernel

Local-first project memory for AI agents and developer workflows.

MemoryKernel (`memk`) gives a project a persistent, queryable memory layer backed by
SQLite. It stores raw memories, extracts fact-like knowledge, builds local retrieval
context, and keeps workspace state isolated so agents can reuse project knowledge
across sessions.

Current status: **beta / active development**. The core storage, retrieval, graph, and
sync paths are covered by tests, but the product surface is still being hardened for
broader professional use.

## What It Does

- Local SQLite storage with WAL mode and forward-only schema migrations.
- Semantic and lexical retrieval over memories and facts.
- Deterministic offline embedding fallback for tests and constrained environments.
- Workspace isolation with generation tracking for stale-context detection.
- Git history ingestion and manual memory capture through CLI, REST API, and SDKs.
- Knowledge graph sidecar tables for entities, mentions, edges, and consolidated facts.
- Delta sync, HLC versioning, Merkle recovery, checkpoints, and conflict visibility.
- Basic observability through diagnostics, metrics, benchmark reports, and health checks.

## Maturity Snapshot

| Area | Status |
| --- | --- |
| Core storage and retrieval | Functional and tested |
| Graph and sync hardening | Functional with stress coverage |
| CLI and REST API | Usable, still being polished |
| Python SDK | Usable synchronous client |
| Node.js SDK | Usable HTTP client, build pipeline still minimal |
| File watcher | MVP, metadata/generation oriented |
| Extraction quality | Regex baseline plus optional spaCy/GLiNER paths |
| Packaging and release process | In progress |

## Installation

```bash
git clone https://github.com/Techssss/MemoryKernel.git
cd MemoryKernel

python -m pip install -e ".[dev]"
```

Start the local daemon:

```bash
memk serve
```

## Quick Start

Initialize a workspace:

```bash
cd /path/to/your/project
memk init
```

Add and search memory:

```bash
memk add "The API endpoint for users is /api/v1/users"
memk search "users API endpoint"
memk context "How do I call the users API?"
memk doctor
```

`memk remember` is kept as a friendly alias for `memk add`.

Ingest recent Git history:

```bash
memk ingest --limit 50
```

Run the file watcher:

```bash
memk watch start --foreground
```

## Python SDK

```python
from memk.sdk import MemoryKernel

mk = MemoryKernel()

memory_id = mk.remember(
    "The billing service owns invoice numbering",
    importance=0.8,
)

results = mk.search("who owns invoice numbering?", limit=5)
context = mk.context("How should an agent update billing code?", max_chars=1200)
status = mk.status()
```

For older integrations, `MemoryKernelClient` is available as an alias of
`MemoryKernel`.

## Node.js SDK

```typescript
import { MemoryKernel } from "@memk/sdk";

const mk = new MemoryKernel();

await mk.remember("The frontend uses React");
const results = await mk.search("frontend framework", { limit: 5 });
const context = await mk.context("What should I know before editing UI code?");
```

## REST API

The daemon exposes versioned endpoints under `/v1`:

```text
GET  /v1/health
POST /v1/remember
POST /v1/search
POST /v1/context
GET  /v1/status
GET  /v1/metrics
POST /v1/ingest/git
```

Legacy daemon endpoints such as `/add`, `/search`, and `/context` are still present for
the current CLI path.

## Project Layout

```text
MemoryKernel/
  memk/                 Core Python package
    api/                FastAPI request models and v1 routes
    cli/                Typer CLI
    context/            RAG context builder
    core/               Runtime, embedding, jobs, metrics, services
    extraction/         Fact/entity extraction paths
    ingestion/          Git ingestion
    retrieval/          Index and hybrid retrieval
    server/             Local daemon and process manager
    storage/            SQLite schema, migrations, graph repository
    sync/               HLC, oplog, Merkle, recovery, conflict handling
    synthesis/          Knowledge synthesis helpers
    watcher/            File change watcher
    workspace/          Workspace manifest and generation state
  sdk/nodejs/           Node.js SDK
  bench/                Synthetic benchmark and stress runners
  tests/                Pytest suite
  examples/             Usage examples and demos
  docs/                 Architecture notes and benchmark reports
  neural-memory-main/   Tracked reference snapshot for comparison work
```

Local runtime state belongs in `.memk/`, `.tmp/`, `tmp_*`, and generated `.db` files;
these are ignored and should not be committed.

## Testing

Regular suite:

```bash
python -m pytest -q -rs tests
```

Current verified result on April 27, 2026:

```text
128 passed, 25 skipped
```

Expected skips:

- `tests/test_spacy_extractor.py` when `en_core_web_sm` is not installed.
- `tests/test_large_scale_graph_sync_stress.py` unless `MEMK_RUN_LARGE_STRESS=1`.

Large stress test:

```powershell
$env:MEMK_RUN_LARGE_STRESS = "1"
python -m pytest -q -rs tests/test_large_scale_graph_sync_stress.py
```

Benchmark runner:

```bash
python -m bench.runner --size 1000
```

See [TESTING_GUIDE.md](./TESTING_GUIDE.md) for more detail.

## Benchmark Snapshot

The current offline benchmark validates pipeline behavior with a deterministic
lightweight embedder. It is useful for regression and bottleneck detection, but it is
not a claim about final production embedding quality.

The latest generated report is written to:

```text
docs/deep_stress_test_report.md
```

## Development Notes

- Python target: 3.10+.
- Package name: `memk`.
- CLI entry point: `memk`.
- Default daemon bind: `127.0.0.1:15301`.
- Data remains local unless an embedding/extraction provider you configure downloads
  or calls external models.
- The project currently prefers correctness and recoverability over raw throughput.

## Roadmap

- Tighten packaging and release automation.
- Add real coverage reporting instead of static badges.
- Finish CLI/API parity and remove legacy endpoint drift.
- Expand real-world benchmark datasets beyond synthetic stress data.
- Improve extractor quality and configurable model management.
- Add stronger daemon security and multi-user deployment guidance.
- Decide whether the `neural-memory-main/` reference snapshot stays in this repository
  or moves to a separate archive.

## License

MIT License. See [LICENSE](./LICENSE).
