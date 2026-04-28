# MemoryKernel

[![CI](https://github.com/Techssss/MemoryKernel/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Techssss/MemoryKernel/actions/workflows/ci.yml)

Project memory that AI agents can carry across sessions.

MemoryKernel (`memk`) gives every AI coding agent a durable project memory that
survives restarts, context resets, and handoffs. It stores raw memories, extracts
fact-like knowledge, builds local retrieval context, and keeps workspace state
isolated in SQLite.

Current status: **beta / active development**. The core storage, retrieval, graph, and
sync paths are covered by tests, but the product surface is still being hardened for
broader professional use.

## What It Does

- MCP-first memory tools for AI agents.
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
| CLI and REST API | Usable, 3-command onboarding added |
| MCP server | Minimal tool surface for agent integrations |
| Python SDK | Usable synchronous client |
| Node.js SDK | Usable HTTP client with CI build and tests |
| File watcher | MVP, metadata/generation oriented |
| Extraction quality | Regex baseline plus optional spaCy/GLiNER paths |
| Packaging and release process | CI smoke-tested, release workflows manual-gated |

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

For a complete fresh-machine path, see
[First 10 Minutes With MemoryKernel](./docs/quickstart_first_10_minutes.md).

You only need three commands to start:

```bash
cd /path/to/your/project
memk remember "The API endpoint for users is /api/v1/users"
memk recall "users API endpoint"
memk health
```

No explicit init is required for first use. MemoryKernel creates `.memk/`
workspace state automatically when the first memory command runs.

`memk search` remains available for developer workflows. `memk context` builds a
compact context block for an agent prompt:

```bash
memk context "How do I call the users API?"
```

## MCP For Agents

Use the MCP server when an AI tool should remember and recall automatically:

```bash
memk-mcp
```

The starter MCP surface is intentionally small:

| Tool | Purpose |
| --- | --- |
| `memk_remember` | Store project memory |
| `memk_recall` | Recall relevant memory |
| `memk_context` | Build compact agent context |
| `memk_health` | Show memory health and next actions |

Setup guides for Claude Code, Cursor, VS Code, and OpenClaw are in
[Agent Setup](./docs/agent_setup.md). Tool details are in
[MCP Tools](./docs/mcp_tools.md).

Create and restore a local memory backup:

```bash
memk backup
memk restore memk-backup-YYYYMMDDTHHMMSSZ.zip --force
```

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
    mcp/                MCP stdio server for agent tools
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
  examples/             Focused integration examples
  docs/                 User, integration, and operations docs
```

Local runtime state belongs in `.memk/`, `.tmp/`, `tmp_*`, and generated `.db` files.
External reference snapshots such as `neural-memory-main/` should stay outside Git
tracking or live in a separate archive repository.

## Testing

Regular suite:

```bash
python -m pytest -q -rs tests
```

Current verified result on April 28, 2026:

```text
135 passed, 28 skipped
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

Benchmark reports are generated locally and ignored by Git.

## Development Notes

- Python target: 3.10+.
- Package name: `memk`.
- CLI entry point: `memk`.
- Default daemon bind: `127.0.0.1:15301`.
- CI runs Python tests with coverage, package smoke tests, and Node SDK tests.
- Data remains local unless an embedding/extraction provider you configure downloads
  or calls external models.
- The project currently prefers correctness and recoverability over raw throughput.
- Set `MEMK_API_TOKEN` to require bearer-token authentication for protected
  daemon endpoints.

## Documentation

- [Docs Index](./docs/README.md)
- [First 10 Minutes](./docs/quickstart_first_10_minutes.md)
- [Agent Setup](./docs/agent_setup.md)
- [MCP Tools](./docs/mcp_tools.md)
- [Architecture](./docs/architecture.md)
- [Troubleshooting](./docs/troubleshooting.md)
- [Compatibility Matrix](./docs/compatibility.md)
- [REST API v1](./docs/api_v1.md)
- [Benchmarking](./docs/benchmarking.md)
- [Backup And Restore](./docs/backup_restore.md)
- [Upgrade Guide](./docs/upgrade_guide.md)
- [Example Apps](./docs/example_apps.md)
- [Release Process](./docs/release.md)
- [Contributing](./CONTRIBUTING.md)
- [Security Policy](./SECURITY.md)

## Roadmap

The detailed product-readiness checklist lives in [TODO.md](./TODO.md).

Near-term priorities:

- Tighten packaging and release automation.
- Finish CLI/API parity and remove legacy endpoint drift.
- Add stronger multi-user deployment guidance.

## License

MIT License. See [LICENSE](./LICENSE).
