# MemoryKernel

[![CI](https://github.com/Techssss/MemoryKernel/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Techssss/MemoryKernel/actions/workflows/ci.yml)

Your AI coding agent forgets project decisions between sessions. MemoryKernel
gives each project a local memory that agents can remember, recall, and check.

Use it when you are tired of re-explaining:

- why the team chose a library, API shape, or deployment path
- where important project conventions live
- what caused a bug and how it was fixed
- user preferences and repeated workflow instructions
- context that Claude, Cursor, VS Code agents, OpenClaw, or Codex should keep
  across sessions

MemoryKernel is local-first. Project memory lives in `.memk/` inside your
workspace and is backed by SQLite.

Current status: **beta / active development**. The core storage, retrieval,
graph, sync, SDK, and CI paths are tested. The product surface is being shaped
for broader professional use.

## Start In 30 Seconds

Install from a clone:

```bash
git clone https://github.com/Techssss/MemoryKernel.git
cd MemoryKernel
python -m pip install -e .
```

Inside any project you want your agent to remember:

```bash
memk remember "The billing service owns invoice numbering"
memk recall "who owns invoice numbering?"
memk health
```

No `init` step is required. The first memory command creates `.memk/`
automatically.

Need the shortest product tour?

```bash
memk guide
```

Base install is intentionally lightweight. It does not require torch,
sentence-transformers, scikit-learn, or spaCy. MemoryKernel falls back to a
deterministic local embedder so setup works offline. In `auto` mode, it uses the
semantic model when installed and otherwise uses hashing.

For stronger semantic recall, install the optional model stack:

```bash
python -m pip install -e ".[semantic]"
```

If model startup is slow or you want deterministic low-memory mode:

```bash
export MEMK_EMBEDDER=hashing
```

## The Three Commands

| Command | Use it for |
| --- | --- |
| `memk remember "..."` | Save a durable project fact, decision, bug fix, preference, or workflow |
| `memk recall "..."` | Ask what the project memory already knows |
| `memk health` | See whether memory is initialized, indexed, and useful |

`memk init` is optional and lightweight. It creates `.memk/` and prints the
same next-step guide without loading an embedding model.

For longer agent prompts:

```bash
memk context "What should I know before changing billing?"
```

## Give Your Agent Memory

The recommended integration path is MCP:

```bash
memk-mcp
```

MemoryKernel exposes five starter tools:

| MCP tool | What the agent does |
| --- | --- |
| `memk_guide` | Explains when to remember, recall, and build context |
| `memk_remember` | Stores project memory |
| `memk_recall` | Recalls relevant memory |
| `memk_context` | Builds compact context before work |
| `memk_health` | Checks memory health and next actions |

Print setup snippets for your tool:

```bash
memk setup claude
memk setup cursor
memk setup vscode
memk setup openclaw
```

Full guides:

- [Agent Setup](./docs/agent_setup.md)
- [MCP Tools](./docs/mcp_tools.md)
- [First 10 Minutes](./docs/quickstart_first_10_minutes.md)

## What Should Be Remembered

Good memories are compact and durable:

```text
Decision: use PostgreSQL for concurrent writes in the sync service.
Bug fix: auth 401 was caused by mismatched MEMK_API_TOKEN.
Workflow: run npm test in sdk/nodejs before publishing the Node SDK.
Preference: user wants concise technical summaries in Vietnamese.
```

Weak memories are temporary observations:

```text
Read README.md.
Ran tests.
Opened file X.
```

The goal is not to record every action. The goal is to preserve knowledge that
would otherwise be lost when the agent session resets.

## Setup By Tool

### Claude Code

```bash
claude mcp add --transport stdio memorykernel --scope user -- memk-mcp
```

### Cursor

Add to `~/.cursor/mcp.json` or `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "memorykernel": {
      "command": "memk-mcp",
      "args": []
    }
  }
}
```

### VS Code

Add to `.vscode/mcp.json` or your user MCP config:

```json
{
  "servers": {
    "memorykernel": {
      "type": "stdio",
      "command": "memk-mcp",
      "args": []
    }
  }
}
```

### OpenClaw

```bash
openclaw mcp set memorykernel '{"command":"memk-mcp"}'
```

## Why MemoryKernel

- **Project scoped**: each repository gets its own `.memk/` memory.
- **MCP-first**: agents can use memory tools directly.
- **Local-first**: SQLite storage by default, no hosted dependency required.
- **Light by default**: heavy embedding and NLP stacks are optional extras.
- **Agent context**: `memk context` turns recalled memory into a compact prompt
  block.
- **Developer friendly**: CLI, REST API, Python SDK, and Node.js SDK.
- **Recoverable**: backup/restore, schema migrations, diagnostics, metrics, and
  sync hardening are part of the project.

## How It Works

```text
agent or developer
  -> memk CLI, memk-mcp, SDK, or REST API
  -> MemoryKernel service
  -> project-local SQLite memory store
  -> ranked memories, facts, and context
```

MemoryKernel stores raw memories, extracts fact-like knowledge, and retrieves
relevant context using local retrieval signals. It also includes graph sidecar
tables, generation tracking, HLC/oplog sync hardening, Merkle recovery, and
workspace diagnostics.

## Python SDK

```python
from memk.sdk import MemoryKernel

mk = MemoryKernel()

mk.remember("The billing service owns invoice numbering", importance=0.8)
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

Start the daemon:

```bash
memk serve
```

Versioned endpoints live under `/v1`:

```text
GET  /v1/health
POST /v1/remember
POST /v1/search
POST /v1/context
GET  /v1/status
GET  /v1/metrics
POST /v1/ingest/git
```

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
    workspace/          Workspace manifest and generation state
  sdk/nodejs/           Node.js SDK
  bench/                Synthetic benchmark and stress runners
  tests/                Pytest suite
  examples/             Focused integration examples
  docs/                 User, integration, and operations docs
```

Local runtime state belongs in `.memk/`, `.tmp/`, `tmp_*`, and generated `.db`
files. External reference snapshots should stay outside Git tracking.

## Testing

```bash
python -m pytest -q -rs tests
```

Current verified result on April 28, 2026:

```text
139 passed, 28 skipped
```

Expected skips:

- `tests/test_spacy_extractor.py` when `en_core_web_sm` is not installed.
- `tests/test_large_scale_graph_sync_stress.py` unless
  `MEMK_RUN_LARGE_STRESS=1`.

See [TESTING_GUIDE.md](./TESTING_GUIDE.md) for more detail.

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

- Expand MCP tool parity.
- Add stronger visual status surfaces.
- Ship editor/plugin templates for common agent workflows.

## License

MIT License. See [LICENSE](./LICENSE).
