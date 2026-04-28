# Architecture

MemoryKernel is a local-first project memory layer for AI agents. The system is
split into a small public surface and isolated runtime components that keep each
workspace's memory independent.

## Public Surface

- CLI: `memk remember`, `memk recall`, `memk context`, `memk health`, and
  daemon/admin commands.
- MCP: `memk-mcp` exposes the starter agent tool surface.
- REST: the local daemon exposes versioned `/v1` endpoints.
- SDKs: Python and Node.js clients wrap the daemon API.

## Runtime Flow

```text
user or agent input
  -> CLI, MCP, SDK, or REST
  -> MemoryKernelService
  -> workspace runtime
  -> SQLite storage plus in-memory retrieval index
  -> ranked memory, facts, or context
```

Each workspace owns its own `.memk/` directory, manifest, SQLite database, and
generation counter. This keeps project memory local and avoids mixing state
between repositories.

## Core Modules

| Module | Responsibility |
| --- | --- |
| `memk.core` | Service layer, runtime management, embeddings, metrics, jobs, scoring |
| `memk.storage` | SQLite schema, migrations, graph tables, sharding helpers |
| `memk.retrieval` | Vector index and ranked retrieval |
| `memk.context` | Agent context assembly |
| `memk.extraction` | Rule-based and optional NLP fact extraction |
| `memk.sync` | HLC, oplog, Merkle recovery, conflict handling |
| `memk.api` | FastAPI request models and `/v1` routes |
| `memk.cli` | Typer command-line interface |
| `memk.mcp` | MCP stdio server for agent tools |
| `memk.sdk` | Python client |

## Storage

MemoryKernel uses SQLite with WAL mode for local persistence. Raw memories,
extracted facts, graph entities, sync metadata, and runtime diagnostics are kept
in the workspace database.

Runtime state under `.memk/` is not committed to Git. Backups are created with
`memk backup` and restored with `memk restore`.

## Retrieval

The read path combines local embeddings, lexical signals, importance,
confidence, and recency. Search returns ranked memories and facts. Context
building trims those results into a compact prompt block for an agent.

The deterministic fallback embedder keeps tests and constrained environments
usable without downloading a production embedding model.

## Agent Integration

MCP is the preferred integration path for AI tools because agents can call
memory tools directly. The initial MCP server intentionally exposes four tools:

- `memk_remember`
- `memk_recall`
- `memk_context`
- `memk_health`

CLI, REST, and SDK paths remain available for developer workflows and custom
integrations.
