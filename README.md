# MemoryKernel (`memk`) 🧠

**A local-first memory infrastructure library for AI agents.**

MemoryKernel is a lightweight Python SDK and CLI that gives your autonomous AI agents deterministic, token-aware, and highly debuggable long-term memory. It bridges the gap between chaotic raw chat logs and precise LLM context windows.

---

## 🛑 The Problem: Why a Vector DB is not enough

If you build AI agents, you eventually run into "the memory problem." 
Most engineers default to tossing chat logs into a Vector Database. This breaks at scale because:
1. **Token Bloat:** You quickly max out the LLM context window with repetitive, noisy paragraphs.
2. **Conflicting Realities:** "User uses Java" and "User switched to Python" exist simultaneously in the Vector DB. The LLM gets confused (hallucinates) trying to reconcile them.
3. **Black Box Ops:** When the agent hallucinates, trying to debug *why* it recalled a specific 2-month-old chat fragment is a nightmare.

**MemoryKernel** handles the complex Memory Lifecycle. It intercepts raw text, distills it into reconcilable Structured Facts (Triplets), ranks them, and neatly packs them into a strict character/token budget.

## ✨ Features

- **Dual-Layer Storage:** Maintains raw immutable chat logs *and* computable Structured Facts.
- **Auto-Reconciliation:** Automatically resolves conflicting facts (shadows the old, elevates the new).
- **Advanced Scored Retrieval (v0.3):** 5-dimensional ranking system combining vector similarity, keyword matching, importance, recency (forgetting curve), and confidence.
- **RAM-First Performance:** Hybrid indexing system for sub-millisecond retrieval in production environments.
- **Async-First Architecture:** Decoupled CPU-bound embedding tasks via a daemon/worker system to prevent event-loop blocking.
- **Token-Aware Context Builder:** Truncates payloads scientifically. Prioritizes User Preferences, Recent Memories, and Stable Facts.
- **Deep Observability:** Built-in `memk doctor` and a dedicated `decisions` telemetry table.

---

## 🚀 Installation

MemoryKernel works right out of the box with standard Python 3.10+.

```bash
# Clone the repository
git clone https://github.com/your-username/MemoryKernel.git
cd MemoryKernel

# Install dependencies (including optional ML features)
pip install -e .
```

Verify the installation:
```bash
memk --help
```

---

## ⚡ Quickstart & CLI Examples

MemoryKernel comes with a rich CLI (`Typer` + `Rich`) that interacts flawlessly with the core engine.

### 1. Initialize the Kernel
Bootstrap your local file-backed database `mem.db` and initialize the schema.
```bash
memk init
```

### 2. Stream Memories
MemoryKernel auto-extracts explicit knowledge while keeping a historical audit log.
```bash
memk add "Project architecture relies on dependency injection"
memk add "System uses Java"

# Later on, the system evolves...
memk add "Actually, System uses Python now"
```
*(MemoryKernel will automatically demographic "Java" and reconcile the active truth to "Python".)*

### 3. Build Token-Aware LLM Payload
Inject directly into your LLM System Prompt. The context builder cleanly groups subjects and fits them tightly within your token budget limits using strict prioritization.
```bash
memk build-context "tech stack" --max-chars 500
```
**Output Example:**
```text
[User Preferences]
• system uses Python

[Recent Memories]
→ Project architecture relies on dependency injection

[Stable Facts]
• user prefers dark mode

[Summary]
Retrieved 2 facts and 1 context logs related to: system, project.
```

### 4. Health & Observability check
Inspect the internal structure of your agent's brain without writing custom SQL.
```bash
memk doctor
```

---

## 🏗 Architecture Overview

MemoryKernel follows strict [Clean Architecture](https://github.com/sickn33/antigravity-awesome-skills).

- **`memk.storage`**: No-ORM, multi-table SQLite engine (v0.3 supports metadata like importance/confidence).
- **`memk.core.scorer`**: The brain of the ranking system. Implements exponential decay for recency and weighted factors for hybrid retrieval.
- **`memk.retrieval`**: Pluggable strategies (`Keyword`, `Hybrid`, `Scored`). `ScoredRetriever` uses RAM-caching and vector similarity for high-performance lookups.
- **`memk.server`**: Optional daemon mode for ultra-low latency. Provides a FastAPI-based IPC layer for async agentic workflows.
- **`memk.context`**: The compiler. Reads the `Retrieval` payload and enforces hard bounds. Protects the LLM from token overflow.

---

## 🛣 Roadmap

- [x] Basic SQLite Storage & CLI interface
- [x] Triplet Extraction & Normalization 
- [x] Fact Strict-Reconciliation algorithm
- [x] Observability (`memk doctor`)
- [x] **Semantic Hybrid Search:** Vector-based retrieval integrated with SQL.
- [x] **Decay & Recency Bias:** Native forgetting curve mathematical models.
- [x] **Async Architecture:** Daemon-mode for non-blocking I/O.
- [ ] **LLM-Based Extraction Class:** Add drop-in APIs for OpenAI/Anthropic extraction.
- [ ] **Graph Visualization:** Web-based dashboard to visualize the knowledge graph.

---

## 🤝 Contribution Guidelines

This project embraces pragmatic, no-nonsense backend development guidelines.

1. Keep it **Local-First**.
2. **Do Not Overengineer:** Standard Library Python (`sqlite3`, `uuid`, etc.) is preferred unless performance justifies external libraries.
3. Every new core mechanic MUST have matching TDD rules in `tests/`.

To run the local test suite:
```bash
python -m pytest
```

**License:** MIT
