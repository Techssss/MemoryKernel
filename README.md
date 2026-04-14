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
- **Token-Aware Context Builder:** Truncates payloads scientifically. Prioritizes rigid facts over chaotic noise.
- **Zero-Friction MVP:** Powered entirely by standard library SQLite. No ORMs, no heavy external dependencies.
- **Deep Observability:** Built-in `memk doctor` and a dedicated `decisions` telemetry table.

---

## 🚀 Installation

MemoryKernel works right out of the box.

```bash
# Clone the repository
git clone https://github.com/your-username/MemoryKernel.git
cd MemoryKernel

# Install as an editable package
pip install -e .
```

Verify the installation:
```bash
memk --help
```

---

## ⚡ Quickstart & CLI Examples

MemoryKernel comes with a rich CLI (`Typer` + `Rich`) that interacts flawlessly with the core SQLite adapter.

### 1. Initialize the Kernel
Bootstrap your local file-backed database `mem.db` cleanly.
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
*(MemoryKernel will automatically demote "Java" and reconcile the active truth to "Python".)*

### 3. Build Token-Aware LLM Payload
Inject directly into your LLM System Prompt. The context builder cleanly groups subjects and fits them tightly within your token budget limits.
```bash
memk build-context "tech stack" --max-chars 500
```
**Output Example:**
```yaml
--- BEGIN CONTEXT ---
Project facts:
  - project architecture relies on dependency injection
  - system uses Python
Raw memories:
  - ... [TRUNCATED_DUE_TO_BUDGET]
--- END CONTEXT ---
```

### 4. Health & Observability check
Inspect the internal structure of your agent's brain without writing custom SQL.
```bash
memk doctor
```

---

## 🏗 Architecture Overview

MemoryKernel follows strict [Clean Architecture/Backend Dev Guidelines](https://github.com/sickn33/antigravity-awesome-skills).

- **`memk.storage`**: No-ORM, multi-table SQLite engine enforcing data immutability for `memories`, mutable states for `facts`, and auditability for `decisions`.
- **`memk.extraction`**: Pluggable Extractor engines. The MVP (`RuleBasedExtractor`) converts raw strings into SPO (Subject-Predicate-Object) Pydantic Models. Designed to be hot-swapped with upcoming OpenAI/Anthropic extractors.
- **`memk.retrieval`**: Retrieves and implicitly merges/scores disparate memory tables. Ensures `Active Facts` rank mathematically higher than unstructured memories.
- **`memk.context`**: The compiler. Reads the `Retrieval` payload and enforces hard bounds. Protects the LLM from token overflow.

---

## 🛣 Roadmap

- [x] Basic SQLite Storage & CLI interface
- [x] Triplet Extraction & Normalization 
- [x] Fact Strict-Reconciliation algorithm
- [x] Observability (`memk doctor`)
- [ ] **VectorDB Integration:** Upgrade retrieval from SQL `LIKE` to Semantic Hybrid Search using `sqlite-vss`.
- [ ] **LLM-Based Extraction Class:** Add drop-in APIs for OpenAI/Anthropic extraction.
- [ ] **Decay & Recency Bias:** Native forgetting curve mathematical models.

---

## 🤝 Contribution Guidelines

This project embraces pragmatic, no-nonsense backend development guidelines.

1. Keep it **Local-First**.
2. **Do Not Overengineer:** Standard Library Python (`sqlite3`, `uuid`, etc.) is preferred over heavy frameworks unless justified.
3. Every new core mechanic MUST have matching TDD rules in `tests/`.

To run the local test suite:
```bash
pip install pytest
pytest tests/ -v
```

**License:** MIT
