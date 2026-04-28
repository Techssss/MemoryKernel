# MemoryKernel Product TODO

This checklist tracks the next work needed to move MemoryKernel from a clean
developer beta toward a professional product.

Current assessment after the April 28, 2026 release-readiness pass:

- Overall professional product readiness: 78-82%.
- Open-source developer beta readiness: about 85%.
- Commercial-grade product readiness: about 70%.

## P0 - Keep The Beta Stable

Goal: protect the current baseline while new product work lands.

- [x] Keep local runtime state out of Git tracking.
- [x] Remove external reference snapshots from the main source tree.
- [x] Add CI for Python tests, Python package build, and Node SDK build.
- [x] Align CLI daemon calls with versioned `/v1` API endpoints.
- [x] Keep README honest about beta status and current limitations.
- [x] Add CI badges to README after the GitHub Actions workflow is verified on
  the remote repository.
- [x] Add a short `CONTRIBUTING.md` covering setup, tests, commit style, and
  expected PR checks.
- [x] Add an `SECURITY.md` with local-first threat model, supported versions,
  and private disclosure contact.

## P1 - Reach 80%: Release-Ready Open Source Beta

Goal: a developer can install, run, test, and trust the package without local
context from the maintainer.

- [x] Verify `python -m build` in a clean virtual environment on Windows, macOS,
  and Linux.
- [x] Add release workflow for tagged Python artifacts, with manual approval
  before publishing to PyPI.
- [x] Add release workflow for the Node SDK, with manual approval before npm
  publish.
- [x] Add smoke tests that install the built wheel and run the core CLI help
  surface, `memk-mcp --help`, and the Python SDK quickstart.
- [x] Replace static coverage claims with real `pytest --cov` reporting.
- [x] Publish a minimal "first 10 minutes" quickstart that starts from an empty
  machine and ends with a successful memory recall.
- [x] Add a troubleshooting page for dependency installs, local model downloads,
  SQLite permissions, and daemon startup issues.
- [x] Add a compatibility matrix for Python versions, OS support, and optional
  model dependencies.

Acceptance criteria:

- A fresh clone can run all documented setup commands successfully.
- CI proves package build and smoke install on every PR.
- README badges reflect real CI and coverage state.

## P2 - Reach 85%: Professional Developer Product

Goal: the API, CLI, and SDKs feel consistent enough for external developers to
build on.

- [x] Rework README around the user problem, three core commands, MCP setup,
  and concrete memory examples.
- [x] Add `memk setup` snippets for Claude Code, Cursor, VS Code, OpenClaw,
  and generic MCP clients.
- [x] Make heavy embedding/NLP dependencies optional so base install starts
  quickly without torch, sentence-transformers, scikit-learn, or spaCy.
- [x] Keep `memk health` lightweight by avoiding embedding model startup when
  the daemon is not running.
- [x] Make `memk init` lightweight and add `memk guide` for first-run product
  guidance.
- [x] Add MCP `memk_guide` and keep MCP health lightweight without loading the
  service runtime.
- [ ] Finish CLI/API parity for remember, search, context, forget, export,
  import, health, and stats operations.
- [x] Remove or clearly deprecate legacy unversioned daemon endpoints.
- [ ] Add typed response contracts for the Python SDK and ensure FastAPI schemas
  match them.
- [x] Add hand-maintained API reference docs for `/v1`.
- [x] Add Node SDK tests that run in CI, not just a TypeScript build.
- [x] Add stable error codes and user-facing error messages for daemon, SDK, and
  CLI paths.
- [ ] Add config migration tests so future storage/config changes do not break
  existing users silently.
- [x] Add benchmark documentation that separates repeatable results from
  experimental claims.

Acceptance criteria:

- The same operation has equivalent behavior across Python SDK, Node SDK, CLI,
  and REST API.
- Breaking changes are documented and versioned.
- Common user failures return actionable errors.

## P2.5 - Fast, Low-RAM Defaults

Goal: make MemoryKernel feel instant for local agent workflows without forcing
heavy model downloads, high idle memory, or long daemon warmup.

Default product stance:

- `lite` is the default profile for new users.
- `balanced` improves recall quality while staying local and modest.
- `quality` is opt-in for heavier semantic models or advanced vector search.

Implementation plan:

- [x] Choose the default performance strategy: SQLite FTS5 candidate search,
  lightweight hashing rerank, and metadata scoring before any heavy semantic
  model path.
- [x] Add `MEMK_PROFILE=lite|balanced|quality` and expose the active profile in
  `memk health`, MCP `memk_health`, and daemon diagnostics.
- [x] Add SQLite FTS5 indexes for `memories.content` and active fact text,
  using external-content tables or triggers so writes stay simple and search
  stays fast.
- [x] Add a candidate-first retriever:
  1. FTS5 gets top 100-300 candidates.
  2. Hashing embedder reranks only those candidates.
  3. Existing importance, recency, confidence, and fact boosts produce the final
     score.
- [x] Make the candidate-first retriever the default in `lite` and `balanced`
  profiles.
- [x] Keep full RAM vector indexing behind `MEMK_INDEX_MODE=ram`, not as the
  default path.
- [x] Lazy-load or gate graph index hydration so `recall` does not load the full
  graph sidecar unless graph expansion is enabled for the profile.
- [x] Lazy-start background job workers only when the first background job is
  submitted.
- [x] Keep spaCy disabled by default in `lite`; enable only through `.[nlp]` and
  an explicit profile/config switch.
- [x] Add low-memory SQLite pragmas for `lite`, including a smaller cache and
  mmap setting than the current production defaults.
- [ ] Add a benchmark target for 10k and 50k synthetic memories that reports:
  cold startup, FTS candidate latency, rerank latency, end-to-end recall p50/p95,
  RSS delta, and index build cost.
- [x] Add regression tests proving lite runtime does not load RAM indexes, graph
  indexes, or worker threads until they are needed.
- [x] Document performance modes in README, `docs/benchmarking.md`, and
  troubleshooting docs.
- [ ] Evaluate optional vector backends after the FTS path lands:
  `sqlite-vec`/SQLite `vec1` for local vector search, and `fastembed` for a
  lighter semantic model stack.

Performance targets:

- `lite` should use no torch, sentence-transformers, scikit-learn, or spaCy by
  default.
- `lite` recall on 50k synthetic memories should target sub-20ms p95
  end-to-end on a warm local process.
- Added runtime memory for 50k memories should stay close to candidate/cache
  size rather than scaling with all stored vectors.
- Daemon cold start should not hydrate every embedding by default.
- `quality` mode can spend more CPU/RAM, but must be explicit and documented.

## P3 - Reach 90%+: Commercial-Grade Product

Goal: make the project credible for long-running, multi-user, or team usage.

- [x] Add daemon authentication guidance and optional API token enforcement.
- [x] Document network exposure risks and default local-only deployment posture.
- [x] Add structured logs with request IDs for daemon operations.
- [x] Add basic metrics for ingestion count, search latency, storage size, and
  error rates.
- [x] Add backup and restore workflow for local memory stores.
- [x] Add upgrade/downgrade guidance for persisted data.
- [x] Add a small dashboard or TUI health view for storage, index, and daemon
  status.
- [x] Add long-run soak tests for daemon stability and storage growth.
- [x] Add real-world example apps: coding agent memory, local research notebook,
  and support assistant memory.

Acceptance criteria:

- Operators can diagnose daemon health without reading source code.
- Storage can be backed up, restored, and upgraded with documented commands.
- Security posture is explicit enough for a professional deployment review.

## Nice-To-Have After 90%

- [ ] Add hosted docs site with versioned docs.
- [ ] Add benchmark comparison pages against known memory/RAG baselines.
- [ ] Add plugin templates for common agent frameworks.
- [ ] Add optional OpenTelemetry instrumentation.
- [ ] Add a public roadmap grouped by beta, stable, and enterprise-oriented
  capabilities.
- [ ] Expand MCP server beyond the starter tools once CLI/API parity is
  complete.
