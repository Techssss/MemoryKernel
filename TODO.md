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
- [ ] Expand MCP server beyond the four starter tools once CLI/API parity is
  complete.
