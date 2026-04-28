# Contributing

MemoryKernel is a beta-stage local-first memory library. Contributions should
keep the core storage, retrieval, API, and SDK behavior predictable.

## Setup

```bash
git clone https://github.com/Techssss/MemoryKernel.git
cd MemoryKernel
python -m pip install -e ".[dev]"
```

Optional Node SDK work:

```bash
cd sdk/nodejs
npm install
npm test
```

## Local Checks

Run the regular Python suite:

```bash
python -m pytest -q -rs tests
```

Run coverage locally:

```bash
python -m pytest -q -rs --cov=memk --cov-report=term-missing tests
```

Build and smoke-test the Python package:

```bash
python -m pip install build
python -m build
python scripts/smoke_install.py
```

## Pull Request Expectations

- Keep changes focused on one behavior or product surface.
- Add or update tests for changed public behavior.
- Update README, TODO, or docs when setup, CLI, API, SDK, or release behavior
  changes.
- Do not commit `.memk/`, local databases, benchmark output, model caches, or
  external comparison snapshots.
- Keep beta limitations explicit. Do not describe experimental behavior as
  production-ready without validation.

## Commit Style

Use short imperative messages:

```text
Add package smoke test
Fix daemon auth header handling
Document release process
```

## Release Changes

Release-related changes should update:

- `CHANGELOG.md`
- `pyproject.toml` and `setup.py` for Python package metadata
- `sdk/nodejs/package.json` for Node SDK metadata
- Relevant docs under `docs/`
