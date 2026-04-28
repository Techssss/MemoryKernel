# Release Process

MemoryKernel release automation is intentionally manual-gated while the project
is beta.

## Python Package

Workflow: `.github/workflows/release-python.yml`

Triggers:

- Manual `workflow_dispatch`
- Git tag matching `v*`

The workflow builds source and wheel distributions, runs `twine check`, uploads
the artifacts, then publishes through the `pypi` GitHub Environment.

Repository maintainers should configure the `pypi` environment with required
reviewers before enabling real publishing.

## Node SDK

Workflow: `.github/workflows/release-nodejs.yml`

Triggers:

- Manual `workflow_dispatch`
- Git tag matching `sdk-nodejs-v*`

The workflow runs Node SDK tests/build, then publishes from `sdk/nodejs` through
the `npm` GitHub Environment using `NPM_TOKEN`.

Repository maintainers should configure the `npm` environment with required
reviewers and add `NPM_TOKEN` as an environment or repository secret.

## Pre-Release Checklist

- Update `CHANGELOG.md`.
- Confirm package versions in `pyproject.toml`, `setup.py`, and
  `sdk/nodejs/package.json`.
- Confirm CI is green on `main`.
- Confirm package smoke tests pass on Linux, macOS, and Windows.
- Create the release tag only after the above checks pass.
