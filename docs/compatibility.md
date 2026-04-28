# Compatibility Matrix

## Python

| Python | Status |
| --- | --- |
| 3.10 | Supported in CI |
| 3.11 | Supported in CI and package smoke tests |
| 3.12 | Not yet part of the supported matrix |

## Operating Systems

| OS | Status |
| --- | --- |
| Linux | Supported in CI |
| macOS | Package smoke tested in CI |
| Windows | Package smoke tested in CI |

## Node.js SDK

| Runtime | Status |
| --- | --- |
| Node.js 20 | Supported in CI |

## Optional Dependencies

| Dependency | Purpose | Required |
| --- | --- | --- |
| spaCy model `en_core_web_sm` | Extractor quality tests | No |
| GLiNER | Optional entity extraction path | No |
| sentence-transformers / torch | Production semantic embeddings | Yes for full semantic quality |

## Storage

SQLite is the default local store. Workspace state is kept under `.memk/` and is
not intended for Git tracking.
