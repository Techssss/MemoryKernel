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
| sentence-transformers / torch | Stronger semantic embeddings via `.[semantic]` | No |
| scikit-learn | TF-IDF fallback via `.[tfidf]` | No |
| spaCy package + `en_core_web_sm` | Extractor quality via `.[nlp]` | No |
| GLiNER | Optional entity extraction path | No |

Base install uses a deterministic hashing embedder when the semantic model stack
is unavailable. `MEMK_EMBEDDER=auto` tries semantic embeddings first, then falls
back to hashing. Set `MEMK_EMBEDDER=hashing`, `MEMK_EMBEDDER=tfidf`, or
`MEMK_EMBEDDER=semantic` to choose a backend explicitly.

## Storage

SQLite is the default local store. Workspace state is kept under `.memk/` and is
not intended for Git tracking.
