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

Base install defaults to `MEMK_PROFILE=lite`, which uses SQLite FTS5 candidate
search and a deterministic hashing reranker. Set `MEMK_PROFILE=quality` plus
`MEMK_EMBEDDER=semantic` when you want the optional semantic model stack.
`MEMK_EMBEDDER=auto` still tries semantic embeddings first and falls back to
hashing, but it is no longer the default for new low-RAM runs.

## Storage

SQLite is the default local store. Workspace state is kept under `.memk/` and is
not intended for Git tracking.
