# Troubleshooting

## Installation Fails

Upgrade packaging tools first:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

The base install avoids torch, sentence-transformers, scikit-learn, and spaCy.
Install model extras only when you need them:

```bash
python -m pip install -e ".[semantic]"
python -m pip install -e ".[tfidf]"
python -m pip install -e ".[nlp]"
```

If optional model dependencies fail, validate the core suite before debugging
model-specific behavior:

```bash
python -m pytest -q -rs tests
```

## Model Loading Or RAM Use Is High

Use the default low-memory profile for fast startup:

```bash
export MEMK_PROFILE=lite
export MEMK_INDEX_MODE=sqlite
```

On PowerShell:

```powershell
$env:MEMK_PROFILE = "lite"
$env:MEMK_INDEX_MODE = "sqlite"
```

`lite` uses SQLite FTS5 candidate search plus deterministic hashing rerank. It
does not load torch, sentence-transformers, scikit-learn, spaCy, the RAM vector
index, or the graph sidecar by default.

For stronger semantic recall, install `.[semantic]` and opt in explicitly:

```bash
export MEMK_PROFILE=quality
export MEMK_EMBEDDER=semantic
```

Set `MEMK_INDEX_MODE=ram` only when you want the daemon to keep a warm in-memory
vector index.

## spaCy Tests Are Skipped

Some extractor tests require the `en_core_web_sm` model. Skips are expected when
that model is not installed.

Install it only when you need spaCy extractor validation:

```bash
python -m spacy download en_core_web_sm
```

## SQLite Permission Errors

MemoryKernel writes local state under `.memk/state/`.

Check:

- The workspace directory is writable.
- The daemon is not running under a different user.
- Old test temp directories are not owned by another process.
- Antivirus or file-sync tooling is not locking SQLite WAL files.

## Daemon Does Not Start

Check whether the port is already in use:

```bash
memk status
```

Then stop any existing daemon:

```bash
memk stop
memk serve
```

## API Token Errors

If `MEMK_API_TOKEN` is set for the daemon, clients must use the same token.

For CLI:

```bash
export MEMK_API_TOKEN="same-token-as-daemon"
memk search "deployment notes"
```

For Python:

```python
from memk.sdk import MemoryKernel

mk = MemoryKernel(api_token="same-token-as-daemon")
```

## Build Module Missing

Install the build frontend:

```bash
python -m pip install build
python -m build
```

CI runs the package build and smoke install automatically.
