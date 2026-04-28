# Troubleshooting

## Installation Fails

Upgrade packaging tools first:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

If optional model dependencies fail, validate the core suite before debugging
model-specific tests:

```bash
python -m pytest -q -rs tests
```

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
