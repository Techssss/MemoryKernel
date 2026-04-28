# Upgrade And Downgrade Guide

MemoryKernel is beta software. Back up local memory state before upgrading.

## Before Upgrading

```bash
memk stop
memk backup --output memk-before-upgrade.zip
```

Confirm the current package version:

```bash
python -c "import memk.sdk as s; print(s.__version__)"
```

## Upgrade

For an editable checkout:

```bash
git pull
python -m pip install -e ".[dev]"
python -m pytest -q -rs tests
memk doctor
```

For a released wheel after package publishing is enabled:

```bash
python -m pip install --upgrade memk
memk doctor
```

## Downgrade

Downgrades across schema-changing releases are not guaranteed during beta.

Preferred path:

```bash
memk stop
git checkout <previous-release-tag>
python -m pip install -e ".[dev]"
memk restore memk-before-upgrade.zip --force
memk doctor
```

## Persisted Data Policy

- Schema migrations should be forward-only.
- A backup is the supported rollback path.
- Release notes must call out persisted data changes.
- Do not run multiple MemoryKernel versions against the same `.memk/` state at
  the same time.
