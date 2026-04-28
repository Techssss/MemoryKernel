# Backup And Restore

MemoryKernel stores workspace memory under `.memk/`. Back up this state before
upgrades, destructive experiments, or moving a workspace between machines.

## Create A Backup

```bash
memk backup
```

This writes a timestamped zip archive in the workspace root:

```text
memk-backup-YYYYMMDDTHHMMSSZ.zip
```

Choose a path:

```bash
memk backup --output ~/backups/my-project-memk.zip
```

The archive contains:

- `manifest.json`
- `state/state.db`
- `state/state.db-wal` if present
- `state/state.db-shm` if present

## Restore A Backup

Stop the daemon first:

```bash
memk stop
```

Restore with explicit confirmation:

```bash
memk restore ~/backups/my-project-memk.zip --force
```

Restore replaces the current workspace `.memk` manifest and SQLite state files.

## Safety Notes

- Keep backups private. They can contain sensitive project knowledge.
- Do not restore archives from untrusted sources.
- Restore validates archive paths and only accepts the expected MemoryKernel
  state files.
- After restore, run `memk doctor` to inspect the workspace health.
