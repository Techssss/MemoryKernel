# Security Policy

MemoryKernel is local-first by default. The daemon binds to `127.0.0.1:15301`
unless the operator changes deployment behavior.

## Supported Versions

| Version | Status |
| --- | --- |
| `0.1.x` | Beta, security fixes accepted |
| `<0.1.0` | Unsupported |

## Reporting A Vulnerability

Do not open a public issue for sensitive vulnerabilities.

Preferred reporting path:

1. Use GitHub private vulnerability reporting if it is enabled for the
   repository.
2. If private reporting is unavailable, contact `dev@memorykernel.dev` with a
   concise description, reproduction steps, affected version, and impact.

Expected response target: 7 days for initial triage.

## Local-First Threat Model

MemoryKernel stores project memory in local workspace state under `.memk/`.
Protect it like source code and local developer secrets.

Primary risks:

- Exposing the daemon outside localhost without authentication.
- Committing `.memk/` state, local SQLite databases, or model cache artifacts.
- Ingesting sensitive Git history and later exposing it through search/context.
- Running untrusted clients against a daemon that can write local memory.

## Daemon Authentication

For shared machines, remote tunnels, or non-localhost deployments, set an API
token:

```bash
export MEMK_API_TOKEN="replace-with-a-long-random-token"
memk serve
```

Clients should send either:

```text
Authorization: Bearer <token>
```

or:

```text
X-Memk-Token: <token>
```

Health endpoints stay public so process supervisors can check liveness.

## Deployment Guidance

- Keep the default bind local unless a deployment review explicitly approves
  broader exposure.
- Use OS-level access control for the workspace directory.
- Rotate `MEMK_API_TOKEN` if it is shared or appears in logs.
- Back up `.memk/state/state.db` before upgrades that change persisted data.
- Treat benchmark and debug exports as potentially sensitive.
