# MemoryKernel REST API v1

Base URL:

```text
http://127.0.0.1:15301/v1
```

All primary endpoints return:

```json
{
  "data": {},
  "metadata": {
    "workspace_id": "default",
    "generation": 0,
    "cache_hit": false,
    "degraded": false,
    "stale_warning": null,
    "timestamp": "2026-04-28T00:00:00+00:00"
  }
}
```

## Authentication

Authentication is disabled unless `MEMK_API_TOKEN` is set on the daemon.

When enabled, send:

```text
Authorization: Bearer <token>
```

or:

```text
X-Memk-Token: <token>
```

## Endpoints

### `GET /v1/health`

Returns liveness information.

### `POST /v1/remember`

Request:

```json
{
  "content": "User prefers TypeScript",
  "importance": 0.5,
  "confidence": 1.0,
  "workspace_id": "optional-workspace-id"
}
```

Response data:

```json
{
  "id": "memory-id",
  "extracted_facts": []
}
```

### `POST /v1/search`

Request:

```json
{
  "query": "language preference",
  "limit": 10,
  "workspace_id": "optional-workspace-id",
  "client_generation": 1
}
```

Response data:

```json
{
  "results": []
}
```

### `POST /v1/context`

Request:

```json
{
  "query": "what should the agent know?",
  "max_chars": 500,
  "threshold": 0.3,
  "workspace_id": "optional-workspace-id",
  "client_generation": 1
}
```

Response data:

```json
{
  "context": "...",
  "char_count": 123
}
```

### `GET /v1/status`

Query params:

```text
workspace_id=optional-workspace-id
```

Returns workspace generation, initialization status, storage stats, and watcher
status.

### `POST /v1/ingest/git`

Request:

```json
{
  "limit": 50,
  "since": "2026-01-01",
  "branch": "HEAD",
  "workspace_id": "optional-workspace-id"
}
```

### `GET /v1/metrics`

Returns request counts, latency percentiles, cache metrics, error/degraded rates,
database size, job counts, and trace summaries.

## Error Shape

Versioned API errors use a stable `detail.code` and user-facing message:

```json
{
  "detail": {
    "code": "search_failed",
    "message": "description of the failure"
  }
}
```

Current internal error codes:

- `remember_failed`
- `search_failed`
- `context_failed`
- `status_failed`
- `ingest_git_failed`
- `metrics_failed`
- `auth_required`

## Legacy Endpoint Deprecation

Unversioned daemon endpoints remain for compatibility, but primary integrations
should use `/v1`. Legacy responses include:

```text
Deprecation: true
Link: </v1/search>; rel="successor-version"
```
