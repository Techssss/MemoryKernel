# MCP Tools

MemoryKernel exposes a small MCP-first surface for agents. The goal is to make
the first integration obvious before adding advanced tools.

## Running The Server

```bash
memk-mcp
```

The server uses MCP's stdio transport. It auto-initializes `.memk/` state in the
current workspace when a memory tool first needs storage.

## `memk_remember`

Store a project memory.

Input:

```json
{
  "content": "We chose PostgreSQL for concurrent writes",
  "importance": 0.8,
  "confidence": 1.0
}
```

Use after:

- A technical decision.
- A bug root cause and fix.
- A user preference.
- A reusable project fact.
- A workflow the agent should repeat.

## `memk_recall`

Recall project memory.

Input:

```json
{
  "query": "database decision",
  "limit": 5
}
```

Returns ranked memory text with scores.

## `memk_context`

Build compact context for an agent prompt.

Input:

```json
{
  "query": "What should I know before editing persistence?",
  "max_chars": 1200,
  "threshold": 0.3
}
```

Use at task start or before answering a project-specific question.

## `memk_health`

Return memory health and next actions.

Input:

```json
{}
```

Returns:

- Grade.
- Memory/fact counts.
- Embedding coverage.
- Database size.
- Suggested next action.

## Agent Guidance

Agents should not store every file read or trivial observation. Store compact,
durable facts:

```text
Good: "Decision: use PostgreSQL for concurrent writes in sync service."
Good: "Bug fix: auth 401 was caused by mismatched MEMK_API_TOKEN."
Weak: "Read file README.md."
Weak: "Ran tests."
```

Prefer memories between 50 and 300 characters. Split large summaries into
separate focused memories.
