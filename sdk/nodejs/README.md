# @memk/sdk

MemoryKernel SDK for Node.js - Local-first project memory for AI applications.

## Installation

```bash
npm install @memk/sdk
```

Or with yarn:

```bash
yarn add @memk/sdk
```

## Quick Start

```typescript
import { MemoryKernel } from '@memk/sdk';

// Initialize
const mk = new MemoryKernel();

// Remember something
await mk.remember("User prefers TypeScript", { importance: 0.8 });

// Search
const results = await mk.search("What does user prefer?");
results.forEach(r => {
  console.log(`${r.score.toFixed(2)}: ${r.content}`);
});

// Build context for LLM
const context = await mk.context("What should I know?", {
  maxChars: 500
});
console.log(context);
```

## Prerequisites

1. Install MemoryKernel:
```bash
pip install memk
```

2. Initialize workspace:
```bash
cd /path/to/project
memk init
```

3. Start daemon:
```bash
memk serve
```

## API

### Constructor

```typescript
new MemoryKernel(options?: MemoryKernelOptions)
```

**Options:**
- `daemonUrl?: string` - Daemon URL (default: `http://localhost:8734`)
- `workspaceId?: string` - Workspace ID (auto-detected if not provided)

### remember()

Add a memory to the workspace.

```typescript
await mk.remember(content: string, options?: RememberOptions): Promise<string>
```

**Options:**
- `importance?: number` - Priority (0-1, default: 0.5)
- `confidence?: number` - Certainty (0-1, default: 1.0)

**Returns:** Memory ID

### search()

Search for relevant memories.

```typescript
await mk.search(query: string, options?: SearchOptions): Promise<MemoryItem[]>
```

**Options:**
- `limit?: number` - Max results (default: 10)

**Returns:** Array of `MemoryItem` objects

### context()

Build RAG context from memories.

```typescript
await mk.context(query: string, options?: ContextOptions): Promise<string>
```

**Options:**
- `maxChars?: number` - Max context length (default: 500)
- `threshold?: number` - Relevance threshold (default: 0.3)

**Returns:** Formatted context string

### status()

Get workspace status.

```typescript
await mk.status(): Promise<WorkspaceStatus>
```

**Returns:** `WorkspaceStatus` object with:
- `workspace_id: string`
- `generation: number`
- `initialized: boolean`
- `workspace_root: string`
- `total_memories: number`
- `total_facts: number`
- `watcher_running: boolean`

### ingestGit()

Ingest knowledge from Git history.

```typescript
await mk.ingestGit(options?: IngestGitOptions): Promise<IngestGitResult>
```

**Options:**
- `limit?: number` - Number of commits (default: 50)
- `since?: string` - Date filter (YYYY-MM-DD)
- `branch?: string` - Branch name (default: HEAD)

**Returns:** Object with `ingested_count` and `categories`

## Types

### MemoryItem

```typescript
interface MemoryItem {
  item_type: string;
  id: string;
  content: string;
  score: number;
  importance: number;
  confidence: number;
  created_at: string;
  access_count?: number;
  decay_score?: number;
}
```

### WorkspaceStatus

```typescript
interface WorkspaceStatus {
  workspace_id: string;
  generation: number;
  initialized: boolean;
  workspace_root: string;
  total_memories: number;
  total_facts: number;
  watcher_running: boolean;
}
```

## Examples

### Basic Usage

```typescript
import { MemoryKernel } from '@memk/sdk';

const mk = new MemoryKernel();

// Add memories
await mk.remember("User prefers dark mode");
await mk.remember("Project uses React");

// Search
const results = await mk.search("What UI framework?");
console.log(results[0].content); // "Project uses React"

// Get status
const status = await mk.status();
console.log(`Memories: ${status.total_memories}`);
```

### Agent Integration

```typescript
import { MemoryKernel } from '@memk/sdk';

class Agent {
  private memory: MemoryKernel;

  constructor() {
    this.memory = new MemoryKernel();
  }

  async process(userInput: string): Promise<string> {
    // Get context
    const context = await this.memory.context(userInput, {
      maxChars: 1000
    });

    // Generate response with LLM
    const response = await llm.generate(userInput, { context });

    // Remember interaction
    await this.memory.remember(`User asked: ${userInput}`);

    return response;
  }
}
```

### CLI Tool

```typescript
import { Command } from 'commander';
import { MemoryKernel } from '@memk/sdk';

const program = new Command();
const mk = new MemoryKernel();

program
  .command('remember <text>')
  .action(async (text) => {
    await mk.remember(text);
    console.log('✓ Remembered');
  });

program
  .command('search <query>')
  .action(async (query) => {
    const results = await mk.search(query);
    results.forEach(r => {
      console.log(`[${r.score.toFixed(2)}] ${r.content}`);
    });
  });

program.parse();
```

## Features

- ✅ TypeScript-first with full type definitions
- ✅ Promise-based async API
- ✅ Auto-discovery of daemon
- ✅ Generation tracking for staleness detection
- ✅ Local-first (no cloud dependencies)
- ✅ Project-scoped memory isolation

## Troubleshooting

### Cannot connect to daemon

```
Error: Cannot connect to daemon at http://localhost:8734
```

**Solution:** Start the daemon:
```bash
memk serve
```

### Workspace not initialized

```
Error: Workspace not initialized
```

**Solution:** Initialize workspace:
```bash
memk init
```

## License

MIT

## Links

- [Documentation](../../docs/QUICKSTART.md)
- [Examples](./examples/)
- [GitHub](https://github.com/memk/memk)

