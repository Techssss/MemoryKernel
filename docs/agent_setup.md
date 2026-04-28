# Agent Setup

MemoryKernel is easiest to use when the agent can call memory tools directly.
Use the MCP server for agent integrations and the CLI for direct manual use.
The examples below all launch the same local stdio server: `memk-mcp`.

## Tool Surface

MemoryKernel starts with four MCP tools:

| Tool | When the agent should use it |
| --- | --- |
| `memk_remember` | After a decision, bug fix, project fact, workflow, or user preference should survive the session |
| `memk_recall` | When the agent needs prior project knowledge |
| `memk_context` | At the start of a task, before editing code, or before answering a project-specific question |
| `memk_health` | When checking whether memory is initialized, indexed, and useful |

The server command is:

```bash
memk-mcp
```

First use auto-creates local `.memk/` state in the current project workspace.

## Claude Code

Install MemoryKernel in the Python environment used by Claude Code:

```bash
python -m pip install -e ".[dev]"
```

Add the MCP server:

```bash
claude mcp add --transport stdio memorykernel --scope user -- memk-mcp
```

Project-local config alternative:

```json
{
  "mcpServers": {
    "memorykernel": {
      "command": "memk-mcp",
      "args": []
    }
  }
}
```

Useful prompts:

```text
Remember: the billing service owns invoice numbering.
Recall what we decided about billing.
Run memk_health and summarize memory status.
```

## Cursor

Add to global `~/.cursor/mcp.json` or project `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "memorykernel": {
      "command": "memk-mcp",
      "args": []
    }
  }
}
```

Restart Cursor after changing MCP config.

## VS Code

Add this to workspace `.vscode/mcp.json` or the user MCP config:

```json
{
  "servers": {
    "memorykernel": {
      "type": "stdio",
      "command": "memk-mcp",
      "args": []
    }
  }
}
```

For direct CLI use inside VS Code terminals:

```bash
memk remember "The frontend uses Vite"
memk recall "frontend build tool"
memk health
```

## OpenClaw

Register MemoryKernel as an outbound MCP server:

```bash
openclaw mcp set memorykernel '{"command":"memk-mcp"}'
```

Equivalent config shape:

```json
{
  "mcp": {
    "servers": {
      "memorykernel": {
        "command": "memk-mcp"
      }
    }
  }
}
```

The important runtime contract is that the client launches `memk-mcp` over
stdio. OpenClaw's native memory plugin can continue to coexist with this MCP
server if you want both Markdown memory and MemoryKernel recall.

## Direct CLI

The manual path uses the same product concepts:

```bash
memk remember "We chose PostgreSQL for concurrent writes"
memk recall "database decision"
memk context "What should I know before changing persistence?"
memk health
```

Run the daemon for repeated SDK or REST use:

```bash
memk serve
```

## Client References

- [Claude Code MCP](https://docs.claude.com/en/docs/claude-code/mcp)
- [Cursor MCP](https://docs.cursor.com/cli/mcp)
- [VS Code MCP configuration](https://code.visualstudio.com/docs/copilot/reference/mcp-configuration)
- [OpenClaw MCP](https://docs.openclaw.ai/cli/mcp)
