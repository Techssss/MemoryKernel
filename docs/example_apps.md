# Real-World Example Apps

Examples live under `examples/`.

## Coding Agent Memory

File: `examples/ai_coding_assistant.py`

Use case:

- Remember architecture decisions.
- Retrieve project-specific implementation hints.
- Build context before editing code.

## Local Research Notebook

File: `examples/research_notebook_memory.py`

Use case:

- Store paper notes and claims.
- Search related findings.
- Compile a short context window before writing.

## Support Assistant Memory

File: `examples/support_assistant_memory.py`

Use case:

- Remember resolved support cases.
- Search for similar incidents.
- Build a response context for a new ticket.

## SDK And Workflow Examples

- `examples/sdk_python_basic.py`: minimal Python SDK usage.
- `examples/sdk_python_agent.py`: agent-oriented Python SDK flow.
- `examples/local_agent_demo.py`: direct local agent memory flow.
- `examples/git_ingestion_demo.py`: ingest recent Git history.
- `examples/terminal_workflow.sh`: shell workflow for CLI usage.

## Running Examples

Start the daemon first:

```bash
memk serve
```

Then run an example:

```bash
python examples/research_notebook_memory.py
```

Examples assume local daemon access. If `MEMK_API_TOKEN` is set for the daemon,
set the same environment variable before running the example.
