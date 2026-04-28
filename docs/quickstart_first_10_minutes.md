# First 10 Minutes With MemoryKernel

This guide starts from a fresh clone and ends with an AI-agent-ready project
memory.

## 1. Install

```bash
git clone https://github.com/Techssss/MemoryKernel.git
cd MemoryKernel
python -m pip install -e ".[dev]"
```

Check the CLI:

```bash
memk --help
```

## 2. Use The Three Core Commands

Run this inside the project you want MemoryKernel to remember:

```bash
cd /path/to/your/project
memk remember "The billing service uses Stripe test mode in local development"
memk recall "billing provider"
memk health
```

MemoryKernel auto-creates local state under `.memk/` on first use. Do not commit
that directory.

## 3. Know What To Store

Store durable project knowledge:

```bash
memk remember "Decision: use PostgreSQL for concurrent writes in billing"
memk remember "Bug fix: 401 locally usually means MEMK_API_TOKEN mismatch"
memk remember "Workflow: run npm test in sdk/nodejs before publishing"
```

Avoid storing temporary observations like "opened README" or "ran tests".

## 4. Build Agent Context

```bash
memk context "What should I know before editing billing?"
```

## 5. Connect An AI Tool

```bash
memk setup claude
memk setup cursor
memk setup vscode
memk setup openclaw
```

Copy the printed snippet into your AI tool. See [Agent Setup](./agent_setup.md)
for the full guide.

Useful prompts:

```text
Remember: the frontend dev server runs on port 5173.
Recall what we know about billing.
Run memk_health and summarize memory status.
```

## 6. Use The Python SDK

```python
from memk.sdk import MemoryKernel

mk = MemoryKernel()
memory_id = mk.remember("User prefers short technical summaries")
results = mk.search("summary preference", limit=3)

print(memory_id)
for item in results:
    print(item.score, item.content)
```

## 7. Optional Daemon Mode

For repeated CLI and SDK use:

```bash
memk serve
```

In another terminal:

```bash
memk status
memk health
```

If you set `MEMK_API_TOKEN`, the CLI and SDK will use it automatically from the
environment.
