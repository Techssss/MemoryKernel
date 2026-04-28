# First 10 Minutes With MemoryKernel

This guide starts from a fresh clone and ends with a working memory search.

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

## 2. Initialize A Workspace

Run this inside the project you want MemoryKernel to remember:

```bash
cd /path/to/your/project
memk init
```

This creates local state under `.memk/`. Do not commit that directory.

## 3. Add Memory

```bash
memk add "The billing service uses Stripe test mode in local development"
```

`memk remember` is an alias for the same operation:

```bash
memk remember "The frontend dev server runs on port 5173"
```

## 4. Search Memory

```bash
memk search "billing provider"
```

Build a compact context block for an agent or LLM prompt:

```bash
memk context "What should I know before editing billing?"
```

## 5. Use The Python SDK

```python
from memk.sdk import MemoryKernel

mk = MemoryKernel()
memory_id = mk.remember("User prefers short technical summaries")
results = mk.search("summary preference", limit=3)

print(memory_id)
for item in results:
    print(item.score, item.content)
```

## 6. Optional Daemon Mode

For repeated CLI and SDK use:

```bash
memk serve
```

In another terminal:

```bash
memk status
memk doctor
```

If you set `MEMK_API_TOKEN`, the CLI and SDK will use it automatically from the
environment.
