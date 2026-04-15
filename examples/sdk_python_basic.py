#!/usr/bin/env python
"""
Basic Python SDK Example
========================
Demonstrates core MemoryKernel SDK operations.
"""

from memk.sdk import MemoryKernel

def main():
    # Initialize client (auto-detects workspace)
    mk = MemoryKernel()
    
    print("=== MemoryKernel SDK Demo ===\n")
    
    # 1. Remember some facts
    print("1. Adding memories...")
    mk.remember("User prefers TypeScript over JavaScript", importance=0.8)
    mk.remember("Project uses React for frontend", importance=0.7)
    mk.remember("Database is PostgreSQL", importance=0.9)
    print("✓ Added 3 memories\n")
    
    # 2. Search for relevant information
    print("2. Searching for 'What language does user prefer?'...")
    results = mk.search("What language does user prefer?", limit=5)
    for r in results:
        print(f"  [{r.score:.2f}] {r.content}")
    print()
    
    # 3. Build context for LLM
    print("3. Building context...")
    context = mk.context("What should I know about this project?", max_chars=500)
    print(f"Context ({len(context)} chars):")
    print(f"  {context[:200]}...")
    print()
    
    # 4. Check status
    print("4. Workspace status:")
    status = mk.status()
    print(f"  Generation: {status.generation}")
    print(f"  Memories: {status.total_memories}")
    print(f"  Facts: {status.total_facts}")
    print(f"  Watcher: {'Running' if status.watcher_running else 'Stopped'}")
    print()
    
    print("✓ Demo complete!")

if __name__ == "__main__":
    main()
