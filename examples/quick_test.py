"""
Quick Test - MemoryKernel
==========================
A simple script to quickly test MemoryKernel functionality.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memk.core.runtime_v2 import get_runtime_v2


async def quick_test():
    """Quick test of core functionality."""
    
    print("🚀 MemoryKernel Quick Test")
    print("=" * 60)
    print()
    
    # 1. Initialize
    print("1️⃣ Initializing...")
    runtime_manager = get_runtime_v2()
    runtime_manager.initialize_global()
    workspace = runtime_manager.get_workspace_runtime("quick-test")
    print(f"✓ Workspace ready (Generation: {workspace.get_generation()})")
    print()
    
    # 2. Add memories
    print("2️⃣ Adding memories...")
    
    embedder = runtime_manager.container.get_embedder()
    
    memories = [
        ("Python is great for AI development", 0.8),
        ("Use type hints for better code quality", 0.7),
        ("FastAPI is excellent for building APIs", 0.6),
        ("SQLite with WAL mode is production-ready", 0.9),
        ("Dependency injection improves testability", 0.8),
    ]
    
    for content, importance in memories:
        embedding = embedder.embed(content)
        mem_id = workspace.db.insert_memory(
            content,
            embedding=embedding,
            importance=importance
        )
        print(f"✓ Added: {content[:40]}... (ID: {mem_id})")
    
    # Bump generation
    new_gen = workspace.bump_generation()
    print(f"✓ Generation bumped to {new_gen}")
    print()
    
    # 3. Search
    print("3️⃣ Searching...")
    query = "How to build good APIs?"
    print(f"Query: {query}")
    print()
    
    results = workspace.retriever.retrieve(query, limit=3)
    
    print(f"Found {len(results)} results:")
    for i, item in enumerate(results, 1):
        print(f"{i}. {item.content}")
        print(f"   Score: {item.score:.3f}, Importance: {item.importance:.2f}")
    print()
    
    # 4. Build context
    print("4️⃣ Building context...")
    context = workspace.builder.build_context(results)
    print("Context:")
    print("-" * 60)
    print(context)
    print("-" * 60)
    print()
    
    # 5. Statistics
    print("5️⃣ Statistics...")
    stats = workspace.db.get_stats()
    diag = workspace.get_diagnostics()
    
    print(f"Total memories: {stats['total_memories']}")
    print(f"Index entries: {diag['index_entries']}")
    print(f"Generation: {diag['generation']}")
    print()
    
    # 6. Test lazy loading
    print("6️⃣ Testing lazy loading...")
    print(f"Components loaded:")
    for comp, loaded in diag['components_loaded'].items():
        status = "✓" if loaded else "✗"
        print(f"  {status} {comp}")
    print()
    
    print("=" * 60)
    print("✅ Quick test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(quick_test())
