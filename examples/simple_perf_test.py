"""
Simple Performance Test
========================
Minimal test to validate core performance.
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memk.core.runtime_v2 import get_runtime_v2


def test_performance():
    """Simple performance test."""
    
    print("🔬 Simple Performance Test")
    print("=" * 60)
    print()
    
    # Initialize with fresh workspace
    workspace_id = f"perf-test-{int(time.time())}"
    print(f"Workspace: {workspace_id}")
    print()
    
    runtime_manager = get_runtime_v2()
    runtime_manager.initialize_global()
    workspace = runtime_manager.get_workspace_runtime(workspace_id)
    
    embedder = runtime_manager.container.get_embedder()
    
    # Test 1: Single insertion
    print("1️⃣ Single insertion test...")
    content = "Python is great for AI development"
    
    start = time.time()
    embedding = embedder.embed(content)
    embed_time = (time.time() - start) * 1000
    
    start = time.time()
    mem_id = workspace.db.insert_memory(content, embedding=embedding)
    insert_time = (time.time() - start) * 1000
    
    print(f"  Embed time: {embed_time:.1f}ms")
    print(f"  Insert time: {insert_time:.1f}ms")
    print(f"  Total: {embed_time + insert_time:.1f}ms")
    print()
    
    # Test 2: Batch insertion (10 items)
    print("2️⃣ Batch insertion test (10 items)...")
    contents = [f"Memory {i}: AI and ML" for i in range(10)]
    
    start = time.time()
    embeddings = embedder.embed_batch(contents)
    batch_embed_time = (time.time() - start) * 1000
    
    start = time.time()
    for content, embedding in zip(contents, embeddings):
        workspace.db.insert_memory(content, embedding=embedding)
    batch_insert_time = (time.time() - start) * 1000
    
    total_time = batch_embed_time + batch_insert_time
    ops_per_sec = 10 / (total_time / 1000)
    
    print(f"  Batch embed time: {batch_embed_time:.1f}ms")
    print(f"  Batch insert time: {batch_insert_time:.1f}ms")
    print(f"  Total: {total_time:.1f}ms")
    print(f"  Speed: {ops_per_sec:.1f} ops/sec")
    print()
    
    # Test 3: Search without index
    print("3️⃣ Search test (DB scan)...")
    query = "AI development"
    
    start = time.time()
    results = workspace.retriever.retrieve(query, limit=5)
    search_time = (time.time() - start) * 1000
    
    print(f"  Search time: {search_time:.1f}ms")
    print(f"  Results: {len(results)}")
    print()
    
    # Test 4: Check index
    print("4️⃣ Index status...")
    print(f"  Index size: {len(workspace.index)}")
    print(f"  DB memories: {workspace.db.get_stats()['total_memories']}")
    print()
    
    # Summary
    print("=" * 60)
    print("📊 Results")
    print("=" * 60)
    print(f"Single insert: {embed_time + insert_time:.1f}ms")
    print(f"Batch speed: {ops_per_sec:.1f} ops/sec")
    print(f"Search time: {search_time:.1f}ms")
    print(f"Index usage: {'Yes' if len(workspace.index) > 0 else 'No'}")
    print()


if __name__ == "__main__":
    test_performance()
