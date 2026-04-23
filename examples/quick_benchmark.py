"""
Quick Performance Benchmark
============================
Fast performance test (< 1 minute) to validate optimizations.
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from memk.core.runtime_v2 import get_runtime_v2


async def quick_benchmark():
    """Quick performance benchmark."""
    
    print("⚡ Quick Performance Benchmark")
    print("=" * 60)
    print()
    
    # Initialize
    print("1️⃣ Initializing...")
    start = time.time()
    runtime_manager = get_runtime_v2()
    runtime_manager.initialize_global()
    workspace = runtime_manager.get_workspace_runtime("quick-bench")
    init_time = time.time() - start
    print(f"✓ Init time: {init_time:.2f}s")
    print()
    
    # Test 1: Batch insertion
    print("2️⃣ Testing batch insertion (100 records)...")
    embedder = runtime_manager.container.get_embedder()
    
    contents = [f"Test memory {i}: Python is great for AI" for i in range(100)]
    
    start = time.time()
    embeddings = embedder.embed_batch(contents)
    
    for content, embedding in zip(contents, embeddings):
        mem_id = workspace.db.insert_memory(content, embedding=embedding)
        
        # Add to index
        from memk.retrieval.index import IndexEntry
        import datetime
        entry = IndexEntry(
            id=mem_id,
            item_type="memory",
            content=content,
            importance=0.5,
            confidence=1.0,
            created_at=datetime.datetime.now().isoformat(),
            decay_score=1.0,
            access_count=0,
        )
        workspace.index.add_entry(entry, embedding)
    
    workspace.bump_generation()
    insert_time = time.time() - start
    ops_per_sec = 100 / insert_time
    
    print(f"✓ Inserted 100 records in {insert_time:.2f}s")
    print(f"  Speed: {ops_per_sec:.0f} ops/sec")
    print(f"  Index size: {len(workspace.index):,}")
    print()
    
    # Test 2: Search performance
    print("3️⃣ Testing search (20 queries)...")
    queries = [
        "Python programming",
        "AI development",
        "Machine learning",
        "Data science",
        "Web development",
    ] * 4
    
    latencies = []
    start = time.time()
    
    for query in queries:
        query_start = time.time()
        results = workspace.retriever.retrieve(query, limit=5)
        latency = (time.time() - query_start) * 1000
        latencies.append(latency)
    
    search_time = time.time() - start
    qps = len(queries) / search_time
    avg_latency = sum(latencies) / len(latencies)
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    
    print(f"✓ Completed {len(queries)} searches in {search_time:.2f}s")
    print(f"  QPS: {qps:.1f}")
    print(f"  Avg latency: {avg_latency:.1f}ms")
    print(f"  P50: {p50:.1f}ms")
    print(f"  P95: {p95:.1f}ms")
    print()
    
    # Test 3: Cache test
    print("4️⃣ Testing cache...")
    
    # First query (cache miss)
    start = time.time()
    results1 = workspace.retriever.retrieve("Python programming", limit=5)
    first_time = (time.time() - start) * 1000
    
    # Second query (should be cached)
    start = time.time()
    results2 = workspace.retriever.retrieve("Python programming", limit=5)
    second_time = (time.time() - start) * 1000
    
    speedup = first_time / second_time if second_time > 0 else 0
    
    print(f"✓ Cache test:")
    print(f"  First query: {first_time:.1f}ms")
    print(f"  Second query: {second_time:.1f}ms")
    print(f"  Speedup: {speedup:.1f}x")
    print()
    
    # Summary
    print("=" * 60)
    print("📊 Summary")
    print("=" * 60)
    print(f"Insertion: {ops_per_sec:.0f} ops/sec {'✓' if ops_per_sec > 20 else '✗'}")
    print(f"Search P50: {p50:.1f}ms {'✓' if p50 < 15 else '✗'}")
    print(f"Search P95: {p95:.1f}ms {'✓' if p95 < 50 else '✗'}")
    print(f"QPS: {qps:.1f} {'✓' if qps > 20 else '✗'}")
    print(f"Cache speedup: {speedup:.1f}x {'✓' if speedup > 2 else '✗'}")
    print()
    
    # Overall
    passed = sum([
        ops_per_sec > 20,
        p50 < 15,
        p95 < 50,
        qps > 20,
        speedup > 2,
    ])
    
    print(f"Overall: {passed}/5 tests passed")
    print()


if __name__ == "__main__":
    asyncio.run(quick_benchmark())
