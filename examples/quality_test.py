import asyncio
import sys
import time
import os
import random
import logging
from pathlib import Path

# Configure logging to see system logs during test
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("quality_test")

# Ensure we can import memk
sys.path.insert(0, str(Path(__file__).parent.parent))

from memk.core.runtime_v2 import get_runtime_v2
from memk.retrieval.index import IndexEntry

async def run_quality_test():
    workspace_id = "super-stress" # Use the 100k DB we just created
    print("=" * 60)
    print("🧠 MEMORY KERNEL - QUALITY & HYDRATION TEST")
    print("=" * 60)

    # 1. HYDRATION TEST (Cold Start)
    print("\n[1/3] Testing Hydration (Load 100k records from DB into RAM)...")
    start_h = time.perf_counter()
    
    runtime_manager = get_runtime_v2()
    # Force initialize global components (embedder, etc)
    runtime_manager.initialize_global()
    
    # Getting workspace runtime triggers hydration if it's not loaded
    workspace = runtime_manager.get_workspace_runtime(workspace_id)
    
    print("[*] Accessing index (this will trigger hydration)...")
    # Ensure index is fully loaded from DB
    # In V2, the index is hydrated on first access or during get_workspace_runtime
    count = len(workspace.index)
    duration_h = time.perf_counter() - start_h
    
    print(f"✓ Hydrated {count:,} records in {duration_h:.2f}s")
    print(f"  > Speed: {count/duration_h:,.0f} records/sec")

    # 2. ACCURACY TEST: THE NEEDLE IN A HAYSTACK
    print("\n[2/3] Testing Search Accuracy (The Needle in a Haystack)...")
    
    # Hide a very specific "needle"
    needle_content = "The secret password for the antigravity engine is 'Quantum-Leap-2026-Alpha'."
    needle_vec = runtime_manager.container.get_embedder().embed(needle_content)
    
    print("[*] Inserting a 'needle' memory into the 100k haystack...")
    mem_id = workspace.db.insert_memory(needle_content, embedding=needle_vec, importance=1.0)
    # Add to index too (as we are bypasssing service layer for speed)
    from datetime import datetime, timezone
    entry = IndexEntry(
        id=mem_id, item_type="memory", content=needle_content,
        importance=1.0, confidence=1.0, created_at=datetime.now(timezone.utc).isoformat(),
        decay_score=1.0, access_count=0
    )
    workspace.index.add_entry(entry, needle_vec)
    
    test_queries = [
        ("What is the secret password?", "Semantic Search"),
        ("antigravity engine Quantum-Leap", "Hybrid Search"),
        ("Quantum-Leap-2026", "Lexical Search")
    ]
    
    for query, test_type in test_queries:
        print(f"\n🔍 Testing {test_type}: '{query}'")
        start_q = time.perf_counter()
        # Use the real retriever
        results = workspace.retriever.retrieve(query, limit=5)
        duration_q = (time.perf_counter() - start_q) * 1000
        
        found = False
        for i, res in enumerate(results):
            # In V2, retriever returns a list of results. Check structure:
            # Usually results are IndexEntry or dict depending on implementation.
            # Assuming IndexEntry objects based on retriever implementation.
            content = res.content if hasattr(res, 'content') else res.get('content', '')
            score = res.score if hasattr(res, 'score') else 0
            
            if "Quantum-Leap" in content:
                print(f"  ✅ FOUND at rank {i+1} (Score: {score:.4f}, Time: {duration_q:.2f}ms)")
                found = True
                break
        
        if not found:
            print(f"  ❌ NOT FOUND in top 5 (Time: {duration_q:.2f}ms)")

    # 3. RANKING TEST: IMPORTANCE vs RECENCY
    print("\n[3/3] Testing Ranking Factors (Importance vs Decay)...")
    
    # Add an old important memory vs a new unimportant one
    # Note: DB created_at has microsecond precision now
    workspace.db.insert_memory("Critical core logic: Always check the safety valve.", importance=1.0)
    workspace.db.insert_memory("Random noise: the weather is nice.", importance=0.1)
    
    print("🔍 Searching for 'logic'...")
    results = workspace.retriever.retrieve("logic", limit=3)
    for i, res in enumerate(results):
        content = res.content if hasattr(res, 'content') else res.get('content', '')
        imp = res.importance if hasattr(res, 'importance') else 0
        print(f"  Rank {i+1}: {content[:50]}... (Imp: {imp})")

    print("\n" + "=" * 60)
    print("QUALITY TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_quality_test())
