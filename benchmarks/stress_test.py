import time
import os
import json
import asyncio
import random
import statistics
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from memk.core.service import MemoryKernelService

async def stress_test():
    # Setup
    db_path = "stress_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    os.environ["MEMK_DB_PATH"] = db_path
    
    # Reset singletons
    from memk.core.runtime import RuntimeManager
    RuntimeManager._instance = None
    
    service = MemoryKernelService()
    service.ensure_initialized()
    
    print("=== STARTING STRESS TEST (Large Data + High Concurrency) ===")
    
    # 1. Bulk Insertion (Sequential for now to build base)
    insert_count = 500
    print(f"\n[1/3] Bulk Inserting {insert_count} memories...")
    start_bulk = time.perf_counter()
    for i in range(insert_count):
        await service.add_memory(
            content=f"Stress test memory chunk {i}: Random data to fill the index. System must scale.",
            importance=random.random(),
            confidence=0.8
        )
        if (i+1) % 100 == 0:
            print(f"    Inserted {i+1}/{insert_count}...")
    
    bulk_time = time.perf_counter() - start_bulk
    print(f"  Bulk insertion took {bulk_time:.1f}s (avg {bulk_time*1000/insert_count:.1f}ms/item)")
    
    # 2. Concurrent Read Load
    print(f"\n[2/3] Simulating High Concurrency (100 total requests in batches of 20)...")
    queries = ["system performance", "scaling", "stress test", "latency", "memory index"]
    
    async def task(tid):
        q = random.choice(queries)
        start = time.perf_counter()
        # Mix of search and context
        if tid % 2 == 0:
            await service.search(q, limit=10)
        else:
            await service.build_context(q, max_chars=1000)
        return (time.perf_counter() - start) * 1000

    start_concurrent = time.perf_counter()
    tasks = [task(i) for i in range(100)] 
    batch_size = 20
    latencies = []
    for i in range(0, len(tasks), batch_size):
        results = await asyncio.gather(*tasks[i:i+batch_size])
        latencies.extend(results)
        print(f"    Batch {i//batch_size + 1} complete...")

    concurrent_time = time.perf_counter() - start_concurrent
    print(f"  Concurrent workload finished in {concurrent_time:.2f}s")
    print(f"  P50: {statistics.median(latencies):.2f}ms")
    print(f"  P95: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")
    print(f"  P99: {max(latencies):.2f}ms")
    
    # 3. Telemetry Check
    telem = service.get_embedding_telemetry()
    concurrency = telem.get("concurrency", {})
    report = service.get_tail_latency_report()
    
    print(f"\n[3/3] System Health Check:")
    print(f"  Final Index Size: {len(service.runtime.index)}")
    print(f"  Slow request rate: {report.get('slow_rate_pct', 0)}%")
    print(f"  Is Busy:         {concurrency.get('is_busy')}")
    
    # Show degradation reasons if any
    slow_paths = report.get("top_slow_paths", [])
    if slow_paths:
        print("\n  Degradation Log:")
        for sp in slow_paths:
            if sp.get('degraded'):
                print(f"    - {sp['operation']}: {sp['degradation_reason']} ({sp['count']} times)")

if __name__ == "__main__":
    asyncio.run(stress_test())
