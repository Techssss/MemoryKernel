import asyncio
import os
import time
import random
import shutil
import numpy as np
import sys
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from memk.core.service import MemoryKernelService
from memk.core.runtime import get_runtime
from memk.workspace.manager import WorkspaceManager
from memk.core.embedder import BaseEmbedder, EmbeddingPipeline, encode_embedding

try:
    import psutil
except ImportError:  # pragma: no cover - benchmark environment fallback
    psutil = None

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

class MockEmbedder(BaseEmbedder):
    def __init__(self, dim=384): self._dim = dim
    @property
    def dim(self): return self._dim
    def embed(self, text): return np.random.rand(self._dim).astype(np.float32)
    def embed_batch(self, texts): return [self.embed(t) for t in texts]

# ---------------------------------------------------------------------------
# High-Efficiency Bulk Insert (simulating Enterprise Optimized Ingestion)
# ---------------------------------------------------------------------------
def bulk_insert_memories(db_path, count, batch_size=5000):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    
    start = time.perf_counter()
    embedder = MockEmbedder()
    
    for i in range(0, count, batch_size):
        actual_batch = min(batch_size, count - i)
        data = []
        for j in range(actual_batch):
            text = f"Enterprise log {i+j}: transaction {random.randint(0, 1000000)} verified at node {j%10}"
            vec = embedder.embed(text)
            data.append((
                str(os.urandom(8).hex()), # id
                text,
                encode_embedding(vec),
                0.5, # importance
                1.0, # confidence
                datetime.now(timezone.utc).isoformat() # created_at
            ))
        
        conn.executemany(
            "INSERT INTO memories (id, content, embedding, importance, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            data
        )
        conn.commit()
        log(f"  Inserted {i+actual_batch}/{count}...")

    elapsed = time.perf_counter() - start
    conn.close()
    return elapsed

async def run_enterprise_benchmark(total_items=100000):
    log(f"Starting Optimized Enterprise Benchmark: {total_items} items")
    
    test_dir = Path("bench_tmp").resolve()
    if test_dir.exists(): shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    ws = WorkspaceManager(start_path=str(test_dir))
    brain_id = ws.init_workspace().brain_id
    db_path = ws.get_db_path()

    # 1. Bulk Ingestion
    log(f"--- Phase 1: High-Speed Ingestion ({total_items} items) ---")
    ingest_time = bulk_insert_memories(db_path, total_items)
    log(f"Ingestion Complete: {ingest_time:.2f}s (Avg: {total_items/ingest_time:.1f} i/s)")

    # 2. Service Startup & Indexing
    log("\n--- Phase 2: Service Startup & Indexing ---")
    process = psutil.Process(os.getpid()) if psutil is not None else None
    mem_before = process.memory_info().rss / 1024 / 1024 if process else 0.0
    
    start_up = time.perf_counter()
    runtime_mgr = get_runtime()
    # Injected Mock Embedder
    runtime_mgr.shared_embedder = MockEmbedder()
    runtime_mgr.embedder_pipeline = EmbeddingPipeline(runtime_mgr.shared_embedder)
    
    # This will trigger DB read and RAM Index hydration (100k items)
    runtime = runtime_mgr.get_workspace_runtime(brain_id, db_path)
    startup_time = time.perf_counter() - start_up
    
    mem_after = process.memory_info().rss / 1024 / 1024 if process else 0.0
    db_size = os.path.getsize(db_path) / 1024 / 1024
    
    log(f"Startup & Indexing Time: {startup_time:.2f}s")
    log(f"RAM Index Size: {len(runtime.index)} entries")
    if process:
        log(f"RAM Usage: {mem_after:.2f} MB (Delta: {mem_after - mem_before:.2f} MB)")
    else:
        log("RAM Usage: unavailable (psutil not installed)")
    log(f"DB Size: {db_size:.2f} MB")

    # 3. Query Performance
    log("\n--- Phase 3: Query Performance (1000 searches) ---")
    service = MemoryKernelService(allow_direct_writes=True)
    latencies = []
    
    for k in range(1000):
        q = f"transaction {random.randint(0, total_items)}"
        t0 = time.perf_counter()
        await service.search(q, workspace_id=brain_id, limit=5)
        latencies.append((time.perf_counter() - t0) * 1000)
        
    log(f"Query Stats (ms):")
    log(f"  Avg: {np.mean(latencies):.2f}")
    log(f"  P50: {np.percentile(latencies, 50):.2f}")
    log(f"  P95: {np.percentile(latencies, 95):.2f}")
    log(f"  P99: {np.percentile(latencies, 99):.2f}")

    # 4. Accuracy Check
    log("\n--- Phase 4: Accuracy Check ---")
    target_text = "SECRET_CODE_ALFA_999: The vault is under the big oak tree."
    # Direct insert via DB to simulate background ingestion
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO memories (id, content, embedding, importance) VALUES (?, ?, ?, ?)",
            ("target-id", target_text, encode_embedding(MockEmbedder().embed(target_text)), 1.0)
        )
    
    # Small delay for index sync simulation or manual rebuild
    runtime.index.clear()
    runtime._hydrate_index()
    
    log("  Searching for 'vault oak tree'...")
    res = await service.search("vault oak tree", workspace_id=brain_id)
    if any("ALFA_999" in r["content"] for r in res.get("results", [])):
        log("  Accuracy: Success! Found targeted memory in 100k haystack.")
    else:
        log("  Accuracy: Failed.")

if __name__ == "__main__":
    os.environ["MEMK_DAEMON_MODE"] = "1"
    asyncio.run(run_enterprise_benchmark(100000))
