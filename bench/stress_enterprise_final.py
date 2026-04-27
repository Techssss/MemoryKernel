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

def bulk_insert_memories(db_path, count, batch_size=2000):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF") # Even faster for bench
    
    start = time.perf_counter()
    embedder = MockEmbedder()
    
    for i in range(0, count, batch_size):
        actual_batch = min(batch_size, count - i)
        data = []
        for j in range(actual_batch):
            text = f"Enterprise event {i+j}: logs from cluster {j%20} are normal."
            vec = embedder.embed(text)
            data.append((str(uuid_hex()), text, encode_embedding(vec), 0.5, 1.0, datetime.now(timezone.utc).isoformat()))
        conn.executemany("INSERT INTO memories (id, content, embedding, importance, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?)", data)
        conn.commit()
    elapsed = time.perf_counter() - start
    conn.close()
    return elapsed

def uuid_hex():
    import uuid
    return uuid.uuid4().hex

async def run_final_benchmark(total_items=10000):
    log(f"Final Enterprise Scale Check: {total_items} items")
    
    test_dir = Path("bench_tmp_final").resolve()
    if test_dir.exists(): shutil.rmtree(test_dir, ignore_errors=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    ws = WorkspaceManager(start_path=str(test_dir))
    brain_id = ws.init_workspace().brain_id
    db_path = ws.get_db_path()

    # 1. Ingestion
    ingest_time = bulk_insert_memories(db_path, total_items)
    log(f"Ingestion {total_items} items: {ingest_time:.2f}s ({total_items/ingest_time:.1f} i/s)")

    # 2. Setup Runtime bypassing heavy model
    runtime_mgr = get_runtime()
    runtime_mgr._is_global_initialized = True
    runtime_mgr.shared_embedder = MockEmbedder()
    runtime_mgr.embedder_pipeline = EmbeddingPipeline(runtime_mgr.shared_embedder)
    
    t0 = time.perf_counter()
    runtime = runtime_mgr.get_workspace_runtime(brain_id, db_path)
    startup_time = time.perf_counter() - t0
    log(f"Hydration Time: {startup_time:.2f}s for {len(runtime.index)} items")

    # 3. Query
    service = MemoryKernelService(allow_direct_writes=True)
    latencies = []
    for _ in range(200):
        t1 = time.perf_counter()
        await service.search("cluster 5 logs", workspace_id=brain_id, limit=3)
        latencies.append((time.perf_counter() - t1) * 1000)
    
    log(f"Query Latency (ms): P50={np.percentile(latencies, 50):.2f}, P95={np.percentile(latencies, 95):.2f}")
    
    # 4. RAM
    if psutil is not None:
        process = psutil.Process()
        log(f"RSS RAM: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    else:
        log("RSS RAM: unavailable (psutil not installed)")

    # 5. Accuracy check
    target_text = "PROJECT_ZULU_CORE: System override initialized by Dr. Smith."
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO memories (id, content, embedding, importance) VALUES (?, ?, ?, 1.0)", ("target-z", target_text, encode_embedding(MockEmbedder().embed(target_text))))
    runtime.index.clear()
    runtime._hydrate_index()
    
    res = await service.search("Dr. Smith override", workspace_id=brain_id)
    if any("ZULU_CORE" in r["content"] for r in res.get("results", [])):
        log("Accuracy: SUCCESS")
    else:
        log("Accuracy: FAILED")

if __name__ == "__main__":
    os.environ["MEMK_DAEMON_MODE"] = "1"
    asyncio.run(run_final_benchmark())
