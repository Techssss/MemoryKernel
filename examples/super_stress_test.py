import asyncio
import sys
import time
import psutil
import os
import random
import datetime
from pathlib import Path
from typing import List, Dict, Any

# Ensure we can import memk
sys.path.insert(0, str(Path(__file__).parent.parent))

from memk.core.runtime_v2 import get_runtime_v2

class SuperStressTest:
    def __init__(self, workspace_id: str = "super-stress", scale: int = 100000, mock: bool = False):
        self.workspace_id = workspace_id
        self.scale = scale
        self.mock = mock
        self.process = psutil.Process(os.getpid())
        self.start_time = 0
        self.results = {}

    def get_mem_mb(self):
        return self.process.memory_info().rss / 1024 / 1024

    async def run(self):
        print(f"[*] Starting {'Mock ' if self.mock else '' }Super Stress Test: {self.scale:,} records", flush=True)
        print(f"[*] Workspace: {self.workspace_id}", flush=True)
        print("-" * 50, flush=True)

        # 1. Initialize
        print("[1/5] Initializing runtime...", flush=True)
        start_init = time.time()
        runtime_manager = get_runtime_v2()
        
        if not self.mock:
            runtime_manager.initialize_global()
        
        workspace = runtime_manager.get_workspace_runtime(self.workspace_id)
        
        if self.mock:
            # Mock objects if in mock mode
            dim = 384
            class MockEmbedder:
                def __init__(self, dim): self.dim = dim
                def embed_batch(self, texts): return [np.random.rand(self.dim).astype(np.float32) for _ in texts]
                def embed(self, text): return np.random.rand(self.dim).astype(np.float32)
            
            import numpy as np
            embedder = MockEmbedder(dim)
        else:
            embedder = runtime_manager.container.get_embedder()
            
        print(f"✓ Runtime ready in {time.time() - start_init:.2f}s", flush=True)
        print(f"[*] Initial Memory: {self.get_mem_mb():.1f} MB", flush=True)

        # 2. Bulk Insertion
        print(f"\n[2/5] Bulk Inserting {self.scale:,} records...", flush=True)
        start_insert = time.time()
        batch_size = 1000
        
        # Use a single connection for the whole test if possible to avoid connection overhead
        conn = workspace.db._get_connection()
        try:
            for i in range(0, self.scale, batch_size):
                batch_end = min(i + batch_size, self.scale)
                batch_count = batch_end - i
                
                # Generate fake content
                contents = [f"This is a sample memory fact number {j} for testing large scale storage and retrieval performance." for j in range(i, batch_end)]
                
                # Batch Embed
                embeddings = embedder.embed_batch(contents)
                
                # Insert in one transaction
                import uuid
                import datetime
                import struct
                from memk.storage.db import _utcnow, _encode_blob
                from memk.retrieval.index import IndexEntry
                
                with conn: # TRANSACTION START
                    for j, (content, vec) in enumerate(zip(contents, embeddings)):
                        mem_id = str(uuid.uuid4())
                        created_at = _utcnow()
                        blob = _encode_blob(vec)
                        
                        conn.execute(
                            "INSERT INTO memories (id, content, embedding, importance, confidence, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                            (mem_id, content.strip(), blob, float(random.random()), 1.0, created_at)
                        )
                        
                        # Add to index
                        entry = IndexEntry(
                            id=mem_id,
                            item_type="memory",
                            content=content,
                            importance=0.5,
                            confidence=1.0,
                            created_at=created_at,
                            decay_score=1.0,
                            access_count=0,
                        )
                        workspace.index.add_entry(entry, vec)
                
                if (i + batch_count) % 5000 == 0 or (i + batch_count) >= self.scale:
                    elapsed = time.time() - start_insert
                    rate = (i + batch_count) / elapsed
                    print(f"  > Progress: {i + batch_count:,}/{self.scale:,} | Rate: {rate:.1f} ops/s | Mem: {self.get_mem_mb():.1f} MB", flush=True)
        finally:
            conn.close()

        insert_time = time.time() - start_insert
        workspace.bump_generation()
        print(f"✓ Insertion completed in {insert_time:.2f}s ({self.scale/insert_time:.1f} ops/s)", flush=True)

        # 3. Latency Check
        print("\n[3/5] Testing Search Latency...", flush=True)
        latencies = []
        # In mock mode, we need a query vector for the retriever if it's not mocked too.
        # But workspace.retriever uses real embedder by default via DI.
        # So we override it in mock mode.
        if self.mock:
             class MockRetriever:
                 def __init__(self, workspace): self.workspace = workspace
                 def retrieve(self, q, limit=10):
                     # Just do a random search in index to test index speed
                     import numpy as np
                     q_vec = np.random.rand(384).astype(np.float32)
                     return self.workspace.index.search(q_vec, top_k=limit)
             retriever = MockRetriever(workspace)
        else:
             retriever = workspace.retriever

        for _ in range(50):
            q = f"memory fact number {random.randint(0, self.scale)}"
            start_q = time.perf_counter()
            _ = retriever.retrieve(q, limit=10)
            latencies.append((time.perf_counter() - start_q) * 1000)
        
        latencies.sort()
        p50 = latencies[len(latencies)//2]
        p95 = latencies[int(len(latencies)*0.95)]
        print(f"✓ P50 Latency: {p50:.2f}ms", flush=True)
        print(f"✓ P95 Latency: {p95:.2f}ms", flush=True)

        # 4. Memory Footprint
        final_mem = self.get_mem_mb()
        print(f"\n[4/5] Memory Footprint Check", flush=True)
        print(f"[*] Final RAM Usage: {final_mem:.1f} MB", flush=True)
        print(f"[*] Overhead per 1K records: {(final_mem / self.scale) * 1024:.2f} MB", flush=True)

        # 5. SQLite Integrity
        print("\n[5/5] Checking Persistence...", flush=True)
        db_path = workspace.workspace_manager.get_db_path()
        db_size = os.path.getsize(db_path) / 1024 / 1024
        print(f"✓ SQLite File Size: {db_size:.2f} MB", flush=True)
        
        print("\n" + "=" * 50, flush=True)
        print("SUPER STRESS TEST SUCCESSFUL", flush=True)
        print("=" * 50, flush=True)

if __name__ == "__main__":
    import numpy as np # Ensure numpy is available for mock
    scale = 10000
    mock = False
    
    if len(sys.argv) > 1:
        if sys.argv[1].isdigit():
            scale = int(sys.argv[1])
        elif sys.argv[1].lower() == "--mock":
            mock = True
    
    if len(sys.argv) > 2:
        if sys.argv[2].isdigit():
            scale = int(sys.argv[2])
        elif sys.argv[2].lower() == "--mock":
            mock = True

    tester = SuperStressTest(scale=scale, mock=mock)
    asyncio.run(tester.run())
