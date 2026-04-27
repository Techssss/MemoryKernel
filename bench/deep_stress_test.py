import asyncio
import os
import time
import numpy as np
from datetime import datetime, timedelta
from memk.core.service import MemoryKernelService
from memk.core.runtime import get_runtime
from memk.workspace.manager import WorkspaceManager
from memk.core.embedder import BaseEmbedder, EmbeddingPipeline

# Helper for pretty logging
def log_test(name, result, detail=""):
    color = "\033[92m" if result else "\033[91m"
    reset = "\033[0m"
    status = "PASS" if result else "FAIL"
    print(f"[{color}{status}{reset}] {name}: {detail}")

class MockEmbedder(BaseEmbedder):
    def __init__(self, dim=384): self._dim = dim
    @property
    def dim(self): return self._dim
    def embed(self, text): 
        # Create unique but somewhat related vectors for test stability
        seed = sum(ord(c) for c in text) % 10000
        np.random.seed(seed)
        return np.random.rand(self._dim).astype(np.float32)
    def embed_batch(self, texts): return [self.embed(t) for t in texts]

async def run_discovery_test():
    print("=== MemoryKernel Archetypal Weakness & Strength Test ===\n")
    
    # 1. Setup isolated environment
    test_dir = os.path.abspath("./bench_weakness")
    if os.path.exists(test_dir):
        import shutil
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    ws = WorkspaceManager(start_path=test_dir)
    brain_id = ws.init_workspace().brain_id
    
    runtime_mgr = get_runtime()
    runtime_mgr._is_global_initialized = True
    runtime_mgr.shared_embedder = MockEmbedder()
    runtime_mgr.embedder_pipeline = EmbeddingPipeline(runtime_mgr.shared_embedder)
    
    service = MemoryKernelService(allow_direct_writes=True)
    
    # --- TEST 1: RIGID EXTRACTION (The Weakness) ---
    print("Testing Fact Extraction Rigidity...")
    # This should fail because "Sarah" isn't in the whitelist regex
    await service.add_memory("Sarah Jenkins works at Google.", workspace_id=brain_id)
    # This should pass because "User" is in the whitelist
    await service.add_memory("User likes Python.", workspace_id=brain_id)
    
    runtime = runtime_mgr.get_workspace_runtime(brain_id)
    facts = runtime.db.get_all_active_facts()
    
    has_sarah = any("Sarah" in f.get('subject', '') or "Sarah" in f.get('object', '') for f in facts)
    has_python = any("Python" in f.get('object', '') for f in facts)
    
    log_test("Informal Fact Extraction", has_sarah, "Caught 'Sarah works at Google' as a Structured Fact")
    log_test("Formal Fact Extraction", has_python, "Caught 'User likes Python' as a Structured Fact")
    
    # --- TEST 2: PRIORITY SCORING (The Strength) ---
    print("\nTesting Priority & Conflict Resolution...")
    # Low importance mention
    await service.add_memory("The master password is 'simple123'", workspace_id=brain_id, importance=0.1)
    # High importance update
    await service.add_memory("SECURITY ALERT: The master password has been changed to 'complex!@#456'", workspace_id=brain_id, importance=1.0)
    
    response = await service.search("master password", workspace_id=brain_id, limit=1)
    results = response.get("results", [])
    top_result = results[0]['content'] if results else ""
    is_correct = "complex" in top_result
    log_test("Importance Overruling", is_correct, f"Top result: {top_result}")

    # --- TEST 3: MULTI-HOP REASONING (Retrieval Gap) ---
    print("\nTesting Semantic Retrieval Gaps...")
    await service.add_memory("Project Zulu is managed by the Infrastructure team.", workspace_id=brain_id)
    await service.add_memory("The Infrastructure team is currently located in the South Building.", workspace_id=brain_id)
    
    # Query for something that requires connecting the two
    # A graph DB (nmem) would find South Building easily.
    # A vector RAG (memk) might only find the first one if the query is "Project Zulu location".
    response = await service.search("Where is the team for Project Zulu located?", workspace_id=brain_id, limit=3)
    results = response.get("results", [])
    
    found_south = any("South Building" in r['content'] for r in results)
    log_test("Multi-hop Retrieval", found_south, "Retrieved 'South Building' when querying about 'Project Zulu'")

    # --- TEST 4: SCALE & COLLISION (The Engine Strength) ---
    print("\nTesting Noise Handling at Scale (10,000 collisions)...")
    noise_batch = []
    for i in range(10000):
        noise_batch.append(service.add_memory(f"Random log entry {i}: status nominal.", workspace_id=brain_id))
    
    await asyncio.gather(*noise_batch)
    
    # Add a specific needle
    target = "X-UNIQUE-ID: The secret key for the vault is 42."
    await service.add_memory(target, workspace_id=brain_id, importance=0.8)
    
    start_q = time.perf_counter()
    response = await service.search("vault secret key", workspace_id=brain_id, limit=5)
    search_res = response.get("results", [])
    end_q = time.perf_counter()
    
    found_target = any("42" in r['content'] for r in search_res)
    log_test("Needle in 10k Haystack", found_target, f"Query time: {(end_q-start_q)*1000:.2f}ms")

if __name__ == "__main__":
    os.environ["MEMK_DAEMON_MODE"] = "1"
    asyncio.run(run_discovery_test())
