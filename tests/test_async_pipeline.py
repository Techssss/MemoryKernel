import os
import sys
import time
import uuid
import pytest

from memk.core.runtime import get_runtime
from memk.core.service import MemoryKernelService

# Bypass the daemon writability check
os.environ["MEMK_DAEMON_MODE"] = "1"

def test_async_extraction_pipeline():
    import asyncio
    asyncio.run(_test_async_extraction_pipeline())

async def _test_async_extraction_pipeline():
    """
    Test that high-importance memory writes successfully enqueue a job 
    into the WorkspaceRuntime's BackgroundJobManager, and that the 
    worker threads successfully consume and execute it.
    """
    workspace_id = f"test_async_ws_{uuid.uuid4().hex[:8]}"
    
    global_runtime = get_runtime()
    
    # Mock global initialization to prevent downloading/loading models
    global_runtime._is_global_initialized = True
    
    # Mock embedder pipeline so we don't need real models in test
    import numpy as np
    class MockSharedEmbedder:
        def __init__(self):
            self.dim = 1536
        def embed(self, t): return np.zeros(1536, dtype=np.float32)
        
    global_runtime.shared_embedder = MockSharedEmbedder()
    
    # Default fallback to shared_embedder
    global_runtime.embedder_pipeline = None
    
    # Isolate from real DB
    tmp_path = f"test_{uuid.uuid4().hex[:8]}.db"
    
    service = MemoryKernelService()
    runtime = global_runtime.get_workspace_runtime(workspace_id, db_path=tmp_path)
    
    # Assert job queue is completely empty
    assert len(runtime.jobs.jobs) == 0
    
    # 1. Insert low importance memory (should NOT enqueue IF facts are found)
    # Wait, if we use a dummy string, spacy might say NO facts, which WILL trigger the async!
    # Let's insert a string that DEFINITELY triggers Spacy facts: "Bob works at Google."
    res1 = await service.add_memory("Bob works at Google.", importance=0.1, confidence=0.1, workspace_id=workspace_id)
    # Wait, the SpacyExtractor might be RuleBasedExtractor if spacy isn't loaded properly in the test.
    # We can just check that AT LEAST ONE job is enqueued if we force importance=0.9
    
    # 2. Insert HIGH importance memory (SHOULD enqueue)
    res2 = await service.add_memory("Alice leads the engineering team.", importance=0.9, workspace_id=workspace_id)
    
    # The job manager should definitely have recorded a job now!
    jobs_dict = dict(runtime.jobs.jobs)
    assert len(jobs_dict) >= 1, "Expected at least one async extraction job to be enqueued."
    
    # Find our job
    job = list(jobs_dict.values())[-1]
    assert job.type == "enhanced_extraction"
    
    # 3. Wait for Worker to Consume it! 
    # BackgroundJobManager starts threads natively, so it should process automatically.
    print(f"\nWaiting for Async Job {job.id} to be executed...")
    
    timeout = 10
    start = time.time()
    while job.status.value in ("pending", "running") and time.time() - start < timeout:
        time.sleep(0.1)
        
    assert job.status.value == "completed", f"Job failed to complete. Status: {job.status.value}, Error: {job.error}"
    assert job.progress == 1.0
    print(f"Async Job {job.id} completed successfully in background!")

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
