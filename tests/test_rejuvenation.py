import os
import uuid
import pytest

from memk.core.runtime import get_runtime
from memk.consolidation.rejuvenation import MemoryRejuvenator

os.environ["MEMK_DAEMON_MODE"] = "1"

def test_memory_rejuvenation_state_machine():
    workspace_id = f"test_rejuv_ws_{uuid.uuid4().hex[:8]}"
    tmp_path = f"test_{uuid.uuid4().hex[:8]}.db"
    
    global_runtime = get_runtime()
    global_runtime._is_global_initialized = True
    
    class MockSharedEmbedder:
        def __init__(self):
            self.dim = 3
        def embed(self, t): return [0.0]*3
        
    global_runtime.shared_embedder = MockSharedEmbedder()
    global_runtime.embedder_pipeline = None
    
    runtime = global_runtime.get_workspace_runtime(workspace_id, db_path=tmp_path)
    db = runtime.db
    
    try:
        # Create raw memory
        mem_id = db.insert_memory("Test old memory", importance=0.5)
        
        # Verify initial state ACTIVE (0)
        with db.connection() as conn:
            row = conn.execute("SELECT archived FROM memories WHERE id = ?", (mem_id,)).fetchone()
            assert row["archived"] == 0
            
        # Simulate Consolidator action -> ARCHIVED (1)
        db.archive_memory(mem_id)
        with db.connection() as conn:
            row = conn.execute("SELECT archived FROM memories WHERE id = ?", (mem_id,)).fetchone()
            assert row["archived"] == 1
            
        # 1. Test Implicit Rejuvenation (Graph Traversal Hits)
        rejuv = MemoryRejuvenator(runtime, access_threshold=3)
        
        rejuv.evaluate_memory_access(mem_id) # hit 1
        with db.connection() as conn:
            row = conn.execute("SELECT archived, access_count FROM memories WHERE id = ?", (mem_id,)).fetchone()
            assert row["archived"] == 1 # still archived!
            assert row["access_count"] == 1
            
        rejuv.evaluate_memory_access(mem_id) # hit 2
        revived = rejuv.evaluate_memory_access(mem_id) # hit 3
        
        assert revived is True
        with db.connection() as conn:
            row = conn.execute("SELECT archived, access_count FROM memories WHERE id = ?", (mem_id,)).fetchone()
            assert row["archived"] == 0 # BACK TO ACTIVE!
            assert row["access_count"] == 3
            
        # Archive it again to test explicit hook
        db.archive_memory(mem_id)
        
        # 2. Test Explicit Contradiction
        recon_flag = rejuv.flag_for_reconsolidation(mem_id, reason="mismatched polarities spotted by agent")
        assert recon_flag is True
        
        with db.connection() as conn:
            row = conn.execute("SELECT archived, importance FROM memories WHERE id = ?", (mem_id,)).fetchone()
            assert row["archived"] == 0 # ACTIVE pool!
            assert row["importance"] >= 0.8 # Promoted!
            
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                os.remove(tmp_path + "-wal")
                os.remove(tmp_path + "-shm")
            except:
                pass
