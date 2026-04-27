import os
import uuid
import time
import pytest
import numpy as np
from memk.core.runtime import get_runtime
from memk.sync.merkle import MerkleService

os.environ["MEMK_DAEMON_MODE"] = "1"

def test_merkle_cleanup_lifecycle():
    workspace_id = f"test_cleanup_ws_{uuid.uuid4().hex[:8]}"
    tmp_path = f"test_cleanup_{uuid.uuid4().hex[:8]}.db"
    
    global_runtime = get_runtime()
    global_runtime._is_global_initialized = True
    
    # Bypass Embedder initialization for DB tests
    class MockSharedEmbedder:
        def __init__(self): self.dim = 3
        def embed(self, t): return [0.0]*3
        
    global_runtime.shared_embedder = MockSharedEmbedder()
    from memk.core.runtime import WorkspaceRuntime
    runtime = WorkspaceRuntime(workspace_id, tmp_path, global_runtime.shared_embedder)
    db = runtime.db
    merkle = MerkleService(runtime, num_buckets=4) # small buckets for easy collision tracking
    
    try:
        # Create a few rows directly using proper hooks to generate row_hash
        memA_id = db.insert_memory("Ghost record to be")
        db.update_memory_embedding(memA_id, np.array([0.1, 0.2, 0.3], dtype=np.float32))
        
        memB_id = db.insert_memory("Valid record")
        db.update_memory_embedding(memB_id, np.array([0.4, 0.5, 0.6], dtype=np.float32))
        
        # Build initial buckets
        merkle.rebuild_or_refresh_merkle_buckets(100)
        
        with db.connection() as conn:
            cnt_row_hash = conn.execute("SELECT COUNT(*) as c FROM row_hash").fetchone()["c"]
            cnt_buckets = conn.execute("SELECT COUNT(*) as c FROM merkle_bucket").fetchone()["c"]
        assert cnt_row_hash >= 2
        
        # 1. Simulate hard physical row deletion without oplog interceptors (Abrupt vanishing)
        with db.connection() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memA_id,))
        
        # Dry Run Orphans
        stats_dry = merkle.cleanup_stale_row_hashes(verify_content_hash=False, dry_run=True)
        assert stats_dry["dry_run"] is True
        assert stats_dry["orphans_deleted"] == 1
        
        # Execution Orphans
        stats = merkle.cleanup_stale_row_hashes(verify_content_hash=False, dry_run=False)
        assert stats["dry_run"] is False
        assert stats["orphans_deleted"] == 1
        
        # Row A hash is now gone. Let's see scope clearance.
        with db.connection() as conn:
            conn.execute("DELETE FROM row_hash WHERE table_name = 'memories'") # Nuke everything to test empty scopes
            conn.execute("DELETE FROM memories")
            
        # 2. Scope Rỗng -> Delete Buckets 
        # Merkle should delete everything now that row_hash is totally empty
        sweep_stats = merkle.rebuild_or_refresh_merkle_buckets(101)
        assert sweep_stats["buckets_deleted"] > 0
        
        with db.connection() as conn:
            cnt_buckets = conn.execute("SELECT COUNT(*) as c FROM merkle_bucket").fetchone()["c"]
        assert cnt_buckets == 0
        
        # 3. Simulate drift (Update row physically without going via _log_sync_operation)
        memC_id = db.insert_memory("Drifting record")
        db.update_memory_embedding(memC_id, np.array([0.1, 0.2, 0.3], dtype=np.float32))
        # Manually tamper with the row_hash to simulate it being stale!
        with db.connection() as conn:
            conn.execute("UPDATE row_hash SET hash_val = 'STALE_BOGUS_HASH' WHERE row_id = ?", (memC_id,))
            
        # Repair!
        stats_repair = merkle.cleanup_stale_row_hashes(verify_content_hash=True, dry_run=False)
        assert stats_repair["hashes_corrected"] == 1
        
        with db.connection() as conn:
            restored = conn.execute("SELECT hash_val FROM row_hash WHERE row_id = ?", (memC_id,)).fetchone()
        assert restored["hash_val"] != "STALE_BOGUS_HASH"
        
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                os.remove(tmp_path + "-wal")
                os.remove(tmp_path + "-shm")
            except: pass
