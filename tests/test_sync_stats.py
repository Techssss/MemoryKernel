import os
import uuid
import pytest
import time
import numpy as np
from memk.core.runtime import get_runtime
from memk.sync.stats import SyncStatsService
from memk.sync.gc import OplogGC

os.environ["MEMK_DAEMON_MODE"] = "1"

def test_sync_stats_reporting():
    workspace_id = f"test_stats_ws_{uuid.uuid4().hex[:8]}"
    tmp_path = f"test_stats_{uuid.uuid4().hex[:8]}.db"
    
    global_runtime = get_runtime()
    global_runtime._is_global_initialized = True
    
    # Mock embedder
    class MockSharedEmbedder:
        def __init__(self): self.dim = 3
        def embed(self, t): return [0.0]*3
        
    global_runtime.shared_embedder = MockSharedEmbedder()
    from memk.core.runtime import WorkspaceRuntime
    runtime = WorkspaceRuntime(workspace_id, tmp_path, global_runtime.shared_embedder)
    db = runtime.db
    stats_service = SyncStatsService(runtime)
    
    try:
        # 1. Base state (Fresh DB)
        stats = stats_service.get_sync_hardening_stats()
        assert stats["oplog"]["count"] == 0
        assert stats["replicas"]["checkpoint_count"] == 0
        assert stats["gc"]["last_run"] == "never"
        
        # 2. Add some data to generate oplog entries
        mem_id = db.insert_memory("Test memory")
        
        # 3. Add a replica checkpoint
        db.upsert_replica_checkpoint("replica_alpha", 12345, "node_remote", 0)
        
        # 4. Trigger a GC run to have GC stats
        OplogGC.run_oplog_gc_job(db, retention_seconds=0) # Prune everything possible
        
        # 5. Check stats again
        stats = stats_service.get_sync_hardening_stats()
        assert stats["oplog"]["count"] > 0
        assert stats["replicas"]["checkpoint_count"] == 1
        assert stats["gc"]["last_run"] != "never"
        
        # 6. Simulate stale state
        # Delete row bypassing sync hooks
        with db._get_connection() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
            
        stats = stats_service.get_sync_hardening_stats()
        assert stats["integrity"]["stale_row_hash_count"] == 1
        
        # Print example output for verification
        print("\nExample Stats Output:")
        import json
        print(json.dumps(stats, indent=2))
        
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                os.remove(tmp_path + "-wal")
                os.remove(tmp_path + "-shm")
            except: pass

if __name__ == "__main__":
    test_sync_stats_reporting()
