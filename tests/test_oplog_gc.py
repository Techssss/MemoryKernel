import os
import uuid
import time
import pytest

from memk.core.runtime import get_runtime
from memk.sync.gc import OplogGC

os.environ["MEMK_DAEMON_MODE"] = "1"

def test_get_safe_prune_boundary():
    workspace_id = f"test_gc_ws_{uuid.uuid4().hex[:8]}"
    tmp_path = f"test_gc_{uuid.uuid4().hex[:8]}.db"
    
    global_runtime = get_runtime()
    global_runtime._is_global_initialized = True
    
    # Bypass Embedder initialization for DB tests
    class MockSharedEmbedder:
        def __init__(self): self.dim = 3
        def embed(self, t): return [0.0]*3
        
    global_runtime.shared_embedder = MockSharedEmbedder()
    global_runtime.embedder_pipeline = None
    
    from memk.core.runtime import WorkspaceRuntime
    runtime = WorkspaceRuntime(workspace_id, tmp_path, global_runtime.shared_embedder)
    db = runtime.db
    
    try:
        now_ms = 1000000000000  # Synthesize a fixed current time
        ret_sec = 86400 * 7 # 7 days
        ret_ms = ret_sec * 1000
        boundary_ttl_only = now_ms - ret_ms
        
        # 1. No checkpoints known -> Should fall back strictly to retention TTL
        assert db.get_min_acknowledged_hlc() is None
        prune_hlc = OplogGC.get_safe_prune_boundary(db, retention_seconds=ret_sec, override_now_ms=now_ms)
        assert prune_hlc == boundary_ttl_only
        
        # 2. Add two replicas that are actively caught up (Past the retention TTL)
        db.upsert_replica_checkpoint("repA", boundary_ttl_only + 5000, "nodeA", 0)
        db.upsert_replica_checkpoint("repB", boundary_ttl_only + 9000, "nodeB", 0)
        
        # Because the lowest replica is still newer than TTL, we MUST clamp to TTL so we don't prematurely prune valid retention history.
        prune_hlc = OplogGC.get_safe_prune_boundary(db, retention_seconds=ret_sec, override_now_ms=now_ms)
        assert prune_hlc == boundary_ttl_only
        
        # 3. Add a slow replica trailing BEHIND the retention window!
        db.upsert_replica_checkpoint("repSLOW", boundary_ttl_only - 10000, "nodeSLOW", 0)
        
        # We must NOT prune up to TTL, we must prune to the slow replica!
        prune_hlc = OplogGC.get_safe_prune_boundary(db, retention_seconds=ret_sec, override_now_ms=now_ms)
        assert prune_hlc == boundary_ttl_only - 10000
        
        # 4. Retention window not enough yet
        prune_hlc_short = OplogGC.get_safe_prune_boundary(db, retention_seconds=2000000, override_now_ms=now_ms)
        expected_short_boundary = now_ms - (2000000 * 1000)
        # expected_short is much further back, so it dominates the slow replica boundary
        assert prune_hlc_short == min(expected_short_boundary, boundary_ttl_only - 10000)
        
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                os.remove(tmp_path + "-wal")
                os.remove(tmp_path + "-shm")
            except:
                pass

def test_run_oplog_gc_job_execution():
    workspace_id = f"test_gc_exec_ws_{uuid.uuid4().hex[:8]}"
    tmp_path = f"test_gc_exec_{uuid.uuid4().hex[:8]}.db"
    
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
    
    try:
        now_ms = 1000000000000
        
        with db.connection() as conn:
            # Insert raw ops manually to simulate historical actions
            # We'll make them all chronologically far behind the retention limit
            for i in range(10):
                conn.execute(
                    "INSERT INTO oplog (version_hlc, version_node, version_seq, operation, table_name, row_id) VALUES (?, ?, ?, ?, ?, ?)",
                    (now_ms - 900000000 + i, "nodeA", i, "INSERT", "memories", f"mem_{i}")
                )
                
            # One new oplog very close to now_ms
            conn.execute(
                "INSERT INTO oplog (version_hlc, version_node, version_seq, operation, table_name, row_id) VALUES (?, ?, ?, ?, ?, ?)",
                (now_ms - 100, "nodeA", 0, "INSERT", "memories", "mem_new")
            )
            
        with db.connection() as conn:
            # Total oplogs: 11
            cnt_opt = conn.execute("SELECT COUNT(*) as c FROM oplog").fetchone()["c"]
        
        assert cnt_opt == 11
        
        # 1. Dry run execution
        stats = OplogGC.run_oplog_gc_job(db, retention_seconds=86400, batch_size=2, dry_run=True, override_now_ms=now_ms)
        assert stats["dry_run"] is True
        assert stats["deleted_count"] == 10 # 10 elements are caught in the boundary
        
        # Oplog size should remain unchanged
        with db.connection() as conn:
            cnt_opt = conn.execute("SELECT COUNT(*) as c FROM oplog").fetchone()["c"]
        assert cnt_opt == 11
        
        # 2. Add a very slow checkpoint trailing ahead of the extreme past but behind now_ms!
        # Example: clamp exactly such that 5 items are eligible to delete
        cutoff_val = (now_ms - 900000000) + 5
        db.upsert_replica_checkpoint("repSLOW", cutoff_val, "nodeSLOW", 0)
        
        # 3. Real execution with batches of 2
        stats = OplogGC.run_oplog_gc_job(db, retention_seconds=86400, batch_size=2, dry_run=False, override_now_ms=now_ms)
        assert stats["dry_run"] is False
        assert stats["deleted_count"] == 5 # 5 elements deleted
        assert stats["batches"] == 3 # 2 + 2 + 1 (+ final return)
        
        # Final physical size
        with db.connection() as conn:
            cnt_opt = conn.execute("SELECT COUNT(*) as c FROM oplog").fetchone()["c"]
        assert cnt_opt == 6 # 11 - 5
        
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                os.remove(tmp_path + "-wal")
                os.remove(tmp_path + "-shm")
            except: pass
