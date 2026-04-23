import os
import uuid
import pytest

from memk.core.runtime import get_runtime

os.environ["MEMK_DAEMON_MODE"] = "1"

def test_replica_checkpoint_crud():
    workspace_id = f"test_checkpoint_ws_{uuid.uuid4().hex[:8]}"
    tmp_path = f"test_checkpoint_{uuid.uuid4().hex[:8]}.db"
    
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
        # 1. Initially Empty
        assert db.get_replica_checkpoint("remote-node-1") is None
        assert db.get_min_acknowledged_hlc() is None
        assert len(db.list_replica_checkpoints()) == 0
        
        # 2. Upsert first checkpoint
        db.upsert_replica_checkpoint(
            replica_id="remote-node-1",
            hlc=1000,
            node="node_A",
            seq=5,
            note="Initial Sync Branch"
        )
        chk1 = db.get_replica_checkpoint("remote-node-1")
        assert chk1 is not None
        assert chk1["last_applied_hlc"] == 1000
        assert chk1["last_applied_node"] == "node_A"
        assert chk1["last_applied_seq"] == 5
        assert chk1["note"] == "Initial Sync Branch"
        
        # 3. Add second checkpoint
        db.upsert_replica_checkpoint("remote-node-2", 800, "node_B", 0, None)
        
        # 4. List and Min HLC
        all_chks = db.list_replica_checkpoints()
        assert len(all_chks) == 2
        
        min_hlc = db.get_min_acknowledged_hlc()
        assert min_hlc == 800
        
        # 5. Update first checkpoint
        db.upsert_replica_checkpoint("remote-node-1", 1200, "node_A", 1, "Completed Full Sync")
        chk1_updated = db.get_replica_checkpoint("remote-node-1")
        assert chk1_updated["last_applied_hlc"] == 1200
        assert chk1_updated["note"] == "Completed Full Sync"
        
        # Note: the update shifts the min_hlc logic if the lower one bumps
        db.upsert_replica_checkpoint("remote-node-2", 1500, "node_B", 0, None)
        assert db.get_min_acknowledged_hlc() == 1200
        
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                os.remove(tmp_path + "-wal")
                os.remove(tmp_path + "-shm")
            except:
                pass
