import os
import time
import uuid

from memk.core.hlc import HLClock, GLOBAL_HLC
from memk.core.runtime import get_runtime

def test_hlc_monotonically_increasing():
    clock = HLClock(node_id="test_node")
    
    # 1. Standard monotonic tick
    v1 = clock.next_version()
    v2 = clock.next_version()
    
    # either time advanced, or sequence advanced
    assert v2[0] >= v1[0]
    if v2[0] == v1[0]:
        assert v2[2] > v1[2]
        
    assert v1[1] == "test_node"

def test_db_insert_applies_hlc():
    workspace_id = f"test_hlc_ws_{uuid.uuid4().hex[:8]}"
    tmp_path = f"test_hlc_{uuid.uuid4().hex[:8]}.db"
    
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
        # Override GLOBAL_HLC node_id for test consistency
        from memk.core.hlc import GLOBAL_HLC
        GLOBAL_HLC.node_id = "test_sync_node"
        
        mem_id = db.insert_memory("Testing distributed sync schemas")
        
        # Verify
        with db._get_connection() as conn:
            row = conn.execute("SELECT version_hlc, version_node, version_seq FROM memories WHERE id = ?", (mem_id,)).fetchone()
            
            assert row is not None
            assert row["version_hlc"] > 0
            assert row["version_node"] == "test_sync_node"
            assert row["version_seq"] >= 0
            
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                os.remove(tmp_path + "-wal")
                os.remove(tmp_path + "-shm")
            except:
                pass
