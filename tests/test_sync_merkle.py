import os
import uuid
import pytest

from memk.core.runtime import get_runtime
from memk.sync.merkle import MerkleService
from memk.core.hlc import GLOBAL_HLC

os.environ["MEMK_DAEMON_MODE"] = "1"

def test_merkle_oplog_and_bucket():
    workspace_id = f"test_merkle_ws_{uuid.uuid4().hex[:8]}"
    tmp_path = f"test_merkle_{uuid.uuid4().hex[:8]}.db"
    
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
        mem_id = db.insert_memory("Test sync payload")
        
        # Verify oplog was created correctly via automatic hook
        with db.connection() as conn:
            oplogs = conn.execute("SELECT * FROM oplog").fetchall()
            assert len(oplogs) == 1
            op = oplogs[0]
            assert op["table_name"] == "memories"
            assert op["operation"] == "INSERT"
            assert op["row_id"] == mem_id
            
            row_hashes = conn.execute("SELECT * FROM row_hash").fetchall()
            assert len(row_hashes) == 1
            h = row_hashes[0]
            assert h["row_id"] == mem_id
            assert len(h["hash_val"]) == 64
            
        # Run merkle rebuild
        merkle = MerkleService(runtime, num_buckets=16)
        hlc_timestamp = GLOBAL_HLC.next_version()[0]
        
        state = merkle.rebuild_buckets(current_hlc=hlc_timestamp)
        assert len(state) == 16
        
        # Verify exactly one bucket got the hash, 15 are empty
        empty_count = sum(1 for v in state.values() if v == "empty")
        assert empty_count == 15
        
        # Ensure it got written to DB properly
        with db.connection() as conn:
            mb = conn.execute("SELECT * FROM merkle_bucket").fetchall()
            assert len(mb) == 16

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                os.remove(tmp_path + "-wal")
                os.remove(tmp_path + "-shm")
            except:
                pass
