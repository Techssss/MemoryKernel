import os
import uuid
import pytest

from memk.core.runtime import get_runtime
from memk.sync.merkle import MerkleService
from memk.sync.protocol import SyncProtocolNode
from memk.core.hlc import GLOBAL_HLC

os.environ["MEMK_DAEMON_MODE"] = "1"

def test_sync_two_replicas():
    ws_id = f"test_sync_protocol_ws_{uuid.uuid4().hex[:8]}"
    db_node_a = f"replica_A_{uuid.uuid4().hex[:8]}.db"
    db_node_b = f"replica_B_{uuid.uuid4().hex[:8]}.db"
    
    global_runtime = get_runtime()
    global_runtime._is_global_initialized = True
    
    class MockSharedEmbedder:
        def __init__(self): self.dim = 3
        def embed(self, t): return [0.0]*3
        
    global_runtime.shared_embedder = MockSharedEmbedder()
    global_runtime.embedder_pipeline = None
    
    from memk.core.runtime import WorkspaceRuntime
    
    # Init Node A
    rt_A = WorkspaceRuntime(ws_id, db_node_a, global_runtime.shared_embedder)
    merkle_A = MerkleService(rt_A, num_buckets=16)
    node_a = SyncProtocolNode(merkle_A)
    
    # Init Node B
    rt_B = WorkspaceRuntime(ws_id, db_node_b, global_runtime.shared_embedder)
    merkle_B = MerkleService(rt_B, num_buckets=16)
    node_b = SyncProtocolNode(merkle_B)
    
    try:
        # 1. State: Both are empty
        hlc_timestamp = GLOBAL_HLC.next_version()[0]
        merkle_A.rebuild_buckets(hlc_timestamp)
        merkle_B.rebuild_buckets(hlc_timestamp)
        
        root_a = node_a.get_root_hash()
        root_b = node_b.get_root_hash()
        assert root_a == root_b
        
        # 2. Add some memories to Node A ONLY
        m1 = rt_A.db.insert_memory("Important facts on Node A")
        m2 = rt_A.db.insert_memory("Specific document sync")
        
        # Rebuild Node A buckets
        hlc_timestamp = GLOBAL_HLC.next_version()[0]
        merkle_A.rebuild_buckets(hlc_timestamp)
        
        # Now roots should differ
        assert node_a.get_root_hash() != node_b.get_root_hash()
        
        # 3. Protocol step: B asks A for buckets, then diffs them locally.
        buckets_a = node_a.get_bucket_hashes()
        mismatched_buckets = node_b.diff_buckets(buckets_a)
        
        assert len(mismatched_buckets) > 0
        
        # 4. B asks A for deltas of the mismatched buckets
        deltas = node_a.fetch_delta_for_buckets(mismatched_buckets)
        
        assert len(deltas) == 2
        assert deltas[0]["row_id"] in [m1, m2]
        
        # 5. B applies deltas locally along with marking Checkpoint for node A
        max_hlc_a = rt_A.db.get_min_acknowledged_hlc() or 0
        node_b.apply_remote_delta(
            deltas,
            remote_replica_id="replica-A-uuid"
        )
        merkle_B.rebuild_buckets(GLOBAL_HLC.next_version()[0])
        
        # 6. Verify full sync
        assert node_a.get_root_hash() == node_b.get_root_hash()
        
        # Check Node B actually has the rows in DB
        with rt_B.db._get_connection() as conn:
            cnt = conn.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
            assert cnt == 2
            
        # 7. Check if Node B saved Replica A's checkpoint successfully
        chk = rt_B.db.get_replica_checkpoint("replica-A-uuid")
        assert chk is not None
        assert chk["note"] == "Delta Batch Applied"
            
    finally:
        for p in [db_node_a, db_node_b]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                    os.remove(p + "-wal")
                    os.remove(p + "-shm")
                except: pass
