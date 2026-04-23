import pytest
import os
import tempfile
import shutil
import numpy as np

from memk.storage.db import MemoryDB
from memk.sync.merkle import MerkleService
from memk.sync.protocol import SyncProtocolNode
from memk.sync.health import ReplicaHealthService, SyncMode
from memk.sync.hybrid import HybridSyncService

class MockRuntime:
    def __init__(self, db, workspace_id="test_ws"):
        self.db = db
        self.workspace_id = workspace_id

@pytest.fixture
def nodes():
    tmp_a = tempfile.mkdtemp()
    tmp_b = tempfile.mkdtemp()
    
    db_a = MemoryDB(os.path.join(tmp_a, "replica_a.db"))
    db_b = MemoryDB(os.path.join(tmp_b, "replica_b.db"))
    
    db_a.init_db()
    db_b.init_db()
    
    merkle_a = MerkleService(MockRuntime(db_a, "node_a"))
    merkle_b = MerkleService(MockRuntime(db_b, "node_b"))
    
    node_a = SyncProtocolNode(merkle_a)
    node_b = SyncProtocolNode(merkle_b)
    
    yield (node_a, node_b)
    
    try:
        shutil.rmtree(tmp_a)
        shutil.rmtree(tmp_b)
    except:
        pass

def test_hybrid_sync_full_lifecycle(nodes):
    node_a, node_b = nodes
    db_a, db_b = node_a.db, node_b.db
    
    # Setup Services for Node B (which will catch up to A)
    hybrid_b = HybridSyncService(node_b)
    
    # 1. INITIAL SYNC (B is new, should choose MERKLE)
    db_a.insert_memory("Initial Data 1")
    db_a.insert_memory("Initial Data 2", embedding=np.array([0.1, 0.2], dtype=np.float32))
    
    # Refresh A's merkle buckets
    node_a.merkle.rebuild_buckets(db_a.get_latest_version_hlc())
    
    # Synchronize
    report = hybrid_b.sync_from_source(node_a, "node_a")
    
    assert report["mode"] == SyncMode.MERKLE_RECOVERY.value
    assert report["stats"]["rows_recovered"] >= 2
    
    # Verify B Caught up
    assert len(db_b.get_all_memories()) == 2
    
    # 2. INCREMENTAL SYNC (B is fresh, should choose OPLOG)
    mem_id = db_a.insert_memory("Incremental Change")
    node_a.merkle.rebuild_buckets(db_a.get_latest_version_hlc())
    
    report = hybrid_b.sync_from_source(node_a, "node_a")
    assert report["mode"] == SyncMode.OPLOG_DELTA.value
    assert report["stats"]["applied_from_oplog"] >= 1
    
    assert len(db_b.get_all_memories()) == 3
    
    # 3. NODE B GOES OFFLINE & A PRUNES OPLOG
    # Create a lot of churn at A
    for i in range(5):
        db_a.insert_memory(f"Offline Churn {i}")
    
    # Update an existing item
    db_a.update_memory_content(mem_id, "Updated Incremental Change")
    
    # Prune Oplog in A completely
    latest_hlcl_a = db_a.get_latest_version_hlc()
    db_a.prune_oplog_entries(latest_hlcl_a + 1)
    
    # Refresh A's merkle
    node_a.merkle.rebuild_buckets(latest_hlcl_a)
    
    # 4. NODE B COMES BACK (Oplog is gone, should FALLBACK to MERKLE)
    report = hybrid_b.sync_from_source(node_a, "node_a")
    
    assert report["mode"] == SyncMode.MERKLE_RECOVERY.value
    assert report["stats"]["rows_recovered"] > 0
    
    # 5. VERIFY FINAL STATE
    mems_b = db_b.get_all_memories()
    assert len(mems_b) == 3 + 5 # 3 initial + 5 churn
    
    # Verify the update was applied correctly (LWW)
    updated_row = db_b.get_memory_by_id(mem_id)
    assert updated_row["content"] == "Updated Incremental Change"
    
    # Verify no data regression (A and B root hashes should match now)
    node_b.merkle.rebuild_buckets(db_b.get_latest_version_hlc())
    assert node_a.get_root_hash() == node_b.get_root_hash()
