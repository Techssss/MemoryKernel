import pytest
import tempfile
import os
import shutil
from memk.storage.db import MemoryDB
from memk.sync.merkle import MerkleService
from memk.sync.protocol import SyncProtocolNode
from memk.sync.recovery import MerkleRecoveryService

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
    
    merkle_a = MerkleService(MockRuntime(db_a))
    merkle_b = MerkleService(MockRuntime(db_b))
    
    node_a = SyncProtocolNode(merkle_a)
    node_b = SyncProtocolNode(merkle_b)
    
    yield (node_a, node_b)
    
    # Cleanup
    try:
        shutil.rmtree(tmp_a)
        shutil.rmtree(tmp_b)
    except:
        pass

def test_stale_replica_recovery_via_merkle(nodes):
    node_a, node_b = nodes
    db_a, db_b = node_a.db, node_b.db
    
    # 1. Fill A with data
    mem_id = db_a.insert_memory("Proprietary Secret A")
    db_a.insert_memory("Another Fact")
    
    # IMPORTANT: Rebuild Merkle buckets so state is visible to protocol
    node_a.merkle.rebuild_buckets(db_a.get_latest_version_hlc())
    node_b.merkle.rebuild_buckets(db_b.get_latest_version_hlc())
    
    # 2. B is empty. We simulate A having pruned its oplog.
    db_a.prune_oplog_entries(db_a.get_latest_version_hlc() + 1)
    
    # 3. Create Recovery Service at B
    recovery_at_b = MerkleRecoveryService(node_b)
    
    # 4. Recover B from A
    results = recovery_at_b.recover_from_remote(node_a, "node_alpha")
    
    assert results["status"] == "completed"
    assert results["rows_recovered"] >= 2
    
    # 5. Verify B now has the data
    memories_b = db_b.get_all_memories()
    contents = [m["content"] for m in memories_b]
    assert "Proprietary Secret A" in contents

def test_full_merkle_recovery_respects_lww(nodes):
    node_a, node_b = nodes
    db_a, db_b = node_a.db, node_b.db
    
    # Use SAME ID for both to test conflict resolution
    shared_id = "target_memory_123"
    
    # Insert V1 at B (Older)
    conn_b = db_b._get_connection()
    try:
        conn_b.execute(
            "INSERT INTO memories (id, content, created_at, version_hlc) VALUES (?, ?, ?, ?)",
            (shared_id, "Content V1", "2024-01-01", 10)
        )
        conn_b.commit()
    finally:
        conn_b.close()

    # Trigger row_hash/merkle for B
    db_b.touch_memory(shared_id) 
    
    actual_id = db_a.insert_memory("Content V2")
    row_a = db_a.get_memory_by_id(actual_id)
    high_hlc = row_a["version_hlc"]
    
    conn_b = db_b._get_connection()
    try:
        conn_b.execute(
            "INSERT INTO memories (id, content, created_at, version_hlc) VALUES (?, ?, ?, ?)",
            (actual_id, "Content V1", "2024-01-01", high_hlc - 1)
        )
        conn_b.commit()
    finally:
        conn_b.close()
    
    # Ensure row_hash exists at both for Merkle to see them
    conn_b = db_b._get_connection()
    try:
        db_b._log_sync_operation(conn_b, "memories", actual_id, "INSERT", {}, (high_hlc - 1, "test", 0))
        conn_b.commit()
    finally:
        conn_b.close()
    
    # Rebuild buckets
    node_a.merkle.rebuild_buckets(high_hlc)
    node_b.merkle.rebuild_buckets(high_hlc - 1)
    
    # Recover B from A (A is newer)
    recovery_at_b = MerkleRecoveryService(node_b)
    recovery_at_b.recover_from_remote(node_a, "node_alpha")
    
    assert db_b.get_memory_by_id(actual_id)["content"] == "Content V2"
    
    # Now try "recovering" A from B (B is older)
    # Update B's merkle just in case (though it shouldn't change its internal version)
    node_b.merkle.rebuild_buckets(high_hlc - 1)
    
    recovery_at_a = MerkleRecoveryService(node_a)
    recovery_at_a.recover_from_remote(node_b, "node_beta")
    
    # A should STAY at V2
    assert db_a.get_memory_by_id(actual_id)["content"] == "Content V2"
