import os
import time
import json
import sqlite3
import tempfile
import pytest
import numpy as np
import hashlib
import logging
from unittest.mock import patch
from memk.storage.db import MemoryDB, DatabaseError
from memk.sync.merkle import MerkleService
from memk.sync.protocol import SyncProtocolNode
from memk.sync.gc import OplogGC

# Configure logging for the stress test
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stress_test")

from types import SimpleNamespace

class ReplicaNode:
    """Helper to simulate a single synchronization participant."""
    def __init__(self, name, node_id, db_path):
        self.name = name
        self.node_id = node_id
        self.db = MemoryDB(db_path)
        self.db.init_db()
        
        # MerkleService needs a runtime object that has a .db attribute
        self.runtime = SimpleNamespace(db=self.db)
        self.merkle = MerkleService(self.runtime)
        self.protocol = SyncProtocolNode(self.merkle)

    def add_memories(self, count=10):
        ids = []
        with patch("memk.storage.db.GLOBAL_HLC.node_id", self.node_id):
            for i in range(count):
                content = f"Memory from {self.name} - item {i} - {np.random.randint(100000)}"
                embedding = np.random.rand(1536).astype(np.float32)
                mem_id = self.db.insert_memory(content, embedding=embedding)
                ids.append(mem_id)
        self.refresh_metadata()
        return ids

    def update_memories(self, ids):
        with patch("memk.storage.db.GLOBAL_HLC.node_id", self.node_id):
            for mid in ids:
                new_content = f"Updated content for {mid[:8]} by {self.name}"
                self.db.update_memory_content(mid, new_content)
        self.refresh_metadata()

    def archive_memories(self, ids):
        with patch("memk.storage.db.GLOBAL_HLC.node_id", self.node_id):
            for mid in ids:
                self.db.archive_memory(mid)
        self.refresh_metadata()

    def unarchive_memories(self, ids):
        with patch("memk.storage.db.GLOBAL_HLC.node_id", self.node_id):
            for mid in ids:
                self.db.unarchive_memory(mid)
        self.refresh_metadata()

    def refresh_metadata(self):
        """Update Merkle tree state based on current row_hashes."""
        now_ms = int(time.time() * 1000)
        self.merkle.cleanup_stale_row_hashes()
        return self.merkle.rebuild_or_refresh_merkle_buckets(now_ms)

    def sync_from(self, remote_node):
        """Simulate pulling delta changes from a remote node."""
        logger.info(f"[{self.name}] Syncing from [{remote_node.name}] (Remote ID: {remote_node.node_id})")
        
        remote_buckets = remote_node.protocol.get_bucket_hashes()
        mismatched = self.protocol.diff_buckets(remote_buckets)
        
        if not mismatched:
            logger.info(f"[{self.name}] Already consistent with [{remote_node.name}]")
            return 0
            
        deltas = remote_node.protocol.fetch_delta_for_buckets(mismatched)
        logger.info(f"[{self.name}] Fetched {len(deltas)} delta items")
        
        self.protocol.apply_remote_delta(deltas, remote_replica_id=remote_node.node_id)
        
        # Refresh local merkle state
        self.refresh_metadata()
        
        return len(deltas)

    def run_gc(self):
        """Execute oplog garbage collection with 0 retention for testing."""
        return OplogGC.run_oplog_gc_job(self.db, retention_seconds=0)

    def get_semantic_state(self):
        """Extract a simplified state hash for comparison."""
        mems = self.db.get_all_memories()
        # Sort by id to ensure deterministic order
        mems.sort(key=lambda x: x["id"])
        
        state = []
        for m in mems:
            # We track content and archive status
            state.append({
                "id": m["id"],
                "content": m["content"],
                "archived": m["archived"],
                "v_hlc": m["version_hlc"]
            })
        return state

def test_delta_sync_hardening_stress():
    """
    Stress test covering the entire Delta Sync lifecycle:
    Writes -> Updates -> Archival -> Sync -> Checkpoint -> GC -> Consistency.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path_a = os.path.join(tmpdir, "replica_a.db")
        path_b = os.path.join(tmpdir, "replica_b.db")
        
        node_a = ReplicaNode("Replica-A", "node_alpha", path_a)
        node_b = ReplicaNode("Replica-B", "node_beta", path_b)
        
        logger.info("=== PHASE 1: Initial Data Load and Sync ===")
        # A generates 100 items
        ids_a = node_a.add_memories(100)
        
        # Initial sync A -> B
        count = node_b.sync_from(node_a)
        assert count == 100
        
        # Verify initial consistency
        assert node_a.get_semantic_state() == node_b.get_semantic_state()
        logger.info("✓ Phase 1 consistency verified")

        logger.info("=== PHASE 2: Updates and Archival ===")
        # A updates 20 items
        node_a.update_memories(ids_a[:20])
        
        # A archives 10 items
        node_a.archive_memories(ids_a[20:30])
        
        # A unarchives 2 items
        node_a.unarchive_memories(ids_a[20:22])
        
        # Sync the changes to B
        node_b.sync_from(node_a)
        
        # Verify state after modifications
        state_a = node_a.get_semantic_state()
        state_b = node_b.get_semantic_state()
        assert state_a == state_b
        
        # Check specific archived items in B
        mems_b = {m["id"]: m for m in node_b.db.get_all_memories()}
        assert mems_b[ids_a[25]]["archived"] == 1
        assert mems_b[ids_a[21]]["archived"] == 0
        logger.info("✓ Phase 2 updates and archive state verified")

        logger.info("=== PHASE 3: Checkpoint and GC Safety ===")
        # Verify B has a valid checkpoint for A
        with node_b.db._get_connection() as conn:
            cp = conn.execute("SELECT * FROM replica_checkpoint WHERE replica_id = 'node_alpha'").fetchone()
            assert cp is not None
            b_last_hlc = cp["last_applied_hlc"]
            logger.info(f"Replica B checkpoint for A: {b_last_hlc}")

        # Simulate B sending checkpoint feedback to A
        # In a real system, this happens via a 'handshake' or 'ack' message.
        with node_a.db._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO replica_checkpoint (replica_id, last_applied_hlc, last_applied_node, last_applied_seq, updated_ts) VALUES (?, ?, ?, ?, ?)",
                ("node_beta", cp["last_applied_hlc"], cp["last_applied_node"], cp["last_applied_seq"], cp["updated_ts"])
            )

        # Before GC, create 10 more items in A (Not yet synced to B)
        # These MUST NOT be pruned even if they are old (though here they are new)
        node_a.add_memories(10)
        
        with node_a.db._get_connection() as conn:
            count_pre = conn.execute("SELECT COUNT(*) FROM oplog").fetchone()[0]
            
        # Run GC on A
        logger.info("Running Oplog GC on A...")
        node_a.run_gc()
        
        with node_a.db._get_connection() as conn:
            count_post = conn.execute("SELECT COUNT(*) FROM oplog").fetchone()[0]
            # Some items should be pruned (synced items)
            # The last 10 items (not yet synced) must remain in oplog.
            assert count_post >= 10
            assert count_post < count_pre
            logger.info(f"Oplog pruned: {count_pre} -> {count_post}")

        logger.info("=== PHASE 4: Final Sync and BLOB check ===")
        # Final sync to get those last 10 items to B
        node_b.sync_from(node_a)
        
        assert node_a.get_semantic_state() == node_b.get_semantic_state()
        
        # Verify BLOB (embeddings) were not corrupted
        mems_a = node_a.db.get_all_memories()
        mems_b_list = node_b.db.get_all_memories()
        b_map = {m["id"]: m for m in mems_b_list}
        
        for ma in mems_a:
            mb = b_map[ma["id"]]
            assert ma["embedding"] == mb["embedding"], "BLOB/Embedding data mismatch!"

        logger.info("✓ PHASE 4: Final consistency and BLOB integrity verified")
        logger.info("=== ALL STRESS TESTS PASSED ===")

if __name__ == "__main__":
    test_delta_sync_hardening_stress()
