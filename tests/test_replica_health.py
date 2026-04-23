import pytest
import tempfile
import os
import shutil
from memk.storage.db import MemoryDB
from memk.sync.health import ReplicaHealthService, SyncState

@pytest.fixture
def db():
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    db = MemoryDB(db_path)
    db.init_db()
    yield db
    # Cleanup
    try:
        db = None # Release connection
        shutil.rmtree(tmpdir)
    except:
        pass

def test_replica_health_states(db):
    health = ReplicaHealthService(db)
    replica_id = "node_beta"
    
    # CASE 0: No checkpoint
    assert health.get_replica_sync_state(replica_id) == SyncState.UNKNOWN
    assert health.is_replica_stale(replica_id) is True

    # CASE 1: Fresh (No data yet)
    # latest_hlc is 0, if checkpoint is 0 -> FRESH
    db.upsert_replica_checkpoint(replica_id, 0, "unknown", 0)
    assert health.get_replica_sync_state(replica_id) == SyncState.FRESH
    
    # CASE 2: Lagging
    # Let's add some data to local DB
    m_id = db.insert_memory("Fact 1") # This logs to oplog and sets latest_hlc
    latest_hlc = db.get_latest_version_hlc()
    assert latest_hlc > 0
    
    # Replica is at 0, Oplog starts at 1
    # Checkpoint is 0, latest is H. Oplog min is H (since only 1 entry)
    # Wait, if replica_hlc (0) < min_op_hlc (H), it should be STALE?
    # Yes, because Oplog doesn't have the "0 -> H" transition explicitly as a log entry starting from 0?
    # Actually, the first entry in Oplog HAS the first HLC.
    # If replica needs HLC 10, and Oplog min is 10, it's fine.
    # If replica is at 5, and Oplog min is 10, it missed 6,7,8,9 -> STALE.
    
    op_range = db.get_oplog_range()
    min_op = op_range["min"]
    
    # Set replica at exactly min_op - it's still lagging (needs everything from min_op onwards)
    db.upsert_replica_checkpoint(replica_id, min_op, "unknown", 0)
    # If replica_hlc == min_op, it means it HAS applied min_op.
    # So it is LAGGING if there are more entries, or FRESH if min_op is the latest.
    assert health.get_replica_sync_state(replica_id) == SyncState.FRESH

    # Add another one
    db.insert_memory("Fact 2")
    new_latest = db.get_latest_version_hlc()
    
    # Replica is still at min_op. New stuff is at new_latest.
    # Oplog has [min_op, new_latest]
    assert health.get_replica_sync_state(replica_id) == SyncState.LAGGING
    assert health.is_replica_stale(replica_id) is False

    # CASE 3: Stale (Pruned)
    # Manually prune the first op
    db.prune_oplog_entries(new_latest) # Prune everything before new_latest
    
    # Now oplog min is new_latest.
    # Replica is still at min_op.
    # min_op < new_latest (pruned) -> STALE
    assert health.get_replica_sync_state(replica_id) == SyncState.STALE
    assert health.is_replica_stale(replica_id) is True

    # CASE 4: Fresh again
    db.upsert_replica_checkpoint(replica_id, new_latest, "unknown", 0)
    assert health.get_replica_sync_state(replica_id) == SyncState.FRESH

def test_health_report(db):
    health = ReplicaHealthService(db)
    db.upsert_replica_checkpoint("node_beta", 100, "node_x", 1)
    report = health.get_health_report("node_beta")
    
    assert report["replica_id"] == "node_beta"
    assert "state" in report
    assert "hlc_lag" in report

def test_choose_sync_mode(db):
    from memk.sync.health import SyncMode
    health = ReplicaHealthService(db)
    replica_id = "node_gamma"
    
    # 1. Unknown -> Merkle
    decision = health.choose_sync_mode(replica_id)
    assert decision["mode"] == SyncMode.MERKLE_RECOVERY.value
    assert "First-time" in decision["reason"]
    
    # 2. Fresh -> No Op
    db.upsert_replica_checkpoint(replica_id, 0, "n", 0)
    decision = health.choose_sync_mode(replica_id)
    assert decision["mode"] == SyncMode.NO_OP.value
    
    # 3. Lagging -> Oplog
    # First, make it fresh at current state
    db.insert_memory("Baseline")
    latest = db.get_latest_version_hlc()
    db.upsert_replica_checkpoint(replica_id, latest, "n", 0)
    
    # Now add another op to make it lag by 1
    db.insert_memory("Op 1")
    decision = health.choose_sync_mode(replica_id)
    assert decision["mode"] == SyncMode.OPLOG_DELTA.value
    assert "local Oplog" in decision["reason"]
    
    # 4. Stale -> Merkle
    # Prune including Op 1
    new_latest = db.get_latest_version_hlc()
    db.insert_memory("Op 2")
    final_latest = db.get_latest_version_hlc()
    db.prune_oplog_entries(final_latest) # Prune everything before Op 2
    
    # Replica still at latest (Op 1). But Oplog starts at final_latest (Op 2).
    # It missed Op 2? No, wait. 
    # If replica is at latest, it needs everything > latest.
    # If Oplog starts at final_latest, and final_latest > latest.
    # If there was an entry between latest and final_latest that was pruned, it's stale.
    decision = health.choose_sync_mode(replica_id)
    assert decision["mode"] == SyncMode.MERKLE_RECOVERY.value
    assert "stale" in decision["reason"]
