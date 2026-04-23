import pytest
import os
import json
import tempfile
import shutil

from memk.storage.db import MemoryDB
from memk.sync.conflict import ConflictRepository, _safe_json


@pytest.fixture
def repo():
    tmp = tempfile.mkdtemp()
    db = MemoryDB(os.path.join(tmp, "test_conflict.db"))
    db.init_db()
    repo = ConflictRepository(db)
    yield repo
    try:
        shutil.rmtree(tmp)
    except:
        pass


# ------------------------------------------------------------------
# Schema
# ------------------------------------------------------------------

def test_schema_migration_creates_table(repo):
    """Confirm that init_db (auto_migrate) created the conflict_record table."""
    with repo.db._get_connection() as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "conflict_record" in tables


def test_schema_indexes_exist(repo):
    """Verify the status and row lookup indexes were created."""
    with repo.db._get_connection() as conn:
        indexes = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
    assert "idx_conflict_status" in indexes
    assert "idx_conflict_row" in indexes


# ------------------------------------------------------------------
# Create
# ------------------------------------------------------------------

def test_create_conflict_record(repo):
    cid = repo.create_conflict_record(
        table_name="memories",
        row_id="mem_001",
        local_hlc=100,
        remote_hlc=200,
        local_payload={"id": "mem_001", "content": "local version"},
        remote_payload={"id": "mem_001", "content": "remote version"},
    )
    assert cid is not None

    record = repo.get_conflict_by_id(cid)
    assert record is not None
    assert record["table_name"] == "memories"
    assert record["row_id"] == "mem_001"
    assert record["local_hlc"] == 100
    assert record["remote_hlc"] == 200
    assert record["status"] == "open"
    assert record["resolution"] is None

    # Snapshots are valid JSON
    local_snap = json.loads(record["local_snapshot"])
    remote_snap = json.loads(record["remote_snapshot"])
    assert local_snap["content"] == "local version"
    assert remote_snap["content"] == "remote version"


def test_create_conflict_handles_blob_payload(repo):
    """Embedding bytes should be hex-encoded in the snapshot, not crash."""
    cid = repo.create_conflict_record(
        table_name="memories",
        row_id="mem_blob",
        local_hlc=10,
        remote_hlc=20,
        local_payload={"id": "mem_blob", "embedding": b"\x00\x01\x02\x03"},
        remote_payload={"id": "mem_blob", "embedding": b"\x04\x05\x06\x07"},
    )
    record = repo.get_conflict_by_id(cid)
    local_snap = json.loads(record["local_snapshot"])
    remote_snap = json.loads(record["remote_snapshot"])

    # bytes → hex string
    assert local_snap["embedding"] == "00010203"
    assert remote_snap["embedding"] == "04050607"


# ------------------------------------------------------------------
# List / Filter
# ------------------------------------------------------------------

def test_list_open_conflicts(repo):
    repo.create_conflict_record("memories", "m1", 1, 2, {}, {})
    repo.create_conflict_record("facts", "f1", 3, 4, {}, {})
    repo.create_conflict_record("memories", "m2", 5, 6, {}, {})

    # All open
    all_open = repo.list_open_conflicts()
    assert len(all_open) == 3

    # Filtered by table
    mem_only = repo.list_open_conflicts(table_name="memories")
    assert len(mem_only) == 2
    assert all(r["table_name"] == "memories" for r in mem_only)


def test_list_conflicts_for_row(repo):
    cid1 = repo.create_conflict_record("memories", "m1", 1, 2, {"v": 1}, {})
    cid2 = repo.create_conflict_record("memories", "m1", 3, 4, {"v": 2}, {})
    repo.create_conflict_record("memories", "m2", 5, 6, {}, {})

    rows = repo.list_conflicts_for_row("memories", "m1")
    assert len(rows) == 2
    ids = {r["conflict_id"] for r in rows}
    assert cid1 in ids
    assert cid2 in ids


# ------------------------------------------------------------------
# Resolve / Ignore
# ------------------------------------------------------------------

def test_mark_conflict_resolved(repo):
    cid = repo.create_conflict_record("memories", "m1", 1, 2, {}, {})

    ok = repo.mark_conflict_resolved(cid, "manual_local")
    assert ok is True

    record = repo.get_conflict_by_id(cid)
    assert record["status"] == "resolved"
    assert record["resolution"] == "manual_local"

    # Re-resolving an already resolved conflict does nothing
    ok2 = repo.mark_conflict_resolved(cid, "merged")
    assert ok2 is False


def test_mark_conflict_ignored(repo):
    cid = repo.create_conflict_record("facts", "f1", 10, 20, {}, {})

    ok = repo.mark_conflict_ignored(cid)
    assert ok is True

    record = repo.get_conflict_by_id(cid)
    assert record["status"] == "ignored"
    assert record["resolution"] == "user_ignored"


def test_resolved_conflicts_excluded_from_open_list(repo):
    cid1 = repo.create_conflict_record("memories", "m1", 1, 2, {}, {})
    cid2 = repo.create_conflict_record("memories", "m2", 3, 4, {}, {})

    repo.mark_conflict_resolved(cid1, "lww_accepted")

    open_list = repo.list_open_conflicts()
    assert len(open_list) == 1
    assert open_list[0]["conflict_id"] == cid2


def test_get_nonexistent_conflict_returns_none(repo):
    assert repo.get_conflict_by_id("does_not_exist") is None


# ------------------------------------------------------------------
# _safe_json helper
# ------------------------------------------------------------------

def test_safe_json_with_mixed_types():
    result = _safe_json({
        "text": "hello",
        "num": 42,
        "blob": b"\xff\xfe",
        "nested": None,
    })
    parsed = json.loads(result)
    assert parsed["text"] == "hello"
    assert parsed["num"] == 42
    assert parsed["blob"] == "fffe"
    assert parsed["nested"] is None
