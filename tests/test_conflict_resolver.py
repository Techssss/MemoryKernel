"""
Tests for ConflictResolver — the three resolution primitives.

Scenarios tested:
1. keep_local when LWW already kept local (no data change)
2. keep_local when LWW applied remote (data reverted to local)
3. accept_remote when LWW already applied remote (no data change)
4. accept_remote when LWW kept local (data reverted to remote)
5. ignore (pure status change, no data mutation)
6. resolving an already-resolved conflict fails gracefully
7. resolved_ts is populated
"""

import pytest
import os
import json
import tempfile
import shutil

from memk.storage.db import MemoryDB
from memk.sync.conflict import ConflictRepository
from memk.sync.resolver import ConflictResolver


@pytest.fixture
def env():
    """Standalone DB with a pre-inserted memory and conflict repo/resolver."""
    tmp = tempfile.mkdtemp()
    db = MemoryDB(os.path.join(tmp, "resolver_test.db"))
    db.init_db()
    repo = ConflictRepository(db)
    resolver = ConflictResolver(db)
    yield db, repo, resolver
    try:
        shutil.rmtree(tmp)
    except:
        pass


def _insert_memory_raw(db, mem_id, content, hlc):
    """Helper: insert a memory row with a specific id and hlc."""
    with db._get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO memories
                (id, content, created_at, version_hlc, version_node, version_seq)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (mem_id, content, "2024-01-01T00:00:00", hlc, "test", 0),
        )


# ------------------------------------------------------------------
# 1. keep_local — LWW already kept local
# ------------------------------------------------------------------

def test_keep_local_noop_when_lww_agreed(env):
    """
    local_hlc > remote_hlc → LWW kept local.
    keep_local should just mark resolved, no data change.
    """
    db, repo, resolver = env
    row_id = "mem_001"
    _insert_memory_raw(db, row_id, "Local wins", 200)

    cid = repo.create_conflict_record(
        table_name="memories",
        row_id=row_id,
        local_hlc=200,
        remote_hlc=100,
        local_payload={"id": row_id, "content": "Local wins", "version_hlc": 200},
        remote_payload={"id": row_id, "content": "Remote lost", "version_hlc": 100},
    )

    result = resolver.resolve_keep_local(cid)

    assert result["ok"] is True
    assert result["strategy"] == "keep_local"
    assert result["data_reverted"] is False

    # Data unchanged
    row = db.get_memory_by_id(row_id)
    assert row["content"] == "Local wins"

    # Conflict marked resolved
    record = repo.get_conflict_by_id(cid)
    assert record["status"] == "resolved"
    assert record["resolution"] == "keep_local"
    assert record["resolved_ts"] is not None


# ------------------------------------------------------------------
# 2. keep_local — LWW applied remote, must revert
# ------------------------------------------------------------------

def test_keep_local_reverts_when_lww_chose_remote(env):
    """
    remote_hlc > local_hlc → LWW applied remote.
    keep_local should force-write local snapshot back.
    """
    db, repo, resolver = env
    row_id = "mem_002"

    # Current DB state = remote version (LWW winner)
    _insert_memory_raw(db, row_id, "Remote text", 300)

    cid = repo.create_conflict_record(
        table_name="memories",
        row_id=row_id,
        local_hlc=200,
        remote_hlc=300,
        local_payload={
            "id": row_id,
            "content": "Original local text",
            "version_hlc": 200,
            "version_node": "local_node",
            "version_seq": 0,
        },
        remote_payload={
            "id": row_id,
            "content": "Remote text",
            "version_hlc": 300,
        },
    )

    result = resolver.resolve_keep_local(cid)

    assert result["ok"] is True
    assert result["data_reverted"] is True

    # Data reverted to local snapshot
    row = db.get_memory_by_id(row_id)
    assert row["content"] == "Original local text"
    assert row["version_hlc"] == 200


# ------------------------------------------------------------------
# 3. accept_remote — LWW already applied remote
# ------------------------------------------------------------------

def test_accept_remote_noop_when_lww_agreed(env):
    """
    remote_hlc >= local_hlc → LWW already applied remote.
    accept_remote should just mark resolved.
    """
    db, repo, resolver = env
    row_id = "mem_003"
    _insert_memory_raw(db, row_id, "Remote applied", 300)

    cid = repo.create_conflict_record(
        table_name="memories",
        row_id=row_id,
        local_hlc=200,
        remote_hlc=300,
        local_payload={"id": row_id, "content": "Old local", "version_hlc": 200},
        remote_payload={"id": row_id, "content": "Remote applied", "version_hlc": 300},
    )

    result = resolver.resolve_accept_remote(cid)

    assert result["ok"] is True
    assert result["strategy"] == "accept_remote"
    assert result["data_reverted"] is False

    row = db.get_memory_by_id(row_id)
    assert row["content"] == "Remote applied"


# ------------------------------------------------------------------
# 4. accept_remote — LWW kept local, must force-apply remote
# ------------------------------------------------------------------

def test_accept_remote_overwrites_when_lww_kept_local(env):
    """
    local_hlc > remote_hlc → LWW kept local.
    accept_remote should force-write the remote snapshot.
    """
    db, repo, resolver = env
    row_id = "mem_004"

    # Current DB = local version (LWW winner)
    _insert_memory_raw(db, row_id, "Local kept by LWW", 400)

    cid = repo.create_conflict_record(
        table_name="memories",
        row_id=row_id,
        local_hlc=400,
        remote_hlc=350,
        local_payload={"id": row_id, "content": "Local kept by LWW", "version_hlc": 400},
        remote_payload={
            "id": row_id,
            "content": "Remote I actually want",
            "version_hlc": 350,
            "version_node": "remote_node",
            "version_seq": 1,
        },
    )

    result = resolver.resolve_accept_remote(cid)

    assert result["ok"] is True
    assert result["data_reverted"] is True

    row = db.get_memory_by_id(row_id)
    assert row["content"] == "Remote I actually want"
    assert row["version_hlc"] == 350


# ------------------------------------------------------------------
# 5. ignore — no data change
# ------------------------------------------------------------------

def test_resolve_ignore(env):
    db, repo, resolver = env
    row_id = "mem_005"
    _insert_memory_raw(db, row_id, "Some content", 100)

    cid = repo.create_conflict_record(
        table_name="memories",
        row_id=row_id,
        local_hlc=100,
        remote_hlc=200,
        local_payload={"id": row_id, "content": "Some content", "version_hlc": 100},
        remote_payload={"id": row_id, "content": "Other content", "version_hlc": 200},
    )

    result = resolver.resolve_ignore(cid)

    assert result["ok"] is True
    assert result["strategy"] == "ignored"
    assert result["data_reverted"] is False

    record = repo.get_conflict_by_id(cid)
    assert record["status"] == "resolved"
    assert record["resolution"] == "ignored"


# ------------------------------------------------------------------
# 6. Already resolved → fails gracefully
# ------------------------------------------------------------------

def test_resolve_already_resolved_returns_false(env):
    db, repo, resolver = env

    cid = repo.create_conflict_record(
        "memories", "m1", 10, 20, {}, {}
    )
    # Resolve once
    resolver.resolve_ignore(cid)

    # Try again
    r1 = resolver.resolve_keep_local(cid)
    assert r1["ok"] is False

    r2 = resolver.resolve_accept_remote(cid)
    assert r2["ok"] is False

    r3 = resolver.resolve_ignore(cid)
    assert r3["ok"] is False


def test_resolve_nonexistent_id(env):
    _, _, resolver = env
    result = resolver.resolve_keep_local("does_not_exist")
    assert result["ok"] is False


# ------------------------------------------------------------------
# 7. resolved_ts is populated
# ------------------------------------------------------------------

def test_resolved_ts_is_set(env):
    db, repo, resolver = env

    cid = repo.create_conflict_record(
        "memories", "m1", 10, 20,
        {"id": "m1", "content": "a", "version_hlc": 10},
        {"id": "m1", "content": "b", "version_hlc": 20},
    )

    resolver.resolve_keep_local(cid)
    record = repo.get_conflict_by_id(cid)

    assert record["resolved_ts"] is not None
    assert "T" in record["resolved_ts"]  # ISO format check
