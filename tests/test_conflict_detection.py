"""
Tests for conflict detection integrated into apply_remote_delta.

Verifies that:
1. Simple overwrites produce NO conflict records.
2. Divergent concurrent edits ARE recorded as conflicts.
3. LWW still determines the final state regardless of conflict.
4. Conflict detection is opt-in (off by default).
5. Cross-state mutations (archived vs content) are detected.
"""

import pytest
import os
import json
import tempfile
import shutil

from memk.storage.db import MemoryDB
from memk.sync.merkle import MerkleService
from memk.sync.protocol import SyncProtocolNode
from memk.sync.conflict import ConflictRepository, ConflictDetector


class MockRuntime:
    def __init__(self, db, workspace_id="test_ws"):
        self.db = db
        self.workspace_id = workspace_id


@pytest.fixture
def setup():
    """Create two isolated replicas with full sync infra."""
    tmp_a = tempfile.mkdtemp()
    tmp_b = tempfile.mkdtemp()

    db_a = MemoryDB(os.path.join(tmp_a, "a.db"))
    db_b = MemoryDB(os.path.join(tmp_b, "b.db"))
    db_a.init_db()
    db_b.init_db()

    node_a = SyncProtocolNode(MerkleService(MockRuntime(db_a, "a")))
    node_b = SyncProtocolNode(MerkleService(MockRuntime(db_b, "b")))
    conflict_repo_b = ConflictRepository(db_b)

    yield node_a, node_b, db_a, db_b, conflict_repo_b

    try:
        shutil.rmtree(tmp_a)
        shutil.rmtree(tmp_b)
    except:
        pass


# ------------------------------------------------------------------
# 1. Simple overwrite — no conflict
# ------------------------------------------------------------------

def test_no_conflict_on_simple_new_row(setup):
    """Applying a brand-new row to an empty DB should never create a conflict."""
    node_a, node_b, db_a, db_b, conflict_repo = setup

    mem_id = db_a.insert_memory("Hello from A")
    row = db_a.get_memory_by_id(mem_id)

    deltas = [{"table": "memories", "row_id": mem_id, "payload": row}]
    node_b.apply_remote_delta(deltas, detect_conflicts=True)

    assert db_b.get_memory_by_id(mem_id)["content"] == "Hello from A"
    assert len(conflict_repo.list_open_conflicts()) == 0


def test_no_conflict_when_text_identical(setup):
    """If both sides have the same text, even with different HLCs, no conflict."""
    node_a, node_b, db_a, db_b, conflict_repo = setup

    # Insert same content at B first
    mem_id = db_b.insert_memory("Same content")
    local_row = db_b.get_memory_by_id(mem_id)

    # Build a remote delta with same content but different (higher) HLC
    remote_payload = dict(local_row)
    remote_payload["version_hlc"] = local_row["version_hlc"] + 100

    deltas = [{"table": "memories", "row_id": mem_id, "payload": remote_payload}]
    node_b.apply_remote_delta(deltas, detect_conflicts=True)

    assert len(conflict_repo.list_open_conflicts()) == 0


# ------------------------------------------------------------------
# 2. Divergent concurrent edits — conflict recorded
# ------------------------------------------------------------------

def test_conflict_logged_on_divergent_content(setup):
    """Two independent edits to the same row with different content → conflict."""
    node_a, node_b, db_a, db_b, conflict_repo = setup

    # B has its own version
    mem_id = db_b.insert_memory("Local version by B")
    local_row = db_b.get_memory_by_id(mem_id)
    local_hlc = local_row["version_hlc"]

    # A has a different version (higher HLC, different content)
    remote_payload = dict(local_row)
    remote_payload["content"] = "Remote version by A"
    remote_payload["version_hlc"] = local_hlc + 50

    deltas = [{"table": "memories", "row_id": mem_id, "payload": remote_payload}]
    node_b.apply_remote_delta(deltas, detect_conflicts=True)

    # Conflict should be recorded
    conflicts = conflict_repo.list_open_conflicts()
    assert len(conflicts) == 1

    c = conflicts[0]
    assert c["table_name"] == "memories"
    assert c["row_id"] == mem_id
    assert c["local_hlc"] == local_hlc
    assert c["remote_hlc"] == local_hlc + 50
    assert c["status"] == "open"

    # Snapshots should contain the divergent content
    local_snap = json.loads(c["local_snapshot"])
    remote_snap = json.loads(c["remote_snapshot"])
    assert local_snap["content"] == "Local version by B"
    assert remote_snap["content"] == "Remote version by A"


# ------------------------------------------------------------------
# 3. LWW still wins — state is correct after conflict
# ------------------------------------------------------------------

def test_lww_determines_final_state_despite_conflict(setup):
    """Even when a conflict is logged, the higher HLC version wins."""
    node_a, node_b, db_a, db_b, conflict_repo = setup

    mem_id = db_b.insert_memory("Old local text")
    local_row = db_b.get_memory_by_id(mem_id)
    local_hlc = local_row["version_hlc"]

    # Remote has higher HLC
    remote_payload = dict(local_row)
    remote_payload["content"] = "New remote text"
    remote_payload["version_hlc"] = local_hlc + 100

    deltas = [{"table": "memories", "row_id": mem_id, "payload": remote_payload}]
    node_b.apply_remote_delta(deltas, detect_conflicts=True)

    # Final state = remote (higher HLC)
    result = db_b.get_memory_by_id(mem_id)
    assert result["content"] == "New remote text"

    # And the conflict was recorded for review
    assert len(conflict_repo.list_open_conflicts()) == 1


def test_lww_keeps_local_when_local_is_newer(setup):
    """If local has higher HLC, LWW keeps local. Conflict still recorded."""
    node_a, node_b, db_a, db_b, conflict_repo = setup

    mem_id = db_b.insert_memory("Newer local text")
    local_row = db_b.get_memory_by_id(mem_id)
    local_hlc = local_row["version_hlc"]

    # Remote has LOWER HLC
    remote_payload = dict(local_row)
    remote_payload["content"] = "Older remote text"
    remote_payload["version_hlc"] = local_hlc - 50

    deltas = [{"table": "memories", "row_id": mem_id, "payload": remote_payload}]
    node_b.apply_remote_delta(deltas, detect_conflicts=True)

    # Final state = local (higher HLC, LWW)
    result = db_b.get_memory_by_id(mem_id)
    assert result["content"] == "Newer local text"

    # Conflict was still recorded (the remote had different content)
    assert len(conflict_repo.list_open_conflicts()) == 1


# ------------------------------------------------------------------
# 4. Opt-in behavior
# ------------------------------------------------------------------

def test_no_conflict_when_detection_disabled(setup):
    """Default (detect_conflicts=False) should never touch conflict_record."""
    node_a, node_b, db_a, db_b, conflict_repo = setup

    mem_id = db_b.insert_memory("Local text")
    local_row = db_b.get_memory_by_id(mem_id)

    remote_payload = dict(local_row)
    remote_payload["content"] = "Different remote text"
    remote_payload["version_hlc"] = local_row["version_hlc"] + 100

    deltas = [{"table": "memories", "row_id": mem_id, "payload": remote_payload}]
    # detect_conflicts defaults to False
    node_b.apply_remote_delta(deltas)

    assert len(conflict_repo.list_open_conflicts()) == 0


# ------------------------------------------------------------------
# 5. Cross-state mutation detection
# ------------------------------------------------------------------

def test_conflict_on_cross_state_mutation(setup):
    """Archived changed on one side + content changed on other → conflict."""
    node_a, node_b, db_a, db_b, conflict_repo = setup

    mem_id = db_b.insert_memory("Original content")
    local_row = db_b.get_memory_by_id(mem_id)
    local_hlc = local_row["version_hlc"]

    # Remote: archived it AND changed the content
    remote_payload = dict(local_row)
    remote_payload["archived"] = 1
    remote_payload["content"] = "Archived and edited remotely"
    remote_payload["version_hlc"] = local_hlc + 10

    deltas = [{"table": "memories", "row_id": mem_id, "payload": remote_payload}]
    node_b.apply_remote_delta(deltas, detect_conflicts=True)

    conflicts = conflict_repo.list_open_conflicts()
    assert len(conflicts) == 1


# ------------------------------------------------------------------
# 6. ConflictDetector unit tests (pure functions)
# ------------------------------------------------------------------

class TestConflictDetectorRules:

    def test_no_conflict_new_row(self):
        """Remote row that doesn't exist locally → no local_row → detector not called."""
        # Detector is only called when local_row exists, so this is a no-op.
        assert ConflictDetector.detect("memories", {}, {"version_hlc": 100}) is None

    def test_no_conflict_same_text(self):
        local = {"version_hlc": 10, "content": "same"}
        remote = {"version_hlc": 20, "content": "same"}
        assert ConflictDetector.detect("memories", local, remote) is None

    def test_conflict_different_text(self):
        local = {"version_hlc": 10, "content": "version A"}
        remote = {"version_hlc": 20, "content": "version B"}
        reason = ConflictDetector.detect("memories", local, remote)
        assert reason == "text_divergence"

    def test_conflict_cross_state(self):
        local = {"version_hlc": 10, "content": "original", "archived": 0}
        remote = {"version_hlc": 20, "content": "changed", "archived": 1}
        reason = ConflictDetector.detect("memories", local, remote)
        # text_divergence takes priority (it's checked first inside concurrent_divergent)
        assert reason in ("text_divergence", "cross_state_mutation")

    def test_no_conflict_for_unknown_table(self):
        """Tables not in TEXT_FIELDS get no text_divergence check."""
        local = {"version_hlc": 10}
        remote = {"version_hlc": 20}
        assert ConflictDetector.detect("some_random_table", local, remote) is None

    def test_no_conflict_same_hlc(self):
        """Same HLC = same write = not concurrent."""
        local = {"version_hlc": 10, "content": "A"}
        remote = {"version_hlc": 10, "content": "B"}
        assert ConflictDetector.detect("memories", local, remote) is None
