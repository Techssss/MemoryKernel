"""
memk.sync.conflict
==================
Repository for recording and managing semantic conflicts detected during
LWW-based synchronization.

When to use conflict records instead of silent LWW overwrite
------------------------------------------------------------
LWW is sufficient when:
  - The newer version clearly supersedes the older (e.g. a corrected typo).
  - The data is machine-generated and deterministic.

Conflict records should be created when:
  - Both sides modified the *same row* independently (concurrent edits with
    different content but close timestamps).
  - The overwritten local data contains user-authored text, decision notes,
    or judgment that can't be reconstructed from the incoming version alone.
  - The semantic difference between local and remote payloads is non-trivial
    (e.g. different fact objects for the same subject+predicate).

The calling code (e.g. apply_remote_delta) decides whether to record a
conflict.  This module only provides the storage layer.
"""

import json
import uuid
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from memk.storage.db import MemoryDB, DatabaseError

logger = logging.getLogger("memk.sync")


def _safe_json(payload: dict) -> str:
    """Serialize a row payload to JSON, hex-encoding any bytes values."""
    sanitized = {}
    for k, v in payload.items():
        if isinstance(v, (bytes, bytearray)):
            sanitized[k] = v.hex()
        else:
            sanitized[k] = v
    return json.dumps(sanitized, sort_keys=True, default=str)


class ConflictRepository:
    """DAO for the conflict_record table."""

    def __init__(self, db: MemoryDB):
        self.db = db

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def create_conflict_record(
        self,
        table_name: str,
        row_id: str,
        local_hlc: int,
        remote_hlc: int,
        local_payload: dict,
        remote_payload: dict,
    ) -> str:
        """
        Persist a new conflict snapshot.

        Parameters
        ----------
        table_name     : Which semantic table the conflict belongs to.
        row_id         : The entity id in that table.
        local_hlc      : version_hlc of the local row *before* overwrite.
        remote_hlc     : version_hlc of the incoming remote row.
        local_payload  : Full dict snapshot of the local row before overwrite.
        remote_payload : Full dict snapshot of the incoming remote row.

        Returns
        -------
        conflict_id : str  — UUID of the newly created record.
        """
        conflict_id = str(uuid.uuid4())
        detected_ts = datetime.now(timezone.utc).isoformat()

        local_snap = _safe_json(local_payload)
        remote_snap = _safe_json(remote_payload)

        try:
            with self.db.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO conflict_record
                        (conflict_id, table_name, row_id,
                         local_hlc, remote_hlc,
                         local_snapshot, remote_snapshot,
                         detected_ts, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open')
                    """,
                    (
                        conflict_id, table_name, row_id,
                        local_hlc, remote_hlc,
                        local_snap, remote_snap,
                        detected_ts,
                    ),
                )
            return conflict_id
        except sqlite3.Error as e:
            raise DatabaseError(f"create_conflict_record failed: {e}") from e

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_open_conflicts(
        self, table_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return all unresolved conflict records, optionally filtered by table."""
        try:
            with self.db.connection() as conn:
                if table_name:
                    rows = conn.execute(
                        """
                        SELECT * FROM conflict_record
                        WHERE status = 'open' AND table_name = ?
                        ORDER BY detected_ts DESC
                        """,
                        (table_name,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT * FROM conflict_record
                        WHERE status = 'open'
                        ORDER BY detected_ts DESC
                        """
                    ).fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"list_open_conflicts failed: {e}") from e

    def get_conflict_by_id(self, conflict_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single conflict by its ID."""
        try:
            with self.db.connection() as conn:
                row = conn.execute(
                    "SELECT * FROM conflict_record WHERE conflict_id = ?",
                    (conflict_id,),
                ).fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            raise DatabaseError(f"get_conflict_by_id failed: {e}") from e

    def list_conflicts_for_row(
        self, table_name: str, row_id: str
    ) -> List[Dict[str, Any]]:
        """All conflicts (any status) that involve a specific entity."""
        try:
            with self.db.connection() as conn:
                rows = conn.execute(
                    """
                    SELECT * FROM conflict_record
                    WHERE table_name = ? AND row_id = ?
                    ORDER BY detected_ts DESC
                    """,
                    (table_name, row_id),
                ).fetchall()
                return [dict(r) for r in rows]
        except sqlite3.Error as e:
            raise DatabaseError(f"list_conflicts_for_row failed: {e}") from e

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def mark_conflict_resolved(
        self,
        conflict_id: str,
        resolution: str = "lww_accepted",
    ) -> bool:
        """
        Close a conflict record.

        Parameters
        ----------
        conflict_id : The conflict to resolve.
        resolution  : Free-text label describing how it was resolved.
                      Examples: 'lww_accepted', 'manual_local', 'manual_remote',
                                'merged', 'ignored'.

        Returns
        -------
        True if a row was actually updated, False if conflict_id not found.
        """
        try:
            with self.db.connection() as conn:
                cur = conn.execute(
                    """
                    UPDATE conflict_record
                    SET status = 'resolved', resolution = ?
                    WHERE conflict_id = ? AND status = 'open'
                    """,
                    (resolution, conflict_id),
                )
                return cur.rowcount > 0
        except sqlite3.Error as e:
            raise DatabaseError(f"mark_conflict_resolved failed: {e}") from e

    def mark_conflict_ignored(self, conflict_id: str) -> bool:
        """Mark a conflict as intentionally ignored."""
        try:
            with self.db.connection() as conn:
                cur = conn.execute(
                    """
                    UPDATE conflict_record
                    SET status = 'ignored', resolution = 'user_ignored'
                    WHERE conflict_id = ? AND status = 'open'
                    """,
                    (conflict_id,),
                )
                return cur.rowcount > 0
        except sqlite3.Error as e:
            raise DatabaseError(f"mark_conflict_ignored failed: {e}") from e


# ---------------------------------------------------------------------------
# Conflict Detector — pure-function rules, no DB writes
# ---------------------------------------------------------------------------

class ConflictDetector:
    """
    Evaluates whether an incoming remote delta constitutes a semantic
    conflict worth recording, even though LWW will still decide the
    final state.

    Detection rules (all must be cheap and deterministic):

    Rule 1 — Concurrent Divergent Edit
        Both local and remote have version_hlc > 0 (i.e. the row exists
        on both sides with independent modifications), AND local was NOT
        simply the original unmodified seed.  The key signal is that
        local_hlc != remote_hlc AND both are non-zero.

    Rule 2 — Cross-State Mutation
        The archived/active flag changed on one side while content
        changed on the other.  This suggests two humans did different
        things to the same record concurrently.

    Rule 3 — Significant Text Divergence
        When both sides modified a text field (content for memories,
        object for facts), and the actual text differs, the overwritten
        version may carry unique semantic value.

    Each rule returns a short reason string or None.
    """

    # Text fields we compare per table
    TEXT_FIELDS = {
        "memories": "content",
        "facts": "object",
        "kg_fact": "canonical_text",
    }

    @classmethod
    def detect(cls, table: str, local_row: dict, remote_row: dict) -> Optional[str]:
        """
        Run all rules. Return the *first* matching reason string,
        or None if no conflict is detected.

        Parameters
        ----------
        table      : Table name (memories, facts, …).
        local_row  : The existing local row (dict from SELECT *).
        remote_row : The incoming remote payload (dict).
        """
        reason = cls._rule_concurrent_divergent(local_row, remote_row)
        if reason:
            # Only flag as conflict if there is also a content difference
            content_reason = cls._rule_text_divergence(table, local_row, remote_row)
            if content_reason:
                return content_reason
            cross = cls._rule_cross_state(local_row, remote_row)
            if cross:
                return cross
            # Both modified but text is identical — no real conflict
            return None

        return None

    # ------------------------------------------------------------------
    # Individual rules
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_concurrent_divergent(local: dict, remote: dict) -> Optional[str]:
        """
        Rule 1: Both sides independently wrote to the same row.
        We detect this when both have version_hlc > 0 and they differ.
        """
        l_hlc = local.get("version_hlc", 0) or 0
        r_hlc = remote.get("version_hlc", 0) or 0

        if l_hlc > 0 and r_hlc > 0 and l_hlc != r_hlc:
            return "concurrent_divergent_edit"
        return None

    @staticmethod
    def _rule_cross_state(local: dict, remote: dict) -> Optional[str]:
        """
        Rule 2: archived flag changed on one side, content on the other.
        """
        l_arch = local.get("archived")
        r_arch = remote.get("archived")
        if l_arch is None or r_arch is None:
            return None

        archived_changed = (l_arch != r_arch)
        if not archived_changed:
            return None

        # Check if content also differs
        for field in ("content", "object", "canonical_text"):
            l_val = local.get(field)
            r_val = remote.get(field)
            if l_val is not None and r_val is not None and l_val != r_val:
                return "cross_state_mutation"

        return None

    @classmethod
    def _rule_text_divergence(cls, table: str, local: dict, remote: dict) -> Optional[str]:
        """
        Rule 3: The primary text field differs between local and remote.
        """
        field = cls.TEXT_FIELDS.get(table)
        if not field:
            return None

        l_text = local.get(field)
        r_text = remote.get(field)

        if l_text is None or r_text is None:
            return None

        # Normalize to strings for comparison (BLOB safety)
        l_str = l_text if isinstance(l_text, str) else str(l_text)
        r_str = r_text if isinstance(r_text, str) else str(r_text)

        if l_str != r_str:
            return "text_divergence"

        return None
