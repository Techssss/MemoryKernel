"""
memk.sync.resolver
===================
Primitives for resolving recorded conflicts.

LWW has already determined the current DB state at detection time.
These functions let a user or agent explicitly choose which version
to keep, overriding the LWW outcome if needed.

Current limitations
-------------------
- No automatic text merge (accept one side wholesale).
- No undo — once resolved, the discarded snapshot is only preserved
  inside the conflict_record row itself.
- Resolving does NOT propagate to other replicas; the write is local.
  A future sync round will propagate the chosen state via normal LWW.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from memk.storage.db import MemoryDB, DatabaseError
from memk.sync.conflict import ConflictRepository

logger = logging.getLogger("memk.sync")


class ConflictResolver:
    """
    High-level resolution operations built on top of ConflictRepository.

    Each method:
    1. Loads the conflict record.
    2. Applies the chosen snapshot to the semantic table (if needed).
    3. Marks the conflict as resolved with strategy + timestamp.
    """

    def __init__(self, db: MemoryDB):
        self.db = db
        self.repo = ConflictRepository(db)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_keep_local(self, conflict_id: str) -> Dict[str, Any]:
        """
        Keep the local version.

        If LWW already kept local (local_hlc >= remote_hlc), this is a
        pure status update.  If LWW overwrote with remote, this reverts
        the row to the local snapshot.
        """
        record = self._load_open_conflict(conflict_id)
        if not record:
            return {"ok": False, "reason": "conflict not found or already resolved"}

        # Determine whether we need to revert
        # LWW winner = max(local_hlc, remote_hlc)
        lww_chose_remote = record["remote_hlc"] >= record["local_hlc"]

        if lww_chose_remote:
            # LWW applied remote, but user wants local → revert
            local_snap = json.loads(record["local_snapshot"])
            self._force_write_snapshot(
                record["table_name"], record["row_id"], local_snap
            )

        self._mark_resolved(conflict_id, "keep_local")
        return {"ok": True, "strategy": "keep_local", "data_reverted": lww_chose_remote}

    def resolve_accept_remote(self, conflict_id: str) -> Dict[str, Any]:
        """
        Accept the remote version.

        If LWW already applied remote (remote_hlc >= local_hlc), this
        is a pure status update.  If LWW kept local, this force-applies
        the remote snapshot.
        """
        record = self._load_open_conflict(conflict_id)
        if not record:
            return {"ok": False, "reason": "conflict not found or already resolved"}

        lww_chose_local = record["local_hlc"] > record["remote_hlc"]

        if lww_chose_local:
            # LWW kept local, but user wants remote → force-apply
            remote_snap = json.loads(record["remote_snapshot"])
            self._force_write_snapshot(
                record["table_name"], record["row_id"], remote_snap
            )

        self._mark_resolved(conflict_id, "accept_remote")
        return {"ok": True, "strategy": "accept_remote", "data_reverted": lww_chose_local}

    def resolve_ignore(self, conflict_id: str) -> Dict[str, Any]:
        """
        Mark as ignored — no data change, just close the conflict.
        """
        record = self._load_open_conflict(conflict_id)
        if not record:
            return {"ok": False, "reason": "conflict not found or already resolved"}

        self._mark_resolved(conflict_id, "ignored")
        return {"ok": True, "strategy": "ignored", "data_reverted": False}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_open_conflict(self, conflict_id: str) -> Optional[Dict[str, Any]]:
        """Fetch conflict only if it is still open."""
        record = self.repo.get_conflict_by_id(conflict_id)
        if not record or record["status"] != "open":
            return None
        return record

    def _mark_resolved(self, conflict_id: str, strategy: str) -> None:
        """Update status, resolution strategy, and resolved_ts atomically."""
        resolved_ts = datetime.now(timezone.utc).isoformat()
        try:
            with self.db.connection() as conn:
                conn.execute(
                    """
                    UPDATE conflict_record
                    SET status = 'resolved',
                        resolution = ?,
                        resolved_ts = ?
                    WHERE conflict_id = ? AND status = 'open'
                    """,
                    (strategy, resolved_ts, conflict_id),
                )
        except sqlite3.Error as e:
            raise DatabaseError(f"_mark_resolved failed: {e}") from e

    def _force_write_snapshot(
        self, table_name: str, row_id: str, snapshot: dict
    ) -> None:
        """
        Overwrite the current row with a snapshot payload.

        This bypasses LWW — it is a direct UPDATE used only during
        explicit conflict resolution.  The version_hlc in the snapshot
        is preserved so downstream sync will see the correct version.
        Also updates row_hash to keep Merkle tree consistent.
        """
        if not snapshot:
            return

        # Build UPDATE SET clause from snapshot keys (exclude 'id')
        update_cols = [k for k in snapshot.keys() if k != "id"]
        if not update_cols:
            return

        set_clause = ", ".join(f"{c} = ?" for c in update_cols)
        values = [snapshot[c] for c in update_cols]
        values.append(row_id)

        try:
            with self.db.connection() as conn:
                conn.execute(
                    f"UPDATE {table_name} SET {set_clause} WHERE id = ?",
                    tuple(values),
                )

                # Keep row_hash in sync with the new state
                import hashlib
                dict_copy = dict(snapshot)
                for k, v in dict_copy.items():
                    if isinstance(v, (bytes, bytearray)):
                        dict_copy[k] = v.hex()
                json_str = json.dumps(dict_copy, sort_keys=True, default=str)
                hash_val = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
                hlc = snapshot.get("version_hlc", 0)

                conn.execute(
                    "INSERT OR REPLACE INTO row_hash (table_name, row_id, hash_val, version_hlc) VALUES (?, ?, ?, ?)",
                    (table_name, row_id, hash_val, hlc),
                )
        except sqlite3.Error as e:
            raise DatabaseError(
                f"_force_write_snapshot failed for {table_name}:{row_id}: {e}"
            ) from e
