import hashlib
import logging
import sqlite3
from typing import Dict, List, Any

from memk.sync.merkle import MerkleService

logger = logging.getLogger("memk.sync")

class SyncProtocolNode:
    """
    Implements a local algorithmic structure for Merkle-tree based Delta Synchronization.
    Capable of querying and calculating exact mismatched buckets, and extracting raw payload deltas.
    """
    def __init__(self, merkle: MerkleService):
        self.merkle = merkle
        self.db = merkle.db
        
    def get_root_hash(self) -> str:
        """Combine all bucket hashes to get a top-level root tree hash."""
        buckets = self.get_bucket_hashes()
        sorted_b = sorted(buckets.items())
        combined = "".join(f"{b_id}:{h}" for b_id, h in sorted_b)
        if not combined:
            return "empty"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
        
    def get_bucket_hashes(self, scope: str = None) -> Dict[int, str]:
        """Fetch current known bucket hashes."""
        with self.db.connection() as conn:
            rows = conn.execute("SELECT bucket_id, hash_val FROM merkle_bucket").fetchall()
            return {r["bucket_id"]: r["hash_val"] for r in rows}
            
    def diff_buckets(self, remote_buckets: Dict[int, str]) -> List[int]:
        """Compare local and remote buckets. Return mismatched bucket IDs."""
        local = self.get_bucket_hashes()
        mismatched = []
        
        all_ids = set(local.keys()) | set(remote_buckets.keys())
        for bid in all_ids:
            lh = local.get(bid, "empty")
            rh = remote_buckets.get(bid, "empty")
            if lh != rh:
                mismatched.append(bid)
        return mismatched
        
    def fetch_delta_for_buckets(self, bucket_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Extract out all row payloads mapped to the mismatched bucket IDs.
        Optimized to avoid full row_hash scan in Python.
        """
        deltas = []
        if not bucket_ids:
            return deltas
            
        with self.db.connection() as conn:
            # Fetch only IDs that fall into the mismatched buckets
            indices_str = ",".join(map(str, bucket_ids))
            
            # Note: We rely on the deterministic prefix-based bucket assignment
            # In SQLite, we can approximate the bucket ID: (abs(instr('0123456789abcdef', substr(hash_val, 1, 1)) - 1) * ...)
            # Simplest for now: Fetch required columns only
            rows = conn.execute(
                f"SELECT table_name, row_id, hash_val FROM row_hash"
            ).fetchall()
            
            # Still some Python filtering but on smaller object set than BEFORE
            to_fetch = []
            for r in rows:
                h_val = r["hash_val"]
                b_idx = int(h_val[:8], 16) % self.merkle.num_buckets
                if b_idx in bucket_ids:
                    to_fetch.append((r["table_name"], r["row_id"]))
            
            # Batch fetch rows from semantic tables
            for tbl, row_id in to_fetch:
                try:
                    row = conn.execute(f"SELECT * FROM {tbl} WHERE id = ?", (row_id,)).fetchone()
                    if row:
                        deltas.append({
                            "table": tbl,
                            "row_id": row_id,
                            "payload": dict(row)
                        })
                except Exception as e:
                    logger.error(f"Delta fetch failed for row {row_id} in {tbl}: {e}")
                        
        return deltas

    def apply_remote_delta(
        self, 
        remote_deltas: List[Dict[str, Any]], 
        remote_replica_id: str = None, 
        remote_cursor: tuple = None,
        detect_conflicts: bool = False
    ) -> None:
        """
        Apply incoming missing payloads from another replica using LWW (Last Writer Wins).
        Only replaces local data if the incoming version_hlc is >= local version_hlc.

        When detect_conflicts=True, each row that will be overwritten is checked
        for semantic conflicts (divergent edits, cross-state mutations, text
        divergence).  Conflicts are *recorded* but LWW still decides the final
        state — no merge logic, no blocking.
        """
        if not remote_deltas and not remote_replica_id:
            return

        # Lazy-init conflict infra only when needed
        conflict_repo = None
        if detect_conflicts:
            from memk.sync.conflict import ConflictDetector, ConflictRepository
            conflict_repo = ConflictRepository(self.db)

        with self.db.connection() as conn:
            for d in remote_deltas:
                tbl = d["table"]
                data = d["payload"]
                row_id = d["row_id"]
                
                cols = list(data.keys())
                vals = [data[k] for k in cols]
                placeholders = ",".join(["?"] * len(cols))
                
                # Build set clause for UPSERT: col1=excluded.col1, ...
                # Excluding 'id' from update set
                set_parts = [f"{c}=excluded.{c}" for c in cols if c != "id"]
                set_clause = ", ".join(set_parts)
                
                # Check if table supports versioning (memories, kg_fact)
                has_versioning = "version_hlc" in cols
                
                # --- Conflict detection (before overwrite) ---
                if detect_conflicts and has_versioning:
                    try:
                        local_row = conn.execute(
                            f"SELECT * FROM {tbl} WHERE id = ?", (row_id,)
                        ).fetchone()
                        
                        if local_row:
                            local_dict = dict(local_row)
                            reason = ConflictDetector.detect(tbl, local_dict, data)
                            if reason:
                                local_hlc = local_dict.get("version_hlc", 0)
                                remote_hlc = data.get("version_hlc", 0)
                                try:
                                    conflict_repo.create_conflict_record(
                                        table_name=tbl,
                                        row_id=row_id,
                                        local_hlc=local_hlc,
                                        remote_hlc=remote_hlc,
                                        local_payload=local_dict,
                                        remote_payload=data,
                                    )
                                    logger.info(
                                        f"Conflict recorded for {tbl}:{row_id} "
                                        f"(reason={reason}, local_hlc={local_hlc}, remote_hlc={remote_hlc})"
                                    )
                                except Exception as e:
                                    logger.warning(f"Conflict recording failed for {tbl}:{row_id}: {e}")
                    except Exception as e:
                        logger.warning(f"Conflict detection skipped for {tbl}:{row_id}: {e}")

                # --- LWW upsert (unchanged semantics) ---
                if has_versioning:
                    sql = f"""
                    INSERT INTO {tbl} ({','.join(cols)}) VALUES ({placeholders})
                    ON CONFLICT(id) DO UPDATE SET
                        {set_clause}
                    WHERE excluded.version_hlc >= {tbl}.version_hlc
                    """
                else:
                    # Fallback to blind replace for non-versioned tables (mostly decisions/etc)
                    sql = f"INSERT OR REPLACE INTO {tbl} ({','.join(cols)}) VALUES ({placeholders})"
                
                try:
                    conn.execute(sql, tuple(vals))
                except sqlite3.Error as e:
                    logger.error(f"Failed to apply delta for {tbl}:{row_id}: {e}")
                    continue
                
                # Update row_hash based on ACTUAL state after upsert.
                # Critical: if LWW rejected the write (local was newer),
                # we must hash the local state, not the rejected remote payload.
                import json
                actual_row = conn.execute(f"SELECT * FROM {tbl} WHERE id = ?", (row_id,)).fetchone()
                if actual_row:
                    dict_copy = dict(actual_row)
                    for k, v in dict_copy.items():
                        if isinstance(v, bytes):
                            dict_copy[k] = v.hex()
                            
                    dict_str = json.dumps(dict_copy, sort_keys=True)
                    hash_val = hashlib.sha256(dict_str.encode("utf-8")).hexdigest()
                    actual_hlc = dict_copy.get("version_hlc", data.get("version_hlc", 0))
                    
                    conn.execute(
                        "INSERT OR REPLACE INTO row_hash (table_name, row_id, hash_val, version_hlc) VALUES (?, ?, ?, ?)",
                        (tbl, row_id, hash_val, actual_hlc)
                    )

                
            # Update the Replica Checkpoint atomically on success
            if remote_replica_id:
                c_hlc, c_node, c_seq = 0, "unknown", 0
                if remote_cursor:
                    c_hlc, c_node, c_seq = remote_cursor
                else:
                    for d in remote_deltas:
                        d_hlc = d["payload"].get("version_hlc", 0)
                        if d_hlc > c_hlc:
                            c_hlc = d_hlc
                            c_node = d["payload"].get("version_node", "unknown")
                            c_seq = d["payload"].get("version_seq", 0)
                
                if c_hlc > 0:
                    from datetime import datetime, timezone
                    now_ts = datetime.now(timezone.utc).isoformat()
                    # Use monotonic update helper logic directly
                    conn.execute(
                        """
                        INSERT INTO replica_checkpoint (replica_id, last_applied_hlc, last_applied_node, last_applied_seq, updated_ts, note)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(replica_id) DO UPDATE SET
                            last_applied_hlc=excluded.last_applied_hlc,
                            last_applied_node=excluded.last_applied_node,
                            last_applied_seq=excluded.last_applied_seq,
                            updated_ts=excluded.updated_ts,
                            note=excluded.note
                        WHERE excluded.last_applied_hlc >= replica_checkpoint.last_applied_hlc
                        """,
                        (remote_replica_id, c_hlc, c_node, c_seq, now_ts, "Delta Batch Applied")
                    )
