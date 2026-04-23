import hashlib
import logging
from memk.core.runtime import WorkspaceRuntime

logger = logging.getLogger("memk.sync")

class MerkleService:
    """
    Maintains a deterministic Merkle tree (bucketed) of database rows.
    Allows for efficient comparisons of state between two devices to isolate deltas.
    """
    def __init__(self, runtime: WorkspaceRuntime, num_buckets: int = 256):
        self.runtime = runtime
        self.db = runtime.db
        self.num_buckets = num_buckets

    def rebuild_buckets(self, current_hlc: int) -> dict:
        """
        Recompute bucket hashes from the current row_hash table.
        Each bucket hash is the deterministic combination of all row_hashes assigned to it.
        Bucket assignment = hash % num_buckets.
        Returns the new state of the buckets.
        """
        try:
            with self.db._get_connection() as conn:
                rows = conn.execute("SELECT hash_val FROM row_hash").fetchall()
                
                buckets = {i: [] for i in range(self.num_buckets)}
                for r in rows:
                    h_str = r["hash_val"]
                    # Calculate bucket assignment purely on raw hash prefix
                    bucket_idx = int(h_str[:8], 16) % self.num_buckets
                    buckets[bucket_idx].append(h_str)
                    
                state = {}
                # Update merkle_bucket table
                for b_idx, hashes in buckets.items():
                    if not hashes:
                        b_hash = "empty"
                    else:
                        # Combine deterministically: sort and hash
                        hashes.sort()
                        combined = "".join(hashes)
                        b_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
                        
                    conn.execute(
                        "INSERT OR REPLACE INTO merkle_bucket (bucket_id, hash_val, updated_hlc) VALUES (?, ?, ?)",
                        (b_idx, b_hash, current_hlc)
                    )
                    state[b_idx] = b_hash
                    
                logger.info(f"[{self.runtime.workspace_id}] Merkle buckets rebuilt. HLC: {current_hlc}")
                return state
        except Exception as e:
            logger.error(f"Merkle rebuilt failed: {e}")
            return {}

    def cleanup_stale_row_hashes(self, verify_content_hash: bool = False, dry_run: bool = False) -> dict:
        """
        Sweep row_hash table to prune orphaned entries when a physical row semantic is deleted abruptly.
        Optionally rebuild stale hashes where the semantic row might lack oplog interceptors.
        """
        stats = {"orphans_deleted": 0, "hashes_corrected": 0, "dry_run": dry_run}
        
        with self.db._get_connection() as conn:
            tables = [r["table_name"] for r in conn.execute("SELECT DISTINCT table_name FROM row_hash").fetchall()]
            
            for t in tables:
                # 1. Orphan cleanups (Physical rows vanish abruptly)
                orphans = conn.execute(
                    f"SELECT row_id FROM row_hash WHERE table_name = ? AND row_id NOT IN (SELECT id FROM {t})", 
                    (t,)
                ).fetchall()
                
                if orphans:
                    stats["orphans_deleted"] += len(orphans)
                    if not dry_run:
                        conn.execute(f"DELETE FROM row_hash WHERE table_name = ? AND row_id NOT IN (SELECT id FROM {t})", (t,))
                
                # 2. Structural Content drift repairs
                if verify_content_hash:
                    import json
                    all_refs = conn.execute("SELECT row_id, hash_val FROM row_hash WHERE table_name = ?", (t,)).fetchall()
                    for ref in all_refs:
                        row_id = ref["row_id"]
                        old_hash = ref["hash_val"]
                        
                        physical = conn.execute(f"SELECT * FROM {t} WHERE id = ?", (row_id,)).fetchone()
                        if not physical:
                            continue
                            
                        # Protocol Serialization mapping
                        p_dict = dict(physical)
                        for k, v in p_dict.items():
                            if isinstance(v, bytes):
                                p_dict[k] = v.hex()
                        
                        new_hash = hashlib.sha256(json.dumps(p_dict, sort_keys=True).encode("utf-8")).hexdigest()
                        
                        if new_hash != old_hash:
                            stats["hashes_corrected"] += 1
                            if not dry_run:
                                conn.execute("UPDATE row_hash SET hash_val = ? WHERE table_name = ? AND row_id = ?", (new_hash, t, row_id))
                                
        return stats

    def rebuild_or_refresh_merkle_buckets(self, current_hlc: int, dry_run: bool = False) -> dict:
        """
        Intelligent rebuilding logic ensuring empty scopes are terminated and unrequired
        bucket recalculations are skipped minimizing DB locks.
        """
        stats = {"buckets_deleted": 0, "buckets_refreshed": 0, "dry_run": dry_run}
        try:
            with self.db._get_connection() as conn:
                rows = conn.execute("SELECT hash_val FROM row_hash").fetchall()
                
                # Aggregate state mappings
                buckets = {i: [] for i in range(self.num_buckets)}
                for r in rows:
                    h_str = r["hash_val"]
                    bucket_idx = int(h_str[:8], 16) % self.num_buckets
                    buckets[bucket_idx].append(h_str)
                    
                # Evaluate differences to prune unused blocks
                for b_idx, hashes in buckets.items():
                    current_bucket_row = conn.execute("SELECT hash_val FROM merkle_bucket WHERE bucket_id = ?", (b_idx,)).fetchone()
                    current_b_hash = current_bucket_row["hash_val"] if current_bucket_row else None
                    
                    if not hashes:
                        # Empty bucket namespace
                        if current_b_hash is not None:
                            stats["buckets_deleted"] += 1
                            if not dry_run:
                                conn.execute("DELETE FROM merkle_bucket WHERE bucket_id = ?", (b_idx,))
                    else:
                        hashes.sort()
                        combined = "".join(hashes)
                        new_b_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
                        
                        if current_b_hash != new_b_hash:
                            stats["buckets_refreshed"] += 1
                            if not dry_run:
                                conn.execute(
                                    "INSERT OR REPLACE INTO merkle_bucket (bucket_id, hash_val, updated_hlc) VALUES (?, ?, ?)",
                                    (b_idx, new_b_hash, current_hlc)
                                )
                                
                return stats
        except Exception as e:
            logger.error(f"Intelligent bucket rebuild failed: {e}")
            return stats
