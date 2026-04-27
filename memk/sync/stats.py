import time
import json
import logging
from typing import Dict, Any
from memk.core.runtime import WorkspaceRuntime
from memk.sync.gc import OplogGC
from memk.sync.merkle import MerkleService

logger = logging.getLogger("memk.sync.stats")

class SyncStatsService:
    """
    Observability service for Delta Sync and Merkle Tree hardening.
    Provides real-time health metrics for sync topology and data integrity.
    """
    def __init__(self, runtime: WorkspaceRuntime):
        self.runtime = runtime
        self.db = runtime.db
        # We use a localized MerkleService instance for analysis
        self.merkle = MerkleService(runtime)

    def get_sync_hardening_stats(self) -> Dict[str, Any]:
        """
        Gathers a structured snapshot of synchronization health.
        
        Returns:
            Dict containing oplog counts, replica lags, stale metadata counts, and GC history.
        """
        now_ms = int(time.time() * 1000)
        
        try:
            with self.db.connection() as conn:
                def scalar(q, params=()):
                    res = conn.execute(q, params).fetchone()
                    return res[0] if res and res[0] is not None else 0

                # 1. Oplog Metrics
                oplog_count = scalar("SELECT COUNT(*) FROM oplog")
                min_hlc = scalar("SELECT MIN(version_hlc) FROM oplog")
                oldest_oplog_age_ms = (now_ms - min_hlc) if oplog_count > 0 else 0
                
                # Use GC logic to determine what's prunable
                boundary_hlc = OplogGC.get_safe_prune_boundary(self.db)
                prunable_oplog_count = scalar("SELECT COUNT(*) FROM oplog WHERE version_hlc < ?", (boundary_hlc,))

                # 2. Replica Metrics
                replica_checkpoint_count = scalar("SELECT COUNT(*) FROM replica_checkpoint")
                min_replica_hlc = scalar("SELECT MIN(last_applied_hlc) FROM replica_checkpoint")
                slowest_replica_lag_ms = (now_ms - min_replica_hlc) if replica_checkpoint_count > 0 else 0

                # 3. Integrity/Stale Metrics (Dry run results)
                # orphans_deleted finds row_hash entries without physical rows
                stale_hashes = self.merkle.cleanup_stale_row_hashes(verify_content_hash=False, dry_run=True)
                stale_row_hash_count = stale_hashes.get("orphans_deleted", 0)
                
                # buckets_refreshed finds buckets whose hash doesn't match current row_hash state
                stale_buckets = self.merkle.rebuild_or_refresh_merkle_buckets(now_ms, dry_run=True)
                stale_merkle_bucket_count = stale_buckets.get("buckets_refreshed", 0)

                # 4. GC metrics from background jobs log
                last_gc_job = conn.execute(
                    "SELECT created_at, result FROM background_jobs WHERE job_type = 'oplog_gc' ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
                
                last_gc_run = last_gc_job["created_at"] if last_gc_job else "never"
                last_gc_deleted_count = 0
                if last_gc_job and last_gc_job["result"]:
                    try:
                        meta = json.loads(last_gc_job["result"])
                        last_gc_deleted_count = meta.get("stats", {}).get("deleted_count", 0)
                    except:
                        pass

            return {
                "oplog": {
                    "count": oplog_count,
                    "oldest_age_seconds": round(max(0, oldest_oplog_age_ms) / 1000, 1),
                    "prunable_count": prunable_oplog_count,
                },
                "replicas": {
                    "checkpoint_count": replica_checkpoint_count,
                    "slowest_lag_seconds": round(max(0, slowest_replica_lag_ms) / 1000, 1),
                },
                "integrity": {
                    "stale_row_hash_count": stale_row_hash_count,
                    "stale_merkle_bucket_count": stale_merkle_bucket_count,
                },
                "gc": {
                    "last_run": last_gc_run,
                    "last_deleted_count": last_gc_deleted_count,
                },
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Failed to gather sync stats: {e}")
            return {"error": str(e), "timestamp": time.time()}
