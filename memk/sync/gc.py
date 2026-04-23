import time
from typing import Optional
from memk.storage.db import MemoryDB

class OplogGC:
    """
    Garbage Collector logic for Delta Sync Oplog operations.
    Determines mathematically safe operational boundaries where historic state 
    mutations can be pruned without breaking remote replications.
    """
    
    @staticmethod
    def get_safe_prune_boundary(
        db: MemoryDB, 
        retention_seconds: int = 86400 * 7,  # Default 7 days
        override_now_ms: Optional[int] = None
    ) -> int:
        """
        Determines the safe HLC threshold beneath which oplog entries can be cleanly deleted.
        
        Rules:
        1. Never prune newer than the `retention_seconds`.
        2. Never prune past the slowest known Remote Replica in the sync topology.
        3. If no replicas are known (single-node), degrade to pure time-based retention bounds.
        """
        now_ms = override_now_ms if override_now_ms is not None else int(time.time() * 1000)
        retention_boundary = now_ms - (retention_seconds * 1000)
        
        # Guard minimum bounds against extreme future anomalies
        retention_boundary = max(0, retention_boundary)
        
        min_replica_hlc = db.get_min_acknowledged_hlc()
        
        if min_replica_hlc is None:
            # Topology lacks replicas (Isolated DB). Prune strictly by retention TTL to avoid bloat.
            return retention_boundary
            
        # Topology exists. Must be both trailing retention AND acknowledged by the slowest node.
        # Boundary is whatever is STRICTER (lower HLC).
        return min(retention_boundary, min_replica_hlc)

    @staticmethod
    def run_oplog_gc_job(
        db: MemoryDB,
        retention_seconds: int = 86400 * 7,
        batch_size: int = 1000,
        dry_run: bool = False,
        override_now_ms: Optional[int] = None
    ) -> dict:
        """
        Execution handler for garbage collecting Oplogs. Loops pruning ops until cleanup is complete.
        Supports dry_run to preview size without deleting.
        Returns operational statistics logs.
        """
        boundary = OplogGC.get_safe_prune_boundary(db, retention_seconds, override_now_ms)
        
        stats = {
            "cutoff_hlc": boundary,
            "batches": 0,
            "deleted_count": 0,
            "dry_run": dry_run
        }
        
        while True:
            deleted_in_batch = db.prune_oplog_entries(
                boundary_hlc=boundary,
                batch_size=batch_size,
                dry_run=dry_run
            )
            stats["deleted_count"] += deleted_in_batch
            stats["batches"] += 1
            
            # If dry run, the "deletion" didn't actually execute, so a loop will infinite-fetch the same dry count.
            # Thus, we break immediately on dry run. We also break if we underflow the batch limits indicating end of chunk.
            if dry_run or deleted_in_batch < batch_size or deleted_in_batch == 0:
                break
                
        # Register explicitly to the Background Jobs tracking engine
        if not dry_run:
            try:
                job_id = db.insert_background_job("oplog_gc", "completed")
                db.complete_background_job(job_id, {"stats": stats})
            except Exception:
                pass # Fail silently if background_jobs structure gets changed
            
        return stats
