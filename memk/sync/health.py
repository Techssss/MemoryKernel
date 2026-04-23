import logging
from enum import Enum
from typing import Dict, Any, Optional
from memk.storage.db import MemoryDB

logger = logging.getLogger("memk.sync")

class SyncState(Enum):
    FRESH = "fresh"      # Replica is fully up to date or extremely close
    LAGGING = "lagging"  # Behind but all missing changes still exist in Oplog
    STALE = "stale"      # Behind and Oplog has been pruned; must use full Merkle recovery
    UNKNOWN = "unknown"  # No checkpoint found for this replica

class SyncMode(Enum):
    OPLOG_DELTA = "oplog_delta"
    MERKLE_RECOVERY = "merkle_recovery"
    NO_OP = "no_op"

class ReplicaHealthService:
    """
    Evaluates the synchronization health of remote replicas.
    Determines if a replica can use the Fast Path (Oplog) or must use the Slow Path (Merkle).
    """
    def __init__(self, db: MemoryDB):
        self.db = db

    def get_replica_sync_state(self, replica_id: str) -> SyncState:
        """
        Analyze replica status based on local Oplog availability and current DB version.
        """
        checkpoint = self.db.get_replica_checkpoint(replica_id)
        if not checkpoint:
            return SyncState.UNKNOWN
            
        replica_hlc = checkpoint.get("last_applied_hlc", 0)
        latest_hlc = self.db.get_latest_version_hlc()
        
        # 1. Check if fully synced
        if replica_hlc >= latest_hlc:
            return SyncState.FRESH
            
        # 2. Check Oplog availability
        op_range = self.db.get_oplog_range()
        min_op_hlc = op_range.get("min")
        
        # If oplog is empty, and we aren't FRESH, we are technically STALE relative to log-based sync
        if min_op_hlc is None:
            return SyncState.STALE
            
        # If the replica needs data older than what we have in Oplog
        if replica_hlc < min_op_hlc:
            return SyncState.STALE
            
        return SyncState.LAGGING

    def choose_sync_mode(self, replica_id: str) -> Dict[str, Any]:
        """
        Strategy selector for synchronization.
        Decides the most efficient path based on the health report.
        """
        state = self.get_replica_sync_state(replica_id)
        checkpoint = self.db.get_replica_checkpoint(replica_id)
        op_range = self.db.get_oplog_range()
        latest_hlc = self.db.get_latest_version_hlc()
        replica_hlc = checkpoint.get("last_applied_hlc", 0) if checkpoint else 0

        res = {
            "replica_id": replica_id,
            "mode": SyncMode.NO_OP.value,
            "reason": "",
            "boundaries": {
                "replica_hlc": replica_hlc,
                "local_latest_hlc": latest_hlc,
                "oplog_min_hlc": op_range.get("min"),
                "oplog_max_hlc": op_range.get("max")
            }
        }

        if state == SyncState.FRESH:
            res["mode"] = SyncMode.NO_OP.value
            res["reason"] = "Replica is already fresh"
        elif state == SyncState.LAGGING:
            res["mode"] = SyncMode.OPLOG_DELTA.value
            res["reason"] = "Perfect match found in local Oplog"
        elif state == SyncState.STALE:
            res["mode"] = SyncMode.MERKLE_RECOVERY.value
            res["reason"] = "Replica lag exceeds Oplog retention (stale)"
        elif state == SyncState.UNKNOWN:
            res["mode"] = SyncMode.MERKLE_RECOVERY.value
            res["reason"] = "First-time sync for this replica (unknown state)"
        
        return res

    def is_replica_stale(self, replica_id: str) -> bool:
        """Helper to quickly check if a full recovery is needed."""
        state = self.get_replica_sync_state(replica_id)
        return state in (SyncState.STALE, SyncState.UNKNOWN)

    def get_health_report(self, replica_id: str) -> Dict[str, Any]:
        """Detailed diagnostic for a specific replica."""
        state = self.get_replica_sync_state(replica_id)
        checkpoint = self.db.get_replica_checkpoint(replica_id)
        latest_hlc = self.db.get_latest_version_hlc()
        op_range = self.db.get_oplog_range()
        
        replica_hlc = checkpoint.get("last_applied_hlc", 0) if checkpoint else 0
        lag = latest_hlc - replica_hlc
        
        return {
            "replica_id": replica_id,
            "state": state.value,
            "replica_hlc": replica_hlc,
            "local_latest_hlc": latest_hlc,
            "hlc_lag": lag,
            "oplog_min_hlc": op_range.get("min"),
            "oplog_max_hlc": op_range.get("max"),
            "can_use_oplog": state == SyncState.LAGGING or state == SyncState.FRESH
        }
