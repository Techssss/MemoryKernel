import logging
from typing import Dict, Any

from memk.sync.health import SyncMode
from memk.sync.protocol import SyncProtocolNode
from memk.sync.recovery import MerkleRecoveryService

logger = logging.getLogger("memk.sync")

class HybridSyncService:
    """
    High-level orchestrator that chooses between Oplog and Merkle sync.
    """
    def __init__(self, local_node: SyncProtocolNode):
        self.node = local_node
        self.recovery = MerkleRecoveryService(local_node)

    def sync_from_source(self, source_node: SyncProtocolNode, source_replica_id: str) -> Dict[str, Any]:
        """
        Main entry point for local node to catch up with a source node.
        """
        # Determine our own checkpoint for the source
        local_checkpoint = self.node.db.get_replica_checkpoint(source_replica_id)
        local_hlc = local_checkpoint["last_applied_hlc"] if local_checkpoint else 0
        
        # 1. Decision phase
        decision = self._decide_mode(source_node, local_hlc)
        mode = decision["mode"]
        
        result = {
            "mode": mode,
            "details": decision,
            "stats": {}
        }
        
        if mode == SyncMode.NO_OP.value:
            logger.info(f"Local node is already up to date with {source_replica_id}.")
            return result
            
        elif mode == SyncMode.OPLOG_DELTA.value:
            logger.info(f"Catching up with {source_replica_id} using Oplog Fast-Path.")
            # Fetch from SOURCE's oplog
            deltas = source_node.db.get_delta_since(local_hlc)
            # Apply to LOCAL
            self.node.apply_remote_delta(deltas, remote_replica_id=source_replica_id)
            
            result["stats"] = {"applied_from_oplog": len(deltas)}
            return result
            
        elif mode == SyncMode.MERKLE_RECOVERY.value:
            logger.info(f"Local state for {source_replica_id} is stale or unknown. Using Merkle Recovery.")
            # Recover LOCAL from SOURCE
            recovery_stats = self.recovery.recover_from_remote(source_node, source_replica_id)
            result["stats"] = recovery_stats
            return result
            
        return result

    def _decide_mode(self, source_node: SyncProtocolNode, local_hlc: int) -> Dict[str, Any]:
        source_db = source_node.db
        latest_hlc = source_db.get_latest_version_hlc()
        op_range = source_db.get_oplog_range()
        min_op_hlc = op_range.get("min")
        
        res = {
            "mode": SyncMode.NO_OP.value,
            "replica_hlc": local_hlc,
            "source_latest_hlc": latest_hlc,
            "source_oplog_min": min_op_hlc
        }
        
        if local_hlc >= latest_hlc:
            res["mode"] = SyncMode.NO_OP.value
        elif min_op_hlc is None or local_hlc < min_op_hlc:
            res["mode"] = SyncMode.MERKLE_RECOVERY.value
        else:
            res["mode"] = SyncMode.OPLOG_DELTA.value
            
        return res
