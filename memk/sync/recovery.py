import logging
from typing import Dict, Any, List
from memk.sync.protocol import SyncProtocolNode

logger = logging.getLogger("memk.sync")

class MerkleRecoveryService:
    """
    Orchestrates full state-based recovery using Merkle trees.
    Used when Oplog deltas are insufficient (stale replicas).
    """
    def __init__(self, local_node: SyncProtocolNode):
        self.node = local_node
        self.db = local_node.db

    def recover_from_remote(self, remote_node: SyncProtocolNode, remote_replica_id: str) -> Dict[str, Any]:
        """
        Execute the recovery workflow:
        1. Compare Root Hashes
        2. If different, find mismatched bucket IDs
        3. Fetch mismatching row payloads from remote
        4. Apply to local using LWW logic
        """
        stats = {
            "root_matched": False,
            "buckets_diffed": 0,
            "rows_recovered": 0,
            "status": "started"
        }

        # 1. Root Hash Comparison
        local_root = self.node.get_root_hash()
        remote_root = remote_node.get_root_hash()

        if local_root == remote_root:
            stats["root_matched"] = True
            stats["status"] = "consistent"
            return stats

        # 2. Bucket Diffing
        remote_buckets = remote_node.get_bucket_hashes()
        mismatched_ids = self.node.diff_buckets(remote_buckets)
        stats["buckets_diffed"] = len(mismatched_ids)

        if not mismatched_ids:
            # Hash collision at root or logic error - should not happen if salts match
            stats["status"] = "hash_conflict_detected"
            return stats

        # 3. Fetch Remote Deltas (The missing data)
        remote_deltas = remote_node.fetch_delta_for_buckets(mismatched_ids)
        stats["rows_recovered"] = len(remote_deltas)

        # 4. Apply locally
        # We don't pass a specific 'remote_cursor' here because this is state-based, 
        # not log-based. But we do pass remote_replica_id to update checkpoint.
        self.node.apply_remote_delta(
            remote_deltas=remote_deltas,
            remote_replica_id=remote_replica_id
        )

        stats["status"] = "completed"
        return stats
