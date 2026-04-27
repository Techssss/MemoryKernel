import time
import os
from types import SimpleNamespace

from memk.storage.db import MemoryDB
from memk.sync.conflict import ConflictRepository
from memk.sync.merkle import MerkleService
from memk.sync.protocol import SyncProtocolNode
from memk.sync.hybrid import HybridSyncService
from bench.dataset import SyntheticDataset
from bench.metrics import MetricsCollector

class SyncStress:
    def __init__(self, work_dir: str, collector: MetricsCollector):
        self.work_dir = work_dir
        self.collector = collector
        self.dataset = SyntheticDataset()
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

    @staticmethod
    def _make_node(db: MemoryDB, workspace_id: str) -> SyncProtocolNode:
        runtime = SimpleNamespace(db=db, workspace_id=workspace_id)
        return SyncProtocolNode(MerkleService(runtime))

    @staticmethod
    def _refresh(node: SyncProtocolNode) -> None:
        node.merkle.rebuild_buckets(node.db.get_latest_version_hlc())

    def run_hybrid_recovery_scenario(self, item_count: int = 1000):
        """
        Scenario B: Offline replica + Oplog GC -> Merkle Recovery.
        """
        snap = self.collector.start_test("Sync_Hybrid_Recovery")
        
        path_a = os.path.join(self.work_dir, "replica_a.db")
        path_b = os.path.join(self.work_dir, "replica_b.db")
        
        db_a = MemoryDB(path_a)
        db_b = MemoryDB(path_b)
        db_a.init_db()
        db_b.init_db()
        
        # 1. Initial state
        mems = self.dataset.generate_memories(item_count)
        for m in mems[:item_count//2]:
            db_a.insert_memory(m["content"])
            
        # 2. Sync B
        node_a = self._make_node(db_a, "node_a")
        node_b = self._make_node(db_b, "node_b")
        hybrid_b = HybridSyncService(node_b)
        
        self._refresh(node_a)
        hybrid_b.sync_from_source(node_a, "node_a")
        self._refresh(node_b)
        
        # 3. B goes offline, A updates more and PRUNES oplog
        for m in mems[item_count//2:]:
            db_a.insert_memory(m["content"])
            
        # Simulate aggressive GC
        db_a.prune_oplog_entries(boundary_hlc=9999999999999) # Clear everything
        self._refresh(node_a)
        
        # 4. B comes back -> Should trigger Merkle
        start_time = time.perf_counter()
        result = hybrid_b.sync_from_source(node_a, "node_a")
        snap.latency_ms.append((time.perf_counter() - start_time) * 1000)
        
        snap.extras["mode_chosen"] = result.get("mode")
        
        # Convergence check
        count_a = db_a.get_stats()["total_memories"]
        count_b = db_b.get_stats()["total_memories"]
        snap.extras["converged"] = (count_a == count_b)
        
        self.collector.record_batch(snap, start_time, 1)
        return snap.summarize()

    def run_conflict_scenario(self):
        """
        Scenario C: Concurrent updates -> Conflict Record detection.
        """
        snap = self.collector.start_test("Sync_Conflict_Detection")
        
        path_a = os.path.join(self.work_dir, "conf_a.db")
        path_b = os.path.join(self.work_dir, "conf_b.db")
        db_a = MemoryDB(path_a)
        db_b = MemoryDB(path_b)
        db_a.init_db()
        db_b.init_db()
        
        db_a.insert_memory("Shared Init")
        
        # Sync initial
        node_a = self._make_node(db_a, "node_a")
        node_b = self._make_node(db_b, "node_b")
        node_b.apply_remote_delta(db_a.get_delta_since(0), remote_replica_id="node_a")
        baseline_hlc = db_a.get_latest_version_hlc()
        
        first_a = db_a.get_all_memories()[0]
        id_val = first_a["id"]
        
        # Concurrent divergent updates
        db_a.update_memory_content(id_val, "Update from A")
        db_b.update_memory_content(id_val, "Update from B")
        
        start_time = time.perf_counter()
        # Apply A's change to B with conflict detection ON
        delta_a = db_a.get_delta_since(baseline_hlc)
        node_b.apply_remote_delta(delta_a, detect_conflicts=True)
        snap.latency_ms.append((time.perf_counter() - start_time) * 1000)
        
        # Check if conflict was recorded
        conflicts = ConflictRepository(db_b).list_open_conflicts()
        snap.extras["conflicts_detected"] = len(conflicts)
        
        self.collector.record_batch(snap, start_time, 1)
        return snap.summarize()
