import time
import numpy as np
from memk.storage.db import MemoryDB
from memk.core.graph_propagation import propagate_ppnp
from bench.metrics import MetricsCollector

class GraphStress:
    def __init__(self, db_path: str, collector: MetricsCollector):
        self.db = MemoryDB(db_path)
        self.collector = collector

    def run_propagation_stress(self, num_entities: int = 1000, density: float = 0.05):
        """
        Simulates node explosion and score drift in a dense entity graph.
        """
        snap = self.collector.start_test(f"Graph_Propagation_{num_entities}")
        
        # Build synthetic adjacency matrix in CSR-like format for PPNP
        # num_entities^2 * density edges
        num_edges = int(num_entities * num_entities * density)
        indices = np.random.randint(0, num_entities, num_edges)
        indptr = np.zeros(num_entities + 1, dtype=int)
        for i in range(num_entities):
            indptr[i+1] = indptr[i] + (num_edges // num_entities)
        
        weights = np.random.rand(num_edges).astype(np.float32)
        
        # Initial seeds
        seed_scores = {i: 1.0 for i in range(10)}
        
        start_time = time.perf_counter()
        try:
            # We call the core propagation function directly to stress it
            activated = propagate_ppnp(
                seed_scores=seed_scores,
                indptr=indptr,
                indices=indices,
                weights=weights,
                num_entities=num_entities,
                alpha=0.15,
                steps=5,
                max_active_entities=500
            )
            snap.latency_ms.append((time.perf_counter() - start_time) * 1000)
            snap.extras["activated_count"] = len(activated)
        except Exception as e:
            snap.errors += 1
            print(f"Graph stress failure: {e}")
            
        self.collector.record_batch(snap, start_time, 1)
        return snap.summarize()
