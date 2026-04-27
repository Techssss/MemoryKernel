import time
from memk.storage.db import MemoryDB
from bench.dataset import SyntheticDataset
from bench.metrics import MetricsCollector, MetricSnapshot

class IngestionStress:
    def __init__(self, db_path: str, collector: MetricsCollector):
        self.db = MemoryDB(db_path)
        self.collector = collector
        self.dataset = SyntheticDataset()

    def run(self, count: int = 1000):
        snap = self.collector.start_test(f"Ingestion_{count}")
        memories = self.dataset.generate_memories(count)
        
        start_time = time.perf_counter()
        batch_size = 100
        errors = 0
        
        for i in range(0, count, batch_size):
            batch = memories[i : i + batch_size]
            batch_start = time.perf_counter()
            try:
                # Assuming batch creation or iterative insertion
                # If memk has a bulk insert, we should use it. 
                # For now, iterative to measure per-item latency if needed, 
                # or batch write if supported.
                for item in batch:
                    # Using the standard insert_memory
                    self.db.insert_memory(content=item["content"], importance=0.5)
                
                batch_duration = (time.perf_counter() - batch_start) * 1000 # ms
                # Record individual item latencies (averaged over batch for efficiency)
                avg_lat = batch_duration / len(batch)
                for _ in range(len(batch)):
                    snap.latency_ms.append(avg_lat)
            except Exception:
                errors += len(batch)
        
        self.collector.record_batch(snap, start_time, count, errors)
        return snap.summarize()
