import time
import uuid
import numpy as np
from typing import List, Dict

from memk.retrieval.index import VectorIndex, IndexEntry

def normalize_vectors(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, norms, out=np.zeros_like(v), where=norms!=0)

def run_benchmark():
    DIM = 384
    N_VECTORS = 50_000
    N_CENTROIDS = 50
    N_PROBE = 5
    Q_RUNS = 100
    
    print(f"--- IVF Routing Benchmark ---")
    print(f"Vectors:   {N_VECTORS:,}")
    print(f"Centroids: {N_CENTROIDS}")
    print(f"N_Probe:   {N_PROBE}")
    
    # 1. Generate Synthetic Centroids & Data
    print("Generating synthetic data...")
    raw_centroids = np.random.randn(N_CENTROIDS, DIM).astype(np.float32)
    centroids = normalize_vectors(raw_centroids)
    
    centroids_map = {f"c_{i}": centroids[i] for i in range(N_CENTROIDS)}
    
    # Generate vectors around centroids
    entries = []
    vectors = np.empty((N_VECTORS, DIM), dtype=np.float32)
    
    for i in range(N_VECTORS):
        c_idx = i % N_CENTROIDS
        # Base centroid + slight noise
        v = centroids[c_idx] + np.random.randn(DIM)*0.1 
        vectors[i] = v
        
        entries.append(IndexEntry(
            id=str(uuid.uuid4()),
            item_type="memory",
            content=f"Synthetic vector {i}",
            importance=0.5,
            confidence=1.0,
            created_at="2026-04-20T00:00:00",
            decay_score=1.0,
            access_count=0,
            centroid_id=f"c_{c_idx}",
            heat_tier=1 # warm
        ))
        
    vectors = normalize_vectors(vectors)
    
    # 2. Setup Index
    index = VectorIndex(dim=DIM)
    index.bulk_add_entries(entries, list(vectors))
    
    # 3. Create Queries
    queries = normalize_vectors(np.random.randn(Q_RUNS, DIM).astype(np.float32))
    
    # ---------------------------------------------------------
    # Test 1: Full Exhaustive Scan (No Centroids Set)
    # ---------------------------------------------------------
    print("\n[Running Brute Force Scan]")
    index.set_centroids({}) # clear just in case
    
    t0 = time.perf_counter()
    for q in queries:
        res = index.search(q, top_k=10)
    t1 = time.perf_counter()
    full_scan_total = t1 - t0
    full_scan_avg = full_scan_total / Q_RUNS
    
    # ---------------------------------------------------------
    # Test 2: IVF Routing Scan
    # ---------------------------------------------------------
    print(f"\n[Running IVF Routing Scan (nprobe={N_PROBE})]")
    index.set_centroids(centroids_map)
    
    t2 = time.perf_counter()
    for q in queries:
        res = index.search(q, top_k=10, nprobe=N_PROBE)
    t3 = time.perf_counter()
    ivf_scan_total = t3 - t2
    ivf_scan_avg = ivf_scan_total / Q_RUNS
    
    print("\n--- Benchmark Results ---")
    print(f"Brute Force Avg Latency: {full_scan_avg*1000:.2f} ms / query")
    print(f"IVF Routing Avg Latency: {ivf_scan_avg*1000:.2f} ms / query")
    print(f"Speedup Factor:          {full_scan_avg / ivf_scan_avg:.2f}x")
    
if __name__ == "__main__":
    run_benchmark()
