import gc
import os
import sys
import time
import uuid
import numpy as np

from memk.storage.migrations import MigrationEngine
from memk.storage.db import MemoryDB
from memk.storage.graph_repository import GraphRepository
from memk.core.graph_index import GraphIndex
from memk.extraction.spacy_extractor import SpaCyExtractor
from memk.core.service import MemoryKernelService
from memk.core.scorer import MemoryScorer, ScoringWeights
from memk.retrieval.retriever import ScoredRetriever

class MockEntry:
    def __init__(self, id, item_type="memory"):
        self.id = id
        self.item_type = item_type
        self.content = "mock"
        self.created_at = "2025-01-01T00:00:00"
        self.importance = 0.5
        self.confidence = 1.0
        self.access_count = 0
        self.decay_score = 1.0

class MockRuntime:
    def __init__(self, repo):
        self.graph_repo = repo

def _make_db():
    p = os.path.join(".", f"bench_{uuid.uuid4().hex[:8]}.db")
    MigrationEngine(p).migrate()
    return p

def _cleanup(p):
    gc.collect()
    for suffix in ["", "-wal", "-shm"]:
        if os.path.exists(p + suffix):
            try: os.remove(p + suffix)
            except Exception: pass

def run_benchmarks():
    print("="*60)
    print("PHASE 1 - GRAPH HYBRID BENCHMARKS")
    print("="*60)
    
    db_path = _make_db()
    db = MemoryDB(db_path)
    db.init_db()
    repo = GraphRepository(db_path)
    ws = "bench_ws"
    
    # ----------------------------------------------------
    # 1. WRITE LATENCY
    # ----------------------------------------------------
    print("\n--- 1. Write Latency ---")
    data = ["Tim Cook is CEO of Apple, based in Cupertino.", "Machine learning relies on datasets."] * 50
    extractor = SpaCyExtractor()
    runtime = MockRuntime(repo)
    
    t0 = time.perf_counter()
    extracted_facts = []
    pure_mems = []
    for txt in data:
        mid = db.insert_memory(txt, importance=0.5)
        pure_mems.append(mid)
    db_time = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    for txt in data:
        facts = extractor.extract_facts(txt)
        extracted_facts.append(facts)
    extract_time = (time.perf_counter() - t0) * 1000
    
    t0 = time.perf_counter()
    for mid, facts in zip(pure_mems, extracted_facts):
        if facts:
            MemoryKernelService._enrich_graph(runtime, ws, mid, facts)
    enrich_time = (time.perf_counter() - t0) * 1000
    
    print(f"Items processed: {len(data)}")
    print(f"Base SQLite Write : {db_time:.2f} ms ({db_time/len(data):.2f} ms/item)")
    print(f"SpaCy Extraction  : {extract_time:.2f} ms ({extract_time/len(data):.2f} ms/item)")
    print(f"Graph Enrichment  : {enrich_time:.2f} ms ({enrich_time/len(data):.2f} ms/item)")
    print(f"Total write path overhead => factor of +{((extract_time+enrich_time)/db_time):.1f}x")
    
    # ----------------------------------------------------
    # 2. GRAPH INDEX REBUILD
    # ----------------------------------------------------
    print("\n--- 2. Build Graph Index Latency ---")
    idx = GraphIndex(db_path)
    
    t0 = time.perf_counter()
    idx.build_from_db(ws)
    build_time = (time.perf_counter() - t0) * 1000
    
    stats = idx.get_stats()
    print(f"Graph Size        : {stats}")
    print(f"Build Time (RAM)  : {build_time:.2f} ms")
    
    # ----------------------------------------------------
    # 3. RETRIEVAL PROPAGATION
    # ----------------------------------------------------
    print("\n--- 3. Retrieval + Propagation Latency ---")
    # Fabricate 1000 candidate items
    index_hits = [(MockEntry(mid), np.random.uniform(0, 1)) for mid in pure_mems] 
    
    # Baseline vector scoring
    scorer = MemoryScorer(ScoringWeights())
    retriever = ScoredRetriever(db, scorer, track_access=False)
    
    t0 = time.perf_counter()
    for _ in range(50):
        retriever.rank_candidates("test", np.zeros(1), index_hits, limit=10)
    base_rank_time = ((time.perf_counter() - t0) * 1000) / 50
    
    # PPNP vector scoring
    g_scorer = MemoryScorer(ScoringWeights(w6=0.5))
    g_retriever = ScoredRetriever(db, g_scorer, track_access=False)
    
    t0 = time.perf_counter()
    for _ in range(50):
        g_retriever.rank_candidates("test", np.zeros(1), index_hits, limit=10, graph_index=idx)
    graph_rank_time = ((time.perf_counter() - t0) * 1000) / 50
    
    print(f"Candidates Ranked : {len(index_hits)}")
    print(f"Base P1 Ranking   : {base_rank_time:.2f} ms (avg over 50 iter)")
    print(f"Graph P2 Ranking  : {graph_rank_time:.2f} ms (avg over 50 iter)")
    print(f"Difference (cost) : {graph_rank_time - base_rank_time:.2f} ms overhead per query")

    _cleanup(db_path)

if __name__ == "__main__":
    run_benchmarks()
