"""
tests.test_graph_retrieval
==========================
Verify that graph propagation correctly intercepts the 
ScoredRetriever flow and bridges multi-hop queries.
"""
import pytest
import os
import uuid
import numpy as np

from memk.storage.migrations import MigrationEngine
from memk.storage.db import MemoryDB
from memk.storage.graph_repository import GraphRepository
from memk.core.graph_index import GraphIndex
from memk.core.scorer import MemoryScorer, ScoringWeights
from memk.retrieval.retriever import ScoredRetriever

class MockEntry:
    def __init__(self, id, item_type="memory", content="", created_at="2025-01-01T00:00:00", importance=0.5, confidence=1.0, access_count=0, decay_score=1.0):
        self.id = id
        self.item_type = item_type
        self.content = content
        self.created_at = created_at
        self.importance = importance
        self.confidence = confidence
        self.access_count = access_count
        self.decay_score = decay_score

def _make_test_db() -> str:
    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "tmp_test_integration")
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"test_{uuid.uuid4().hex[:8]}.db")
    engine = MigrationEngine(path)
    engine.migrate()
    return path

def _cleanup(path: str):
    import gc
    gc.collect()
    for suffix in ("", "-wal", "-shm"):
        p = path + suffix
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

def test_graph_bonus_via_entities():
    db_path = _make_test_db()
    db = MemoryDB(db_path=db_path)
    db.init_db()
    repo = GraphRepository(db_path)
    
    workspace = "default"
    
    try:
        # DB setup
        mem1_id = db.insert_memory("Tim Cook is the CEO of Apple.", importance=0.5)
        mem2_id = db.insert_memory("Tim Cook graduated from Duke University.", importance=0.5)
        mem3_id = db.insert_memory("Completely unrelated memory about space.", importance=0.5)
        
        # Populate graph repo directly to simulate enrichment
        # Entity 1: Tim Cook, Entity 2: Apple, Entity 3: Duke
        tc_id = repo.upsert_entity(workspace, "Tim Cook")
        apl_id = repo.upsert_entity(workspace, "Apple")
        duke_id = repo.upsert_entity(workspace, "Duke")
        
        # Mentions: memory <-> entity
        repo.add_mention(mem1_id, tc_id, role_hint="subject")
        repo.add_mention(mem1_id, apl_id, role_hint="object")
        repo.add_mention(mem2_id, tc_id, role_hint="subject")
        repo.add_mention(mem2_id, duke_id, role_hint="object")
        
        # Edges (e2e)
        repo.add_edge(workspace, tc_id, "ceo_of", apl_id, provenance_memory_id=mem1_id)
        repo.add_edge(workspace, tc_id, "educated_at", duke_id, provenance_memory_id=mem2_id)
        
        # Build RAM graph index
        g_index = GraphIndex(db_path)
        g_index.build_from_db(workspace)
        
        # Prepare ranking engine with graph weights activated
        # Note: we set score_threshold very low so mem2 and mem3 can pass if they have 0 base score.
        weights = ScoringWeights(w1=0.5, w6=0.5) # Emphasize vectors and graphs
        retriever = ScoredRetriever(db, weights=weights, score_threshold=0.0)
        
        # MOCK VECTOR INDEX hits.
        # Suppose the user queried "CEO of Apple"
        # Mem 1 gets Perfect Hit (1.0)
        # Mem 2 gets No Hit (0.0)
        # Mem 3 gets No Hit (0.0)
        index_hits = [
            (MockEntry(id=mem1_id), 1.0),
            (MockEntry(id=mem2_id), 0.0),
            (MockEntry(id=mem3_id), 0.0)
        ]
        
        results = retriever.rank_candidates(
            query="Apple CEO", 
            q_vec=np.zeros(1536), 
            index_hits=index_hits, 
            limit=5, 
            graph_index=g_index
        )
        
        assert len(results) == 3
        
        # mem1 should be top because 1.0 sim
        # mem2 should be second because Graph propagated from Mem 1 -> Tim Cook -> Mem 2!
        # mem3 should have no graph propagation because it shares no entities.
        
        result_map = {r.id: r for r in results}
        
        bd1 = result_map[mem1_id].breakdown
        bd2 = result_map[mem2_id].breakdown
        bd3 = result_map[mem3_id].breakdown
        
        # Validate graph scores 
        assert bd1.graph_score >= 0.0
        assert bd2.graph_score > 0.0  # Hoisted by Tim Cook!
        assert bd3.graph_score == 0.0 # Completely isolated
        
        # Validate Mem 2 ended up ranked strictly higher than Mem 3
        assert result_map[mem2_id].score > result_map[mem3_id].score
        
    finally:
        _cleanup(db_path)

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
