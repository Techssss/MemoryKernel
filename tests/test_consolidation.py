import os
import uuid
import pytest
import numpy as np

from memk.core.runtime import get_runtime
from memk.consolidation.consolidator import ConsolidatorService
from memk.storage.db import _encode_blob

# Bypass daemon mode
os.environ["MEMK_DAEMON_MODE"] = "1"

def test_consolidation_job():
    workspace_id = f"test_consolidate_ws_{uuid.uuid4().hex[:8]}"
    tmp_path = f"test_{uuid.uuid4().hex[:8]}.db"
    
    global_runtime = get_runtime()
    global_runtime._is_global_initialized = True
    
    # Mock embedder pipeline
    class MockSharedEmbedder:
        def __init__(self):
            self.dim = 3
        def embed(self, t): return np.zeros(3, dtype=np.float32)
        
    global_runtime.shared_embedder = MockSharedEmbedder()
    global_runtime.embedder_pipeline = None
    
    runtime = global_runtime.get_workspace_runtime(workspace_id, db_path=tmp_path)
    
    try:
        # 1. Insert distinct memories bypassing standard service to inject specific embeddings
        db = runtime.db
        # Insert 3 identical/similar vector memories (cos sim > 0.9)
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.9, 0.1, 0.0], dtype=np.float32) # cos sim highly similar to vec1
        vec3 = np.array([0.8, 0.2, 0.0], dtype=np.float32) # also similar
        vec_diff = np.array([0.0, 1.0, 0.0], dtype=np.float32) # orthogonal
        
        id1 = db.insert_memory("Alice works at Google.", embedding=vec1)
        id2 = db.insert_memory("Alice is employed by Google.", embedding=vec2)
        id3 = db.insert_memory("Alice's current company is Google.", embedding=vec3)
        id_diff = db.insert_memory("Bob lives in Seattle.", embedding=vec_diff)
        
        # 2. Run consolidator
        consolidator = ConsolidatorService(runtime, cosine_threshold=0.85) # slightly lowered
        before_cands = consolidator.get_candidate_memories()
        assert len(before_cands) == 4
        
        clusters_made = consolidator.run_consolidation_job()
        assert clusters_made == 1
        
        # 3. Assert raw are archived
        after_cands = consolidator.get_candidate_memories()
        assert len(after_cands) == 1
        assert after_cands[0]["id"] == id_diff # Bob memory survives
        
        # 4. Check facts table
        with db._get_connection() as conn:
            facts = conn.execute("SELECT * FROM kg_fact").fetchall()
            assert len(facts) == 1
            f = facts[0]
            assert "Alice" in f["canonical_text"]
            assert "source_memories" in f["summary_json"]
            
            # 5. Check raw memories STILL exist (soft delete check)
            all_memories = conn.execute("SELECT id, archived FROM memories").fetchall()
            assert len(all_memories) == 4
            archived_mems = [m["id"] for m in all_memories if m["archived"] == 1]
            assert sorted(archived_mems) == sorted([id1, id2, id3])

    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                os.remove(tmp_path + "-wal")
                os.remove(tmp_path + "-shm")
            except:
                pass
