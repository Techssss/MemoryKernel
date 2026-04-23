"""
tests.test_graph_index
======================

Unit tests for the NumPy-based GraphIndex loader.
Checks that SQLite relational graph structures are correctly transformed
into CSR format for high-speed computation.
"""
import os
import sys
import uuid
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memk.storage.migrations import MigrationEngine
from memk.storage.db import MemoryDB
from memk.storage.graph_repository import GraphRepository
from memk.core.graph_index import GraphIndex

def _make_test_db() -> str:
    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "tmp_test_graphindex")
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"test_{uuid.uuid4().hex[:8]}.db")
    engine = MigrationEngine(path)
    engine.migrate()
    return path

class TestGraphIndex:

    def setup_method(self):
        self.ws_id = "ws1"
        self.db_path = _make_test_db()
        self.db = MemoryDB(self.db_path)
        self.db.init_db()
        self.repo = GraphRepository(self.db_path)
        self.index = GraphIndex(self.db_path)

    def teardown_method(self):
        import gc
        gc.collect()
        for suffix in ("-wal", "-shm", ""):
            p = self.db_path + suffix
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    def test_empty_graph(self):
        self.index.build_from_db(self.ws_id)
        assert self.index.num_entities == 0
        assert self.index.num_memories == 0
        
        # Check empty arrays
        assert len(self.index.e2e_indptr) == 1
        assert len(self.index.e2e_indices) == 0

    def test_build_from_db_populates_csr(self):
        # 1. Define entities
        e_sarah = self.repo.upsert_entity(self.ws_id, "Sarah")
        e_google = self.repo.upsert_entity(self.ws_id, "Google")
        e_apple = self.repo.upsert_entity(self.ws_id, "Apple")
        
        # 1.5 Define memories inside DB to pass Foreign Key constraint
        mem1 = self.db.insert_memory("Test memory 1", importance=0.5)
        mem2 = self.db.insert_memory("Test memory 2", importance=0.5)

        # 2. Add some test edges
        self.repo.add_edge(self.ws_id, e_sarah, "works_at", e_google, provenance_memory_id=mem1)
        self.repo.add_edge(self.ws_id, e_sarah, "lives_in", e_apple, provenance_memory_id=mem2)

        # 3. Add mentions (Sarah mentioned in mem1 and mem2, Google in mem1, Apple in mem2)
        self.repo.add_mention(mem1, e_sarah, role_hint="subject")
        self.repo.add_mention(mem1, e_google, role_hint="object")
        self.repo.add_mention(mem2, e_sarah, role_hint="subject")
        self.repo.add_mention(mem2, e_apple, role_hint="object")

        # 4. Build index
        self.index.build_from_db(self.ws_id)

        # Basic Stats checks
        stats = self.index.get_stats()
        assert stats["num_entities"] == 3
        assert stats["num_memories"] == 2
        assert stats["e2e_edges"] == 2
        assert stats["m2e_mentions"] == 4
        assert stats["e2m_mentions"] == 4

        # Validate Identity Mappings
        assert self.index.entity_id_map[e_sarah] in range(3)
        assert self.index.entity_id_map[e_google] in range(3)
        assert self.index.entity_id_map[e_apple] in range(3)
        
        sarah_idx = self.index.entity_id_map[e_sarah]
        google_idx = self.index.entity_id_map[e_google]
        apple_idx = self.index.entity_id_map[e_apple]

        # CSR Entity -> Entity 
        start_ptr = self.index.e2e_indptr[sarah_idx]
        end_ptr = self.index.e2e_indptr[sarah_idx + 1]
        outbound = list(self.index.e2e_indices[start_ptr:end_ptr])
        
        assert google_idx in outbound
        assert apple_idx in outbound

        # CSR Memory -> Entity (mem1 should point to Sarah and Google)
        mem1_idx = self.index.memory_id_map[mem1]
        m1_start = self.index.m2e_indptr[mem1_idx]
        m1_end = self.index.m2e_indptr[mem1_idx + 1]
        m1_mentions = list(self.index.m2e_indices[m1_start:m1_end])
        
        assert sarah_idx in m1_mentions
        assert google_idx in m1_mentions
        assert apple_idx not in m1_mentions

        # CSR Entity -> Memory (Google is only in mem1)
        g_start = self.index.e2m_indptr[google_idx]
        g_end = self.index.e2m_indptr[google_idx + 1]
        g_memories = list(self.index.e2m_indices[g_start:g_end])
        
        assert mem1_idx in g_memories

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
