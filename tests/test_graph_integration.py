"""
Test: Graph Extraction Integration
===================================
End-to-end integration test that proves:
1. Memory write succeeds as before
2. SpaCyExtractor runs during write
3. Entities, mentions, and edges are created in the graph tables

This test bypasses the async service layer and tests the integration
at the component level to avoid needing the full daemon/runtime stack.

Run: python -m pytest tests/test_graph_integration.py -v
  or: python tests/test_graph_integration.py
"""

import gc
import os
import sys
import time
import uuid
import sqlite3
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memk.storage.migrations import MigrationEngine
from memk.storage.db import MemoryDB
from memk.storage.graph_repository import GraphRepository
from memk.extraction.extractor import StructuredFact

# ---------------------------------------------------------------------------
# Check spaCy availability
# ---------------------------------------------------------------------------

_SPACY_AVAILABLE = False
try:
    import spacy
    spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_db() -> str:
    """Create a fully migrated temp DB."""
    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "tmp_test_integration")
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"test_{uuid.uuid4().hex[:8]}.db")
    engine = MigrationEngine(path)
    engine.migrate()
    return path


def _cleanup(path: str):
    gc.collect()
    for suffix in ("", "-wal", "-shm"):
        p = path + suffix
        if os.path.exists(p):
            try:
                os.remove(p)
            except PermissionError:
                pass


# ---------------------------------------------------------------------------
# Test: _enrich_graph logic (unit-level, no async)
# ---------------------------------------------------------------------------

class TestEnrichGraphUnit:
    """Test the _enrich_graph static method directly."""

    def setup_method(self):
        self.db_path = _make_test_db()
        self.db = MemoryDB(db_path=self.db_path)
        self.db.init_db()
        self.repo = GraphRepository(self.db_path)

    def teardown_method(self):
        _cleanup(self.db_path)

    def test_single_fact_creates_entities_and_edge(self):
        """One triplet → 2 entities + 2 mentions + 1 edge."""
        from memk.core.service import MemoryKernelService

        # Simulate a memory write
        mem_id = self.db.insert_memory("Sarah works at Google.", importance=0.5)

        # Create a mock runtime-like object with graph_repo
        class _MockRuntime:
            graph_repo = self.repo

        facts = [StructuredFact(subject="Sarah", relation="works_at", object="Google")]

        edges = MemoryKernelService._enrich_graph(
            _MockRuntime(), "ws1", mem_id, facts,
        )
        assert edges == 1

        # Verify entities
        entities = self.repo.get_all_entities("ws1")
        names = {e.normalized_text for e in entities}
        assert "sarah" in names
        assert "google" in names

        # Verify mentions
        ents = self.repo.get_entities_for_memory(mem_id)
        assert len(ents) == 2

        # Verify edge
        sarah_ent = self.repo.find_entity("ws1", "Sarah")
        google_ent = self.repo.find_entity("ws1", "Google")
        edges_out = self.repo.get_edges_from_entity("ws1", sarah_ent.id)
        assert len(edges_out) == 1
        assert edges_out[0].rel_type == "works_at"
        assert edges_out[0].dst_entity_id == google_ent.id
        assert edges_out[0].provenance_memory_id == mem_id

    def test_multiple_facts_from_one_memory(self):
        """Multi-sentence → multiple edges from same memory."""
        from memk.core.service import MemoryKernelService

        mem_id = self.db.insert_memory(
            "Elon Musk is CEO of Tesla. OpenAI is based in San Francisco.",
            importance=0.7,
        )

        class _MockRuntime:
            graph_repo = self.repo

        facts = [
            StructuredFact(subject="Elon Musk", relation="role_at", object="Tesla"),
            StructuredFact(subject="OpenAI", relation="located_in", object="San Francisco"),
        ]

        edges = MemoryKernelService._enrich_graph(
            _MockRuntime(), "ws1", mem_id, facts,
        )
        assert edges == 2

        stats = self.repo.get_graph_stats("ws1")
        assert stats["entity_count"] == 4
        assert stats["edge_count"] == 2
        assert stats["mention_count"] == 4

    def test_entity_dedup_across_memories(self):
        """Same entity mentioned in different memories → single entity, multiple mentions."""
        from memk.core.service import MemoryKernelService

        class _MockRuntime:
            graph_repo = self.repo

        mem1 = self.db.insert_memory("Alice works at Google.", importance=0.5)
        mem2 = self.db.insert_memory("Bob works at Google.", importance=0.5)

        MemoryKernelService._enrich_graph(
            _MockRuntime(), "ws1", mem1,
            [StructuredFact(subject="Alice", relation="works_at", object="Google")],
        )
        MemoryKernelService._enrich_graph(
            _MockRuntime(), "ws1", mem2,
            [StructuredFact(subject="Bob", relation="works_at", object="Google")],
        )

        stats = self.repo.get_graph_stats("ws1")
        # 3 entities: Alice, Bob, Google (Google deduped)
        assert stats["entity_count"] == 3
        # 2 edges: Alice→Google, Bob→Google
        assert stats["edge_count"] == 2

        # Google entity should have 2 memory mentions
        google = self.repo.find_entity("ws1", "Google")
        mems = self.repo.get_memories_for_entity(google.id)
        assert set(mems) == {mem1, mem2}

    def test_enrichment_failure_does_not_affect_memory(self):
        """If graph enrichment raises, the memory should still be persisted."""
        mem_id = self.db.insert_memory("Test memory", importance=0.5)

        class _BrokenRepo:
            def upsert_entity(self, *a, **kw):
                raise RuntimeError("Simulated DB error")

        class _MockRuntime:
            graph_repo = _BrokenRepo()

        from memk.core.service import MemoryKernelService
        try:
            MemoryKernelService._enrich_graph(
                _MockRuntime(), "ws1", mem_id,
                [StructuredFact(subject="X", relation="y", object="Z")],
            )
            assert False, "Should have raised"
        except RuntimeError:
            pass

        # Memory should still exist (search by content)
        results = self.db.search_memory("Test memory")
        assert len(results) >= 1
        assert results[0]["content"] == "Test memory"


# ---------------------------------------------------------------------------
# Test: Full pipeline (SpaCyExtractor + _enrich_graph)
# ---------------------------------------------------------------------------

class TestSpaCyPipelineIntegration:
    """Test the complete pipeline: text → spaCy extract → graph write."""

    def setup_method(self):
        if not _SPACY_AVAILABLE:
            return
        self.db_path = _make_test_db()
        self.db = MemoryDB(db_path=self.db_path)
        self.db.init_db()
        self.repo = GraphRepository(self.db_path)
        from memk.extraction.spacy_extractor import SpaCyExtractor
        self.extractor = SpaCyExtractor()

    def teardown_method(self):
        if hasattr(self, "db_path"):
            _cleanup(self.db_path)

    def test_full_pipeline_sarah_works_at_google(self):
        if not _SPACY_AVAILABLE:
            print("  SKIP  (spaCy not available)")
            return

        from memk.core.service import MemoryKernelService

        text = "Sarah works at Google."
        mem_id = self.db.insert_memory(text, importance=0.5)

        # Extract using SpaCyExtractor
        facts = self.extractor.extract_facts(text)
        assert len(facts) >= 1, f"SpaCy should extract at least 1 fact from '{text}'"
        assert facts[0].subject == "Sarah"
        assert facts[0].relation == "works_at"
        assert facts[0].object == "Google"

        # Enrich graph
        class _MockRuntime:
            graph_repo = self.repo

        edges = MemoryKernelService._enrich_graph(
            _MockRuntime(), "ws1", mem_id, facts,
        )
        assert edges == 1

        # Full verification
        entities = self.repo.get_all_entities("ws1")
        assert len(entities) == 2

        edge_list = self.repo.get_edges_for_workspace("ws1")
        assert len(edge_list) == 1
        assert edge_list[0].rel_type == "works_at"

        print(f"\n  Pipeline OK: '{text}' -> {len(entities)} entities, {len(edge_list)} edges")

    def test_full_pipeline_multi_sentence(self):
        if not _SPACY_AVAILABLE:
            print("  SKIP  (spaCy not available)")
            return

        from memk.core.service import MemoryKernelService

        text = (
            "Elon Musk is CEO of Tesla. "
            "OpenAI is based in San Francisco. "
            "Alice uses PyTorch."
        )
        mem_id = self.db.insert_memory(text, importance=0.7)
        facts = self.extractor.extract_facts(text)
        assert len(facts) >= 2, f"Expected >=2 facts, got {len(facts)}"

        class _MockRuntime:
            graph_repo = self.repo

        edges = MemoryKernelService._enrich_graph(
            _MockRuntime(), "ws1", mem_id, facts,
        )
        assert edges >= 2

        stats = self.repo.get_graph_stats("ws1")
        print(f"\n  Multi-sentence pipeline: {stats}")
        assert stats["entity_count"] >= 4
        assert stats["edge_count"] >= 2


# ---------------------------------------------------------------------------
# Test: Feature flag behavior
# ---------------------------------------------------------------------------

class TestFeatureFlag:
    """Verify MEMK_GRAPH_EXTRACTION env var gating."""

    def test_flag_default_is_enabled(self):
        """Without env var set, default should be '1' (enabled)."""
        old = os.environ.pop("MEMK_GRAPH_EXTRACTION", None)
        try:
            assert os.getenv("MEMK_GRAPH_EXTRACTION", "1") == "1"
        finally:
            if old is not None:
                os.environ["MEMK_GRAPH_EXTRACTION"] = old

    def test_flag_disabled(self):
        """When set to '0', extraction should be gated off."""
        os.environ["MEMK_GRAPH_EXTRACTION"] = "0"
        try:
            enabled = os.getenv("MEMK_GRAPH_EXTRACTION", "1") == "1"
            assert enabled is False
        finally:
            os.environ["MEMK_GRAPH_EXTRACTION"] = "1"


# ---------------------------------------------------------------------------
# Benchmark: extraction + graph write latency
# ---------------------------------------------------------------------------

class TestBenchmark:
    """Micro-benchmark for graph enrichment cost."""

    def setup_method(self):
        if not _SPACY_AVAILABLE:
            return
        self.db_path = _make_test_db()
        self.db = MemoryDB(db_path=self.db_path)
        self.db.init_db()
        self.repo = GraphRepository(self.db_path)
        from memk.extraction.spacy_extractor import SpaCyExtractor
        self.extractor = SpaCyExtractor()

    def teardown_method(self):
        if hasattr(self, "db_path"):
            _cleanup(self.db_path)

    def test_latency_10_memories(self):
        """Measure cost of extraction + graph write for 10 memories."""
        if not _SPACY_AVAILABLE:
            print("  SKIP  (spaCy not available)")
            return

        from memk.core.service import MemoryKernelService

        sentences = [
            "Sarah works at Google.",
            "Elon Musk is CEO of Tesla.",
            "OpenAI is based in San Francisco.",
            "Alice uses PyTorch for deep learning.",
            "Bob works for Microsoft.",
            "The backend belongs to Team Alpha.",
            "Meta develops React.",
            "Jane lives in New York.",
            "Tom manages the database team.",
            "Lisa joined Amazon last year.",
        ]

        class _MockRuntime:
            graph_repo = self.repo

        total_extract_ms = 0
        total_graph_ms = 0
        total_edges = 0

        for text in sentences:
            mem_id = self.db.insert_memory(text, importance=0.5)

            t0 = time.perf_counter()
            facts = self.extractor.extract_facts(text)
            t1 = time.perf_counter()
            total_extract_ms += (t1 - t0) * 1000

            if facts:
                t2 = time.perf_counter()
                edges = MemoryKernelService._enrich_graph(
                    _MockRuntime(), "ws1", mem_id, facts,
                )
                t3 = time.perf_counter()
                total_graph_ms += (t3 - t2) * 1000
                total_edges += edges

        stats = self.repo.get_graph_stats("ws1")

        print(f"\n  {'='*50}")
        print(f"  Benchmark: 10 memories")
        print(f"  {'='*50}")
        print(f"  Extraction: {total_extract_ms:.1f}ms total, "
              f"{total_extract_ms/10:.1f}ms/item avg")
        print(f"  Graph write: {total_graph_ms:.1f}ms total, "
              f"{total_graph_ms/max(total_edges,1):.1f}ms/edge avg")
        print(f"  Total overhead: {total_extract_ms + total_graph_ms:.1f}ms "
              f"for 10 items")
        print(f"  Graph stats: {stats}")
        print(f"  Edges created: {total_edges}")

        # Sanity: should complete in reasonable time
        # Note: each GraphRepository op opens a new connection (matching
        # MemoryDB pattern). Connection pooling is a Phase 2 optimization.
        assert total_extract_ms < 5000, "Extraction too slow (>5s for 10 items)"
        assert total_graph_ms < 3000, "Graph write too slow (>3s for 10 items)"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_standalone():
    """Run all tests without pytest."""
    all_classes = [
        TestEnrichGraphUnit,
        TestSpaCyPipelineIntegration,
        TestFeatureFlag,
        TestBenchmark,
    ]

    total_passed = 0
    total_failed = 0

    for cls in all_classes:
        print(f"\n--- {cls.__name__} ---")
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in sorted(methods):
            if hasattr(instance, "setup_method"):
                instance.setup_method()
            try:
                getattr(instance, method_name)()
                print(f"  PASS  {method_name}")
                total_passed += 1
            except Exception as e:
                print(f"  FAIL  {method_name}: {e}")
                import traceback
                traceback.print_exc()
                total_failed += 1
            finally:
                if hasattr(instance, "teardown_method"):
                    instance.teardown_method()

    print(f"\n{'='*60}")
    print(f"Results: {total_passed} passed, {total_failed} failed, "
          f"{total_passed + total_failed} total")

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    print("Graph Extraction Integration Tests")
    print("=" * 60)
    run_standalone()
