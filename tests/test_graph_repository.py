"""
Test: GraphRepository — Knowledge Graph Storage Layer
=====================================================
Covers all CRUD operations, normalization, idempotency, and edge cases.

Run: python -m pytest tests/test_graph_repository.py -v
  or: python tests/test_graph_repository.py
"""

import os
import sys
import uuid
import json
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memk.storage.migrations import MigrationEngine
from memk.storage.graph_models import normalize_entity_text
from memk.storage.graph_repository import GraphRepository


def _make_temp_db() -> str:
    """Create a migrated temp DB for testing."""
    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "tmp_test_graph")
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"test_{uuid.uuid4().hex[:8]}.db")
    engine = MigrationEngine(path)
    engine.migrate()
    return path


def _cleanup_db(db_path: str):
    """Force-close lingering connections and remove DB files (Windows-safe)."""
    gc.collect()
    for suffix in ("", "-wal", "-shm"):
        p = db_path + suffix
        if os.path.exists(p):
            try:
                os.remove(p)
            except PermissionError:
                pass  # Windows WAL lock — harmless in test cleanup


class TestNormalizeEntityText:
    """Test the normalization helper."""

    def test_lowercase(self):
        assert normalize_entity_text("FastAPI") == "fastapi"

    def test_strip_whitespace(self):
        assert normalize_entity_text("  Google LLC  ") == "google llc"

    def test_collapse_spaces(self):
        assert normalize_entity_text("Team   Alpha   Beta") == "team alpha beta"

    def test_tabs_and_newlines(self):
        assert normalize_entity_text("hello\t\n  world") == "hello world"

    def test_empty_string(self):
        assert normalize_entity_text("") == ""

    def test_single_word(self):
        assert normalize_entity_text("Python") == "python"


class TestEntityCRUD:
    """Test entity upsert and query operations."""

    def setup_method(self):
        self.db_path = _make_temp_db()
        self.repo = GraphRepository(self.db_path)

    def teardown_method(self):
        _cleanup_db(self.db_path)

    def test_upsert_creates_new_entity(self):
        eid = self.repo.upsert_entity("ws1", "Google LLC", entity_type="ORG")
        assert isinstance(eid, int)
        assert eid > 0

    def test_upsert_returns_same_id_on_duplicate(self):
        eid1 = self.repo.upsert_entity("ws1", "Google LLC", entity_type="ORG")
        eid2 = self.repo.upsert_entity("ws1", "  google   llc  ", entity_type="ORG")
        assert eid1 == eid2, "Same normalized text should return same entity"

    def test_upsert_different_type_creates_new_entity(self):
        eid1 = self.repo.upsert_entity("ws1", "Python", entity_type="LANG")
        eid2 = self.repo.upsert_entity("ws1", "Python", entity_type="PERSON")
        assert eid1 != eid2, "Different entity_type should create separate entities"

    def test_upsert_different_workspace_creates_new_entity(self):
        eid1 = self.repo.upsert_entity("ws1", "Team Alpha")
        eid2 = self.repo.upsert_entity("ws2", "Team Alpha")
        assert eid1 != eid2, "Different workspace should create separate entities"

    def test_upsert_bumps_confidence_to_max(self):
        self.repo.upsert_entity("ws1", "FastAPI", confidence=0.3)
        self.repo.upsert_entity("ws1", "FastAPI", confidence=0.9)
        entity = self.repo.find_entity("ws1", "FastAPI")
        assert entity is not None
        assert entity.confidence == 0.9

    def test_upsert_does_not_lower_confidence(self):
        self.repo.upsert_entity("ws1", "FastAPI", confidence=0.9)
        self.repo.upsert_entity("ws1", "FastAPI", confidence=0.3)
        entity = self.repo.find_entity("ws1", "FastAPI")
        assert entity.confidence == 0.9

    def test_upsert_updates_last_seen(self):
        self.repo.upsert_entity("ws1", "React")
        e1 = self.repo.find_entity("ws1", "React")
        first_ts = e1.last_seen_ts

        self.repo.upsert_entity("ws1", "React")
        e2 = self.repo.find_entity("ws1", "React")
        assert e2.last_seen_ts >= first_ts

    def test_upsert_empty_text_raises(self):
        try:
            self.repo.upsert_entity("ws1", "")
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_get_entity_by_id(self):
        eid = self.repo.upsert_entity("ws1", "NumPy", entity_type="LIB")
        entity = self.repo.get_entity(eid)
        assert entity is not None
        assert entity.canonical_text == "NumPy"
        assert entity.normalized_text == "numpy"
        assert entity.entity_type == "LIB"

    def test_get_entity_nonexistent(self):
        entity = self.repo.get_entity(99999)
        assert entity is None

    def test_find_entity_normalizes_query(self):
        self.repo.upsert_entity("ws1", "Google LLC", entity_type="ORG")
        entity = self.repo.find_entity("ws1", "  GOOGLE   LLC  ", entity_type="ORG")
        assert entity is not None
        assert entity.canonical_text == "Google LLC"

    def test_get_all_entities(self):
        self.repo.upsert_entity("ws1", "Alpha")
        self.repo.upsert_entity("ws1", "Beta")
        self.repo.upsert_entity("ws2", "Gamma")

        entities = self.repo.get_all_entities("ws1")
        assert len(entities) == 2
        names = {e.normalized_text for e in entities}
        assert names == {"alpha", "beta"}

    def test_upsert_entity_type_none_handled(self):
        """entity_type=None should be handled distinctly."""
        eid1 = self.repo.upsert_entity("ws1", "Thing", entity_type=None)
        eid2 = self.repo.upsert_entity("ws1", "Thing", entity_type=None)
        assert eid1 == eid2


class TestMentionCRUD:
    """Test mention link operations."""

    def setup_method(self):
        self.db_path = _make_temp_db()
        self.repo = GraphRepository(self.db_path)

    def teardown_method(self):
        _cleanup_db(self.db_path)

    def test_add_mention_basic(self):
        eid = self.repo.upsert_entity("ws1", "FastAPI")
        mem_id = str(uuid.uuid4())
        self.repo.add_mention(mem_id, eid, role_hint="subject")

        entities = self.repo.get_entities_for_memory(mem_id)
        assert len(entities) == 1
        assert entities[0].id == eid

    def test_add_mention_idempotent(self):
        """Same (memory_id, entity_id, role_hint) should not create duplicates."""
        eid = self.repo.upsert_entity("ws1", "Python")
        mem_id = str(uuid.uuid4())
        self.repo.add_mention(mem_id, eid, role_hint="subject")
        self.repo.add_mention(mem_id, eid, role_hint="subject")

        mentions = self.repo.get_mentions_for_memory(mem_id)
        assert len(mentions) == 1

    def test_add_mention_different_roles(self):
        """Same entity can be mentioned with different roles in same memory."""
        eid = self.repo.upsert_entity("ws1", "Python")
        mem_id = str(uuid.uuid4())
        self.repo.add_mention(mem_id, eid, role_hint="subject")
        self.repo.add_mention(mem_id, eid, role_hint="object")

        mentions = self.repo.get_mentions_for_memory(mem_id)
        assert len(mentions) == 2

    def test_add_mention_with_span(self):
        eid = self.repo.upsert_entity("ws1", "SQLite")
        mem_id = str(uuid.uuid4())
        self.repo.add_mention(
            mem_id, eid,
            start_char=10, end_char=16,
            role_hint="object", weight=0.8,
        )

        mentions = self.repo.get_mentions_for_memory(mem_id)
        assert len(mentions) == 1
        m = mentions[0]
        assert m.start_char == 10
        assert m.end_char == 16
        assert m.weight == 0.8

    def test_get_entities_for_memory_multiple(self):
        e1 = self.repo.upsert_entity("ws1", "Team A")
        e2 = self.repo.upsert_entity("ws1", "Project X")
        mem_id = str(uuid.uuid4())
        self.repo.add_mention(mem_id, e1, role_hint="subject")
        self.repo.add_mention(mem_id, e2, role_hint="object")

        entities = self.repo.get_entities_for_memory(mem_id)
        assert len(entities) == 2

    def test_get_entities_for_memory_empty(self):
        entities = self.repo.get_entities_for_memory("nonexistent")
        assert entities == []

    def test_get_memories_for_entity(self):
        eid = self.repo.upsert_entity("ws1", "Django")
        mem1 = str(uuid.uuid4())
        mem2 = str(uuid.uuid4())
        self.repo.add_mention(mem1, eid, role_hint="subject")
        self.repo.add_mention(mem2, eid, role_hint="object")

        memory_ids = self.repo.get_memories_for_entity(eid)
        assert set(memory_ids) == {mem1, mem2}


class TestEdgeCRUD:
    """Test edge operations."""

    def setup_method(self):
        self.db_path = _make_temp_db()
        self.repo = GraphRepository(self.db_path)

    def teardown_method(self):
        _cleanup_db(self.db_path)

    def _setup_two_entities(self):
        src = self.repo.upsert_entity("ws1", "Team Alpha", entity_type="TEAM")
        dst = self.repo.upsert_entity("ws1", "Project X", entity_type="PROJECT")
        return src, dst

    def test_add_edge_basic(self):
        src, dst = self._setup_two_entities()
        mem_id = str(uuid.uuid4())
        edge_id = self.repo.add_edge(
            "ws1", src, "manages", dst,
            provenance_memory_id=mem_id, confidence=0.85,
        )
        assert isinstance(edge_id, int)
        assert edge_id > 0

    def test_add_edge_without_provenance_raises(self):
        src, dst = self._setup_two_entities()
        try:
            self.repo.add_edge(
                "ws1", src, "manages", dst,
                provenance_memory_id="",
            )
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_add_edge_empty_rel_type_raises(self):
        src, dst = self._setup_two_entities()
        try:
            self.repo.add_edge(
                "ws1", src, "", dst,
                provenance_memory_id="mem-123",
            )
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_get_edges_from_entity(self):
        src, dst = self._setup_two_entities()
        mem_id = str(uuid.uuid4())
        self.repo.add_edge(
            "ws1", src, "manages", dst,
            provenance_memory_id=mem_id,
        )

        edges = self.repo.get_edges_from_entity("ws1", src)
        assert len(edges) == 1
        assert edges[0].rel_type == "manages"
        assert edges[0].dst_entity_id == dst

    def test_get_edges_to_entity(self):
        src, dst = self._setup_two_entities()
        mem_id = str(uuid.uuid4())
        self.repo.add_edge(
            "ws1", src, "manages", dst,
            provenance_memory_id=mem_id,
        )

        edges = self.repo.get_edges_to_entity("ws1", dst)
        assert len(edges) == 1
        assert edges[0].src_entity_id == src

    def test_get_edges_for_workspace(self):
        src, dst = self._setup_two_entities()
        mem1, mem2 = str(uuid.uuid4()), str(uuid.uuid4())
        self.repo.add_edge("ws1", src, "manages", dst, provenance_memory_id=mem1)
        self.repo.add_edge("ws1", dst, "depends_on", src, provenance_memory_id=mem2)

        edges = self.repo.get_edges_for_workspace("ws1")
        assert len(edges) == 2

    def test_get_edges_for_workspace_excludes_archived(self):
        src, dst = self._setup_two_entities()
        mem_id = str(uuid.uuid4())
        edge_id = self.repo.add_edge(
            "ws1", src, "manages", dst,
            provenance_memory_id=mem_id,
        )
        self.repo.archive_edge(edge_id)

        edges = self.repo.get_edges_for_workspace("ws1")
        assert len(edges) == 0

    def test_get_edges_for_workspace_includes_archived(self):
        src, dst = self._setup_two_entities()
        mem_id = str(uuid.uuid4())
        edge_id = self.repo.add_edge(
            "ws1", src, "manages", dst,
            provenance_memory_id=mem_id,
        )
        self.repo.archive_edge(edge_id)

        edges = self.repo.get_edges_for_workspace("ws1", include_archived=True)
        assert len(edges) == 1
        assert edges[0].archived == 1

    def test_get_edges_for_memory(self):
        src, dst = self._setup_two_entities()
        mem_id = str(uuid.uuid4())
        self.repo.add_edge("ws1", src, "manages", dst, provenance_memory_id=mem_id)
        self.repo.add_edge("ws1", dst, "uses", src, provenance_memory_id=mem_id)

        edges = self.repo.get_edges_for_memory(mem_id)
        assert len(edges) == 2

    def test_multiple_edges_between_same_entities(self):
        """Different rel_type or provenance should create separate edges."""
        src, dst = self._setup_two_entities()
        m1, m2 = str(uuid.uuid4()), str(uuid.uuid4())
        self.repo.add_edge("ws1", src, "manages", dst, provenance_memory_id=m1)
        self.repo.add_edge("ws1", src, "funds", dst, provenance_memory_id=m2)

        edges = self.repo.get_edges_from_entity("ws1", src)
        assert len(edges) == 2
        rel_types = {e.rel_type for e in edges}
        assert rel_types == {"manages", "funds"}


class TestKGFactCRUD:
    """Test consolidated fact operations."""

    def setup_method(self):
        self.db_path = _make_temp_db()
        self.repo = GraphRepository(self.db_path)

    def teardown_method(self):
        _cleanup_db(self.db_path)

    def test_create_fact_basic(self):
        fact_id = self.repo.create_fact(
            "ws1", "Team Alpha manages Project X", confidence=0.85,
        )
        assert isinstance(fact_id, str)
        assert len(fact_id) == 36  # UUID format

    def test_create_fact_with_json(self):
        summary = json.dumps({
            "subject": "Team Alpha",
            "rel": "manages",
            "object": "Project X",
        })
        fact_id = self.repo.create_fact(
            "ws1", "Team Alpha manages Project X",
            summary_json=summary, confidence=0.9,
        )
        facts = self.repo.get_facts_for_workspace("ws1")
        assert len(facts) == 1
        assert facts[0].id == fact_id
        parsed = json.loads(facts[0].summary_json)
        assert parsed["subject"] == "Team Alpha"

    def test_create_fact_empty_text_raises(self):
        try:
            self.repo.create_fact("ws1", "")
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_get_facts_for_workspace(self):
        self.repo.create_fact("ws1", "Fact 1")
        self.repo.create_fact("ws1", "Fact 2")
        self.repo.create_fact("ws2", "Fact 3")

        facts = self.repo.get_facts_for_workspace("ws1")
        assert len(facts) == 2


class TestGraphStats:
    """Test diagnostics."""

    def setup_method(self):
        self.db_path = _make_temp_db()
        self.repo = GraphRepository(self.db_path)

    def teardown_method(self):
        _cleanup_db(self.db_path)

    def test_stats_empty(self):
        stats = self.repo.get_graph_stats("ws1")
        assert stats["entity_count"] == 0
        assert stats["edge_count"] == 0
        assert stats["mention_count"] == 0
        assert stats["kg_fact_count"] == 0

    def test_stats_populated(self):
        src = self.repo.upsert_entity("ws1", "A")
        dst = self.repo.upsert_entity("ws1", "B")
        mem_id = str(uuid.uuid4())
        self.repo.add_mention(mem_id, src, role_hint="subject")
        self.repo.add_edge("ws1", src, "knows", dst, provenance_memory_id=mem_id)
        self.repo.create_fact("ws1", "A knows B")

        stats = self.repo.get_graph_stats("ws1")
        assert stats["entity_count"] == 2
        assert stats["mention_count"] == 1
        assert stats["edge_count"] == 1
        assert stats["kg_fact_count"] == 1


class TestEndToEndScenario:
    """Simulate a realistic graph construction from a memory."""

    def setup_method(self):
        self.db_path = _make_temp_db()
        self.repo = GraphRepository(self.db_path)

    def teardown_method(self):
        _cleanup_db(self.db_path)

    def test_two_hop_graph_construction(self):
        """
        Scenario: Build graph from two memories and verify 2-hop traversal.

        Memory 1: "Team Alpha manages Project X"
          → entities: Team Alpha (TEAM), Project X (PROJECT)
          → edge: Team Alpha --manages--> Project X

        Memory 2: "Team Alpha is located in Building C"
          → entities: Team Alpha (TEAM), Building C (LOCATION)
          → edge: Team Alpha --located_in--> Building C

        Query: "Where is Project X?"
          → Project X → (inbound edge) → Team Alpha → (outbound edge) → Building C

        This test verifies the STORAGE layer supports this traversal pattern.
        The actual graph propagation will be in a separate module.
        """
        ws = "ws1"
        mem1 = str(uuid.uuid4())
        mem2 = str(uuid.uuid4())

        # Memory 1: Team Alpha manages Project X
        team_id = self.repo.upsert_entity(ws, "Team Alpha", entity_type="TEAM")
        proj_id = self.repo.upsert_entity(ws, "Project X", entity_type="PROJECT")
        self.repo.add_mention(mem1, team_id, role_hint="subject")
        self.repo.add_mention(mem1, proj_id, role_hint="object")
        self.repo.add_edge(
            ws, team_id, "manages", proj_id,
            provenance_memory_id=mem1, confidence=0.9,
        )

        # Memory 2: Team Alpha is located in Building C
        # Team Alpha already exists — upsert should return same id
        team_id_2 = self.repo.upsert_entity(ws, "Team Alpha", entity_type="TEAM")
        assert team_id_2 == team_id, "Upsert should return same entity"

        bldg_id = self.repo.upsert_entity(ws, "Building C", entity_type="LOCATION")
        self.repo.add_mention(mem2, team_id, role_hint="subject")
        self.repo.add_mention(mem2, bldg_id, role_hint="object")
        self.repo.add_edge(
            ws, team_id, "located_in", bldg_id,
            provenance_memory_id=mem2, confidence=0.85,
        )

        # Traversal: Project X → inbound edges → Team Alpha
        inbound = self.repo.get_edges_to_entity(ws, proj_id)
        assert len(inbound) == 1
        assert inbound[0].src_entity_id == team_id

        # Traversal: Team Alpha → outbound edges → Building C
        outbound = self.repo.get_edges_from_entity(ws, team_id)
        assert len(outbound) == 2
        rel_types = {e.rel_type for e in outbound}
        assert "located_in" in rel_types

        # Find Building C via 2-hop
        location_edges = [e for e in outbound if e.rel_type == "located_in"]
        assert len(location_edges) == 1
        bldg_entity = self.repo.get_entity(location_edges[0].dst_entity_id)
        assert bldg_entity.canonical_text == "Building C"

        # Stats
        stats = self.repo.get_graph_stats(ws)
        assert stats["entity_count"] == 3
        assert stats["edge_count"] == 2
        assert stats["mention_count"] == 4

        print(f"\n  2-hop traversal verified:")
        print(f"    Project X <--[manages]-- Team Alpha --[located_in]--> Building C")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_standalone():
    """Run all test classes without pytest."""
    all_classes = [
        TestNormalizeEntityText,
        TestEntityCRUD,
        TestMentionCRUD,
        TestEdgeCRUD,
        TestKGFactCRUD,
        TestGraphStats,
        TestEndToEndScenario,
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
                total_failed += 1
            finally:
                if hasattr(instance, "teardown_method"):
                    instance.teardown_method()

    print(f"\n{'='*50}")
    print(f"Results: {total_passed} passed, {total_failed} failed, "
          f"{total_passed + total_failed} total")

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    print("GraphRepository Unit Tests")
    print("=" * 50)
    run_standalone()
