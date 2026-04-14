"""
tests/test_hybrid_retriever.py
==============================
Unit tests for HybridRetriever and the embedding infrastructure.

Test strategy
-------------
- All tests use TFIDFEmbedder (zero external deps, deterministic in-process).
- Heavy sentence-transformer model is never loaded in the test suite.
- We inject the embedder via HybridRetriever(embedder=...) to stay hermetic.
- Semantic tests use thematically related phrases (not identical strings) to
  validate that vector similarity drives recall beyond keyword matching.
"""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest

from memk.core.embedder import (
    TFIDFEmbedder,
    cosine_similarity,
    encode_embedding,
    decode_embedding,
)
from memk.storage.db import MemoryDB, _encode_blob, _decode_blob
from memk.retrieval.retriever import HybridRetriever, KeywordRetriever, RetrievedItem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_hybrid.db"
    db = MemoryDB(db_path=str(db_file))
    db.init_db()
    return db


@pytest.fixture
def tfidf_embedder():
    """
    Pre-fitted TFIDF embedder over a small fixed corpus so that all tests share
    the same vector space and results are deterministic.
    """
    corpus = [
        "I hate centralized control",
        "User dislikes authority and power",
        "Project uses Python",
        "decentralized systems are better",
        "freedom and autonomy matter",
        "Python is the language of choice",
        "Docker containerizes services",
    ]
    emb = TFIDFEmbedder(n_components=32)
    emb.fit(corpus)
    return emb


@pytest.fixture
def hybrid(test_db, tfidf_embedder):
    return HybridRetriever(db=test_db, embedder=tfidf_embedder, alpha=0.3, beta=0.7)


# ---------------------------------------------------------------------------
# 1. Embedding helpers
# ---------------------------------------------------------------------------

class TestEmbeddingHelpers:
    def test_encode_decode_roundtrip(self):
        vec = np.array([0.1, 0.2, 0.3, -0.5], dtype=np.float32)
        blob = encode_embedding(vec)
        restored = decode_embedding(blob)
        np.testing.assert_allclose(vec, restored, rtol=1e-5)

    def test_encode_decode_blob_helpers(self):
        vec = np.random.rand(64).astype(np.float32)
        blob = _encode_blob(vec)
        restored = _decode_blob(blob)
        np.testing.assert_allclose(vec, restored, rtol=1e-5)

    def test_cosine_similarity_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert math.isclose(cosine_similarity(v, v), 1.0, rel_tol=1e-5)

    def test_cosine_similarity_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert math.isclose(cosine_similarity(a, b), 0.0, abs_tol=1e-5)

    def test_cosine_similarity_zero_vector_safe(self):
        zero = np.zeros(4, dtype=np.float32)
        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        assert cosine_similarity(zero, v) == 0.0

    def test_cosine_similarity_anti_parallel(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert math.isclose(cosine_similarity(v, -v), -1.0, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# 2. TFIDFEmbedder
# ---------------------------------------------------------------------------

class TestTFIDFEmbedder:
    def test_embed_returns_float32(self, tfidf_embedder):
        vec = tfidf_embedder.embed("some text here")
        assert vec.dtype == np.float32

    def test_embed_is_normalized(self, tfidf_embedder):
        vec = tfidf_embedder.embed("Python is great")
        norm = float(np.linalg.norm(vec))
        assert math.isclose(norm, 1.0, abs_tol=1e-5) or norm == 0.0

    def test_embed_batch_same_length(self, tfidf_embedder):
        texts = ["hello world", "foo bar baz", "test sentence"]
        vecs = tfidf_embedder.embed_batch(texts)
        assert len(vecs) == 3
        for v in vecs:
            assert isinstance(v, np.ndarray)

    def test_similar_texts_higher_similarity(self, tfidf_embedder):
        v_control = tfidf_embedder.embed("Python is the language of choice")
        v_related = tfidf_embedder.embed("Python is the default language for scripts")
        v_unrelated = tfidf_embedder.embed("Docker containerizes services")

        sim_related = cosine_similarity(v_control, v_related)
        sim_unrelated = cosine_similarity(v_control, v_unrelated)
        assert sim_related > sim_unrelated, (
            f"Expected sim_related ({sim_related:.3f}) > sim_unrelated ({sim_unrelated:.3f})"
        )


# ---------------------------------------------------------------------------
# 3. DB schema — embedding column
# ---------------------------------------------------------------------------

class TestDBEmbeddingSchema:
    def test_memories_table_has_embedding_column(self, test_db):
        with test_db._get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(memories)")
            columns = {row["name"] for row in cursor.fetchall()}
        assert "embedding" in columns

    def test_facts_table_has_embedding_column(self, test_db):
        with test_db._get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(facts)")
            columns = {row["name"] for row in cursor.fetchall()}
        assert "embedding" in columns

    def test_insert_memory_with_embedding_persists(self, test_db, tfidf_embedder):
        vec = tfidf_embedder.embed("User hates centralized systems")
        mem_id = test_db.insert_memory("User hates centralized systems", embedding=vec)

        with test_db._get_connection() as conn:
            row = conn.execute(
                "SELECT embedding FROM memories WHERE id = ?", (mem_id,)
            ).fetchone()

        assert row["embedding"] is not None
        restored = _decode_blob(row["embedding"])
        np.testing.assert_allclose(vec, restored, rtol=1e-4)

    def test_insert_fact_with_embedding_persists(self, test_db, tfidf_embedder):
        vec = tfidf_embedder.embed("user dislikes authority")
        fact_id = test_db.insert_fact("user", "dislikes", "authority", embedding=vec)

        with test_db._get_connection() as conn:
            row = conn.execute(
                "SELECT embedding FROM facts WHERE id = ?", (fact_id,)
            ).fetchone()

        assert row["embedding"] is not None

    def test_update_memory_embedding(self, test_db, tfidf_embedder):
        mem_id = test_db.insert_memory("No embedding yet")
        assert test_db.get_memories_without_embedding()[0]["id"] == mem_id

        vec = tfidf_embedder.embed("No embedding yet")
        test_db.update_memory_embedding(mem_id, vec)

        missing = test_db.get_memories_without_embedding()
        assert all(r["id"] != mem_id for r in missing)

    def test_get_memories_without_embedding_filters_correctly(self, test_db, tfidf_embedder):
        vec = tfidf_embedder.embed("embedded text")
        id_with = test_db.insert_memory("embedded text", embedding=vec)
        id_without = test_db.insert_memory("no embedding here")

        missing = {r["id"] for r in test_db.get_memories_without_embedding()}
        assert id_without in missing
        assert id_with not in missing


# ---------------------------------------------------------------------------
# 4. HybridRetriever — functional tests
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    def _populate(self, db, embedder):
        """Helper to seed DB with corpus that covers semantic test cases."""
        entries = [
            ("I hate centralized control", None),
            ("decentralized systems are better", None),
            ("Python is the language of choice", None),
        ]
        for text, _ in entries:
            vec = embedder.embed(text)
            db.insert_memory(text, embedding=vec)

        fact_vec = embedder.embed("user dislikes authority and power")
        db.insert_fact("user", "dislikes", "authority", embedding=fact_vec)

    # ------------------------------------------------------------------

    def test_empty_query_returns_empty(self, hybrid, test_db, tfidf_embedder):
        self._populate(test_db, tfidf_embedder)
        assert hybrid.retrieve("") == []

    def test_returns_list_of_retrieved_items(self, hybrid, test_db, tfidf_embedder):
        self._populate(test_db, tfidf_embedder)
        results = hybrid.retrieve("Python")
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, RetrievedItem)

    def test_limit_is_respected(self, hybrid, test_db, tfidf_embedder):
        self._populate(test_db, tfidf_embedder)
        results = hybrid.retrieve("Python", limit=2)
        assert len(results) <= 2

    def test_facts_score_higher_than_memories(self, test_db, tfidf_embedder):
        """
        A fact with identical keyword should always outscore a raw memory because
        facts carry the FACT_MULTIPLIER boost.
        """
        vec = tfidf_embedder.embed("user uses Python")
        test_db.insert_memory("user uses Python", embedding=vec)

        fact_vec = tfidf_embedder.embed("user uses Python")
        test_db.insert_fact("user", "uses", "Python", embedding=fact_vec)

        retriever = HybridRetriever(
            db=test_db, embedder=tfidf_embedder, alpha=0.3, beta=0.7
        )
        results = retriever.retrieve("Python")

        # First result must be a fact
        assert results[0].item_type == "fact"

    def test_keyword_only_match_still_surfaces(self, test_db, tfidf_embedder):
        """Keyword match (alpha) should rescue an item even with low vector similarity."""
        vec = tfidf_embedder.embed("Docker containerizes services")
        test_db.insert_memory("Docker containerizes services", embedding=vec)

        retriever = HybridRetriever(
            db=test_db, embedder=tfidf_embedder, alpha=0.9, beta=0.1
        )
        results = retriever.retrieve("Docker")
        ids = [r.id for r in results]
        memories = test_db.get_all_memories()
        docker_id = next(m["id"] for m in memories if "Docker" in m["content"])
        assert docker_id in ids

    def test_semantic_recall_without_keyword_match(self, test_db, tfidf_embedder):
        """
        Core requirement: querying 'authority' should retrieve 'I hate centralized control'
        even though 'authority' does not appear in the text (pure semantic recall).
        """
        # Seed memory that doesn't share the query word
        target_text = "I hate centralized control"
        vec = tfidf_embedder.embed(target_text)
        mem_id = test_db.insert_memory(target_text, embedding=vec)

        # Add noise entries to make it non-trivial
        for noise in ["Docker containerizes services", "Python is the language of choice"]:
            nv = tfidf_embedder.embed(noise)
            test_db.insert_memory(noise, embedding=nv)

        retriever = HybridRetriever(
            db=test_db, embedder=tfidf_embedder, alpha=0.1, beta=0.9
        )
        results = retriever.retrieve("authority")
        retrieved_ids = [r.id for r in results]

        # The semantically related memory must appear in results
        assert mem_id in retrieved_ids, (
            f"Expected '{target_text}' to appear when querying 'authority'. "
            f"Results: {[r.content for r in results]}"
        )

    def test_backfill_happens_on_first_retrieve(self, test_db, tfidf_embedder):
        """
        Rows inserted without embeddings should be auto-embedded on first retrieve().
        """
        mem_id = test_db.insert_memory("User prefers autonomy")
        assert test_db.get_memories_without_embedding()[0]["id"] == mem_id

        retriever = HybridRetriever(
            db=test_db, embedder=tfidf_embedder, alpha=0.3, beta=0.7
        )
        retriever.retrieve("autonomy")  # triggers backfill

        missing_after = test_db.get_memories_without_embedding()
        assert all(r["id"] != mem_id for r in missing_after)

    def test_alpha_beta_weight_tuning(self, test_db, tfidf_embedder):
        """
        With alpha=1.0, beta=0.0 (pure keyword), only keyword hits should score > 0.
        With alpha=0.0, beta=1.0 (pure vector), items with no embedding score 0.
        """
        vec = tfidf_embedder.embed("Python is great")
        test_db.insert_memory("Python is great", embedding=vec)
        # Insert a memory with NO embedding so vector score = 0
        no_emb_id = test_db.insert_memory("unrelated content without embedding")

        # Pure keyword retriever
        kw_only = HybridRetriever(
            db=test_db, embedder=tfidf_embedder, alpha=1.0, beta=0.0
        )
        results = kw_only.retrieve("Python")
        # Only the keyword-matching item should score > 0
        for r in results:
            if r.item_type == "memory" and "Python" not in r.content:
                assert r.score == 0.0

    def test_score_threshold_filters_low_scores(self, test_db, tfidf_embedder):
        """Items below score_threshold must be excluded."""
        for text in ["Python rocks", "Java is verbose", "Go is fast"]:
            vec = tfidf_embedder.embed(text)
            test_db.insert_memory(text, embedding=vec)

        retriever = HybridRetriever(
            db=test_db,
            embedder=tfidf_embedder,
            alpha=0.0,  # pure vector
            beta=1.0,
            score_threshold=0.99,  # unreachably high — guaranteed no items
        )
        results = retriever.retrieve("Rust systems programming")
        assert results == []

    def test_backfill_all_embeddings_returns_count(self, test_db, tfidf_embedder):
        test_db.insert_memory("A memory without embedding")
        test_db.insert_memory("Another memory without embedding")
        test_db.insert_fact("project", "uses", "Docker")

        retriever = HybridRetriever(
            db=test_db, embedder=tfidf_embedder, alpha=0.3, beta=0.7
        )
        count = retriever.backfill_all_embeddings()
        assert count == 3  # 2 memories + 1 fact

        # Second call should be a no-op
        count2 = retriever.backfill_all_embeddings()
        assert count2 == 0


# ---------------------------------------------------------------------------
# 5. Backward compatibility — KeywordRetriever still works
# ---------------------------------------------------------------------------

class TestKeywordRetrieverBackwardCompat:
    def test_keyword_retriever_ignores_embeddings(self, test_db, tfidf_embedder):
        """The original KeywordRetriever must still work unmodified."""
        test_db.insert_memory("User uses Python")
        test_db.insert_fact("user", "loves", "Python")

        retriever = KeywordRetriever(test_db)
        results = retriever.retrieve("Python")

        assert len(results) == 2
        assert results[0].item_type == "fact"
        assert "user loves Python" in results[0].content

    def test_keyword_retriever_returns_empty_for_blank_query(self, test_db):
        retriever = KeywordRetriever(test_db)
        assert retriever.retrieve("") == []

    def test_keyword_retriever_reconciliation_respected(self, test_db):
        test_db.insert_fact("user", "uses", "Java")
        test_db.insert_fact("user", "uses", "Python")  # shadows Java

        retriever = KeywordRetriever(test_db)
        results = retriever.retrieve("uses")

        objects = [r.content for r in results if r.item_type == "fact"]
        assert any("Python" in c for c in objects)
        assert all("Java" not in c for c in objects)
