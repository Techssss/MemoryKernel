"""Tests for Tantivy-based TextIndex in InfinityDB."""

from __future__ import annotations

import pytest

from neural_memory.pro.infinitydb.text_index import TextIndex, is_tantivy_available

pytestmark = pytest.mark.skipif(not is_tantivy_available(), reason="tantivy-py not installed")


@pytest.fixture
def text_index():
    """Create an in-memory TextIndex for testing."""
    idx = TextIndex()
    idx.open()
    yield idx
    idx.close()


@pytest.fixture
def text_index_with_data(text_index: TextIndex):
    """TextIndex pre-loaded with test documents."""
    text_index.add("n1", "Python supports async I/O natively")
    text_index.add("n2", "Neural memory uses spreading activation")
    text_index.add("n3", "Deep learning models require GPU training")
    text_index.add("n4", "Python asyncio event loop is single-threaded")
    text_index.add("n5", "Memory consolidation happens during sleep")
    text_index.commit()
    return text_index


class TestTextIndexLifecycle:
    def test_open_close(self):
        idx = TextIndex()
        assert not idx.is_open
        idx.open()
        assert idx.is_open
        idx.close()
        assert not idx.is_open

    def test_close_idempotent(self, text_index: TextIndex):
        text_index.close()
        text_index.close()  # should not raise

    def test_operations_before_open_are_noop(self):
        idx = TextIndex()
        idx.add("n1", "test")  # should not raise
        idx.commit()
        assert idx.count == 0
        assert idx.search("test") == []


class TestTextIndexAdd:
    def test_add_single(self, text_index: TextIndex):
        text_index.add("n1", "hello world")
        text_index.commit()
        assert text_index.count == 1

    def test_add_multiple(self, text_index: TextIndex):
        text_index.add("n1", "first document")
        text_index.add("n2", "second document")
        text_index.add("n3", "third document")
        text_index.commit()
        assert text_index.count == 3

    def test_add_batch(self, text_index: TextIndex):
        items = [("n1", "doc one"), ("n2", "doc two"), ("n3", "doc three")]
        text_index.add_batch(items)
        text_index.commit()
        assert text_index.count == 3

    def test_add_replaces_existing(self, text_index: TextIndex):
        text_index.add("n1", "original content")
        text_index.commit()
        text_index.add("n1", "updated content")
        text_index.commit()
        assert text_index.count == 1
        results = text_index.search("updated")
        assert len(results) == 1
        assert results[0][0] == "n1"
        # Original content should not be searchable
        assert text_index.search("original") == []


class TestTextIndexSearch:
    def test_search_basic(self, text_index_with_data: TextIndex):
        results = text_index_with_data.search("Python")
        assert len(results) >= 1
        # Results are (neuron_id, score) tuples
        nids = [nid for nid, _ in results]
        assert "n1" in nids or "n4" in nids

    def test_search_returns_scores(self, text_index_with_data: TextIndex):
        results = text_index_with_data.search("neural memory")
        assert len(results) >= 1
        for nid, score in results:
            assert isinstance(nid, str)
            assert isinstance(score, float)
            assert score > 0

    def test_search_respects_limit(self, text_index_with_data: TextIndex):
        results = text_index_with_data.search("Python", limit=1)
        assert len(results) <= 1

    def test_search_no_results(self, text_index_with_data: TextIndex):
        results = text_index_with_data.search("nonexistent_xyz_term")
        assert results == []

    def test_search_empty_index(self, text_index: TextIndex):
        results = text_index.search("anything")
        assert results == []

    def test_search_contains(self, text_index_with_data: TextIndex):
        results = text_index_with_data.search_contains("async")
        assert isinstance(results, list)
        assert all(isinstance(nid, str) for nid in results)
        assert "n1" in results or "n4" in results

    def test_search_auto_commits_pending(self, text_index: TextIndex):
        text_index.add("n1", "uncommitted document")
        # Don't call commit() — search should auto-commit
        results = text_index.search("uncommitted")
        assert len(results) == 1
        assert results[0][0] == "n1"


class TestTextIndexDelete:
    def test_delete(self, text_index_with_data: TextIndex):
        text_index_with_data.delete("n1")
        text_index_with_data.commit()
        assert text_index_with_data.count == 4
        # n1's content should not be found
        nids = [nid for nid, _ in text_index_with_data.search("async I/O")]
        assert "n1" not in nids

    def test_delete_nonexistent(self, text_index_with_data: TextIndex):
        text_index_with_data.delete("nonexistent")
        text_index_with_data.commit()
        # Count should be unchanged
        assert text_index_with_data.count == 5


class TestTextIndexCount:
    def test_count_empty(self, text_index: TextIndex):
        assert text_index.count == 0

    def test_count_after_add(self, text_index_with_data: TextIndex):
        assert text_index_with_data.count == 5

    def test_count_closed_index(self):
        idx = TextIndex()
        assert idx.count == 0


class TestTextIndexDiskBacked:
    def test_persistent_index(self, tmp_path):
        fts_path = tmp_path / "fts"
        idx = TextIndex(path=fts_path)
        idx.open()
        idx.add("n1", "persistent data")
        idx.commit()
        idx.close()

        # Reopen and verify data persists
        idx2 = TextIndex(path=fts_path)
        idx2.open()
        assert idx2.count == 1
        results = idx2.search("persistent")
        assert len(results) == 1
        assert results[0][0] == "n1"
        idx2.close()


class TestTextIndexDirtyTracking:
    def test_not_dirty_after_open(self, text_index: TextIndex):
        assert not text_index._dirty

    def test_dirty_after_add(self, text_index: TextIndex):
        text_index.add("n1", "test")
        assert text_index._dirty

    def test_not_dirty_after_commit(self, text_index: TextIndex):
        text_index.add("n1", "test")
        text_index.commit()
        assert not text_index._dirty

    def test_dirty_after_delete(self, text_index_with_data: TextIndex):
        text_index_with_data.delete("n1")
        assert text_index_with_data._dirty

    def test_close_auto_commits(self, text_index: TextIndex):
        text_index.add("n1", "auto commit on close")
        assert text_index._dirty
        text_index.close()
        # Reopen in-memory won't have data, but close should not error
