"""Phase 3 tests: Sparse Selective Restore + cache invalidation."""

from __future__ import annotations

from datetime import datetime

import pytest

from neural_memory.cache.models import ActivationCache, CachedState


class _FakeNeuron:
    """Minimal neuron stub exposing only metadata for selector tests."""

    def __init__(self, nid: str, embedding: list[float] | None = None) -> None:
        self.id = nid
        self.metadata: dict[str, object] = {}
        if embedding is not None:
            self.metadata["_embedding"] = embedding


class _FakeStorage:
    """In-memory storage stub — returns neurons with embeddings."""

    def __init__(self, neurons: dict[str, _FakeNeuron]) -> None:
        self._neurons = neurons

    async def get_neuron(self, nid: str):  # type: ignore[no-untyped-def]
        return self._neurons.get(nid)


class _FakeProvider:
    """Deterministic embedding provider: maps any query to a fixed vector."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    async def embed(self, text: str) -> list[float]:
        return list(self._vec)


class TestSelector:
    """Tests for SSC-lite sparse selective restore."""

    @pytest.mark.asyncio
    async def test_empty_cached_states(self) -> None:
        from neural_memory.cache.selector import select_relevant

        result = await select_relevant(
            cached_states=[],
            query="anything",
            embedding_provider=_FakeProvider([1.0, 0.0]),
            storage=_FakeStorage({}),
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_falls_back_when_provider_none(self) -> None:
        from neural_memory.cache.selector import select_relevant

        states = [CachedState(neuron_id=f"n{i}", activation_level=0.1 * i) for i in range(5)]
        result = await select_relevant(
            cached_states=states,
            query="q",
            embedding_provider=None,
            storage=_FakeStorage({}),
            top_k=3,
        )
        assert len(result) == 3
        assert [s.neuron_id for s in result] == ["n4", "n3", "n2"]

    @pytest.mark.asyncio
    async def test_ranks_by_cosine(self) -> None:
        from neural_memory.cache.selector import select_relevant

        states = [
            CachedState(neuron_id="close", activation_level=0.2),
            CachedState(neuron_id="orth", activation_level=0.9),
            CachedState(neuron_id="far", activation_level=0.5),
        ]
        storage = _FakeStorage(
            {
                "close": _FakeNeuron("close", [1.0, 0.0]),
                "orth": _FakeNeuron("orth", [0.0, 1.0]),
                "far": _FakeNeuron("far", [-1.0, 0.0]),
            }
        )
        result = await select_relevant(
            cached_states=states,
            query="q",
            embedding_provider=_FakeProvider([1.0, 0.0]),
            storage=storage,
            top_k=2,
            min_similarity=-1.0,
        )
        assert result[0].neuron_id == "close"

    @pytest.mark.asyncio
    async def test_missing_embeddings_fall_back(self) -> None:
        from neural_memory.cache.selector import select_relevant

        states = [CachedState(neuron_id="x", activation_level=0.7)]
        storage = _FakeStorage({"x": _FakeNeuron("x", None)})
        result = await select_relevant(
            cached_states=states,
            query="q",
            embedding_provider=_FakeProvider([1.0, 0.0]),
            storage=storage,
        )
        assert len(result) == 1
        assert result[0].neuron_id == "x"

    @pytest.mark.asyncio
    async def test_dimension_mismatch_falls_back(self, caplog) -> None:
        """All cached embeddings have wrong dim → warn + activation fallback."""
        import logging

        from neural_memory.cache.selector import select_relevant

        states = [
            CachedState(neuron_id="a", activation_level=0.3),
            CachedState(neuron_id="b", activation_level=0.9),
        ]
        storage = _FakeStorage(
            {
                "a": _FakeNeuron("a", [1.0, 0.0, 0.0]),  # 3-dim
                "b": _FakeNeuron("b", [0.0, 1.0, 0.0]),  # 3-dim
            }
        )
        provider = _FakeProvider([1.0, 0.0])  # 2-dim query
        with caplog.at_level(logging.WARNING, logger="neural_memory.cache.selector"):
            result = await select_relevant(
                cached_states=states,
                query="q",
                embedding_provider=provider,
                storage=storage,
                top_k=2,
                min_similarity=-1.0,
            )
        assert any("dim != query dim" in r.message for r in caplog.records)
        # Fallback ranks by activation → b first
        assert result[0].neuron_id == "b"

    @pytest.mark.asyncio
    async def test_empty_query_uses_fallback(self) -> None:
        from neural_memory.cache.selector import select_relevant

        states = [
            CachedState(neuron_id="a", activation_level=0.1),
            CachedState(neuron_id="b", activation_level=0.9),
        ]
        result = await select_relevant(
            cached_states=states,
            query="",
            embedding_provider=_FakeProvider([1.0, 0.0]),
            storage=_FakeStorage({}),
            top_k=1,
        )
        assert result[0].neuron_id == "b"

    def test_warm_activations_from_states(self) -> None:
        from neural_memory.cache.selector import warm_activations_from_states

        states = [
            CachedState(neuron_id="a", activation_level=0.5),
            CachedState(neuron_id="b", activation_level=0.9),
        ]
        warm = warm_activations_from_states(states)
        assert warm == {"a": 0.5, "b": 0.9}


class TestInvalidation:
    """Tests for partial/full cache invalidation logic."""

    def test_tracker_records_changes(self) -> None:
        from neural_memory.cache.invalidation import InvalidationTracker

        tracker = InvalidationTracker()
        tracker.record_change("neuron_add", "n1")
        tracker.record_change("neuron_delete", "n2")
        assert tracker.dirty_neurons == {"n1", "n2"}
        assert not tracker.consolidation_triggered

    def test_tracker_consolidation(self) -> None:
        from neural_memory.cache.invalidation import InvalidationTracker

        tracker = InvalidationTracker()
        tracker.record_change("consolidation")
        assert tracker.consolidation_triggered
        assert tracker.dirty_neurons == set()

    def test_tracker_bulk(self) -> None:
        from neural_memory.cache.invalidation import InvalidationTracker

        tracker = InvalidationTracker()
        tracker.record_bulk("neuron_delete", ["n1", "n2", "n3"])
        assert tracker.dirty_neurons == {"n1", "n2", "n3"}
        assert tracker.change_counts["neuron_delete"] == 3

    def test_tracker_bulk_empty(self) -> None:
        from neural_memory.cache.invalidation import InvalidationTracker

        tracker = InvalidationTracker()
        tracker.record_bulk("neuron_delete", [])
        assert tracker.is_empty()

    def test_tracker_reset(self) -> None:
        from neural_memory.cache.invalidation import InvalidationTracker

        tracker = InvalidationTracker()
        tracker.record_change("neuron_add", "n1")
        tracker.record_change("consolidation")
        tracker.reset()
        assert tracker.is_empty()

    def test_tracker_summary(self) -> None:
        from neural_memory.cache.invalidation import InvalidationTracker

        tracker = InvalidationTracker()
        tracker.record_change("neuron_add", "n1")
        tracker.record_change("neuron_add", "n2")
        summary = tracker.summary()
        assert summary["dirty_neurons"] == 2
        assert summary["count_neuron_add"] == 2

    def test_remove_neurons_preserves_immutability(self) -> None:
        from neural_memory.cache.invalidation import remove_neurons

        now = datetime(2026, 1, 1, 12, 0, 0)
        entries = tuple(CachedState(neuron_id=f"n{i}", activation_level=0.5) for i in range(3))
        cache = ActivationCache(brain_id="b", brain_name="t", cached_at=now, entries=entries)
        new_cache = remove_neurons(cache, {"n1"})
        assert cache.entry_count == 3
        assert new_cache.entry_count == 2
        assert all(e.neuron_id != "n1" for e in new_cache.entries)

    def test_remove_neurons_noop_when_absent(self) -> None:
        from neural_memory.cache.invalidation import remove_neurons

        now = datetime(2026, 1, 1, 12, 0, 0)
        entries = (CachedState(neuron_id="a", activation_level=0.5),)
        cache = ActivationCache(brain_id="b", brain_name="t", cached_at=now, entries=entries)
        assert remove_neurons(cache, {"zzz"}) is cache

    def test_remove_neurons_empty_input(self) -> None:
        from neural_memory.cache.invalidation import remove_neurons

        now = datetime(2026, 1, 1, 12, 0, 0)
        entries = (CachedState(neuron_id="a", activation_level=0.5),)
        cache = ActivationCache(brain_id="b", brain_name="t", cached_at=now, entries=entries)
        assert remove_neurons(cache, set()) is cache

    def test_should_fully_invalidate_on_consolidation(self) -> None:
        from neural_memory.cache.invalidation import (
            InvalidationTracker,
            should_fully_invalidate,
        )

        now = datetime(2026, 1, 1, 12, 0, 0)
        tracker = InvalidationTracker()
        tracker.record_change("consolidation")
        cache = ActivationCache(brain_id="b", brain_name="t", cached_at=now)
        assert should_fully_invalidate(tracker, cache)

    def test_should_fully_invalidate_on_ratio(self) -> None:
        from neural_memory.cache.invalidation import (
            InvalidationTracker,
            should_fully_invalidate,
        )

        now = datetime(2026, 1, 1, 12, 0, 0)
        entries = tuple(CachedState(neuron_id=f"n{i}", activation_level=0.5) for i in range(4))
        cache = ActivationCache(brain_id="b", brain_name="t", cached_at=now, entries=entries)
        tracker = InvalidationTracker()
        tracker.record_bulk("neuron_delete", ["n0", "n1"])
        assert should_fully_invalidate(tracker, cache)

    def test_should_not_invalidate_below_ratio(self) -> None:
        from neural_memory.cache.invalidation import (
            InvalidationTracker,
            should_fully_invalidate,
        )

        now = datetime(2026, 1, 1, 12, 0, 0)
        entries = tuple(CachedState(neuron_id=f"n{i}", activation_level=0.5) for i in range(10))
        cache = ActivationCache(brain_id="b", brain_name="t", cached_at=now, entries=entries)
        tracker = InvalidationTracker()
        tracker.record_change("neuron_delete", "n0")
        assert not should_fully_invalidate(tracker, cache)

    def test_should_not_invalidate_when_dirty_not_in_cache(self) -> None:
        from neural_memory.cache.invalidation import (
            InvalidationTracker,
            should_fully_invalidate,
        )

        now = datetime(2026, 1, 1, 12, 0, 0)
        entries = (CachedState(neuron_id="a", activation_level=0.5),)
        cache = ActivationCache(brain_id="b", brain_name="t", cached_at=now, entries=entries)
        tracker = InvalidationTracker()
        tracker.record_change("neuron_delete", "outsider")
        assert not should_fully_invalidate(tracker, cache)


class TestApplyInvalidation:
    """Tests for apply_invalidation orchestration against a real manager."""

    @pytest.mark.asyncio
    async def test_no_cache_reports_no_cache(self) -> None:
        from unittest.mock import MagicMock

        from neural_memory.cache.invalidation import InvalidationTracker, apply_invalidation
        from neural_memory.cache.manager import ActivationCacheManager

        mgr = ActivationCacheManager(MagicMock())
        tracker = InvalidationTracker()
        tracker.record_change("neuron_add", "n1")
        report = await apply_invalidation(mgr, tracker)
        assert report["action"] == "no_cache"
        assert tracker.is_empty()  # tracker reset

    @pytest.mark.asyncio
    async def test_no_changes_reports_no_changes(self) -> None:
        from unittest.mock import MagicMock

        from neural_memory.cache.invalidation import InvalidationTracker, apply_invalidation
        from neural_memory.cache.manager import ActivationCacheManager

        now = datetime(2026, 1, 1, 12, 0, 0)
        mgr = ActivationCacheManager(MagicMock())
        mgr._loaded_cache = ActivationCache(brain_id="b", brain_name="t", cached_at=now)
        report = await apply_invalidation(mgr, InvalidationTracker())
        assert report["action"] == "no_changes"

    @pytest.mark.asyncio
    async def test_partial_invalidation(self) -> None:
        from unittest.mock import MagicMock

        from neural_memory.cache.invalidation import InvalidationTracker, apply_invalidation
        from neural_memory.cache.manager import ActivationCacheManager

        now = datetime(2026, 1, 1, 12, 0, 0)
        entries = tuple(CachedState(neuron_id=f"n{i}", activation_level=0.5) for i in range(10))
        mgr = ActivationCacheManager(MagicMock())
        mgr._loaded_cache = ActivationCache(
            brain_id="b", brain_name="t", cached_at=now, entries=entries
        )
        tracker = InvalidationTracker()
        tracker.record_change("neuron_delete", "n0")
        report = await apply_invalidation(mgr, tracker)
        assert report["action"] == "partial"
        assert report["removed"] == 1
        assert mgr._loaded_cache is not None
        assert mgr._loaded_cache.entry_count == 9

    @pytest.mark.asyncio
    async def test_full_invalidation_on_consolidation(self, tmp_path) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from neural_memory.cache.invalidation import InvalidationTracker, apply_invalidation
        from neural_memory.cache.manager import ActivationCacheManager

        now = datetime(2026, 1, 1, 12, 0, 0)
        entries = (CachedState(neuron_id="a", activation_level=0.5),)
        storage = MagicMock()
        storage.brain_id = ""
        storage.get_brain = AsyncMock(return_value=None)
        mgr = ActivationCacheManager(storage, data_dir=tmp_path)
        mgr._loaded_cache = ActivationCache(
            brain_id="b", brain_name="t", cached_at=now, entries=entries
        )
        tracker = InvalidationTracker()
        tracker.record_change("consolidation")
        report = await apply_invalidation(mgr, tracker)
        assert report["action"] == "full"
        assert mgr._loaded_cache is None


class TestDetectStaleness:
    """Tests for hash-based staleness detection."""

    @pytest.mark.asyncio
    async def test_no_cache_is_not_stale(self) -> None:
        from unittest.mock import MagicMock

        from neural_memory.cache.invalidation import detect_staleness
        from neural_memory.cache.manager import ActivationCacheManager

        mgr = ActivationCacheManager(MagicMock())
        assert not await detect_staleness(mgr)

    @pytest.mark.asyncio
    async def test_missing_hash_is_not_stale(self) -> None:
        from unittest.mock import MagicMock

        from neural_memory.cache.invalidation import detect_staleness
        from neural_memory.cache.manager import ActivationCacheManager

        now = datetime(2026, 1, 1, 12, 0, 0)
        mgr = ActivationCacheManager(MagicMock())
        mgr._loaded_cache = ActivationCache(
            brain_id="b", brain_name="t", cached_at=now, brain_hash=""
        )
        assert not await detect_staleness(mgr)

    @pytest.mark.asyncio
    async def test_hash_mismatch_is_stale(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from neural_memory.cache.invalidation import detect_staleness
        from neural_memory.cache.manager import ActivationCacheManager

        now = datetime(2026, 1, 1, 12, 0, 0)
        mgr = ActivationCacheManager(MagicMock())
        mgr._loaded_cache = ActivationCache(
            brain_id="b", brain_name="t", cached_at=now, brain_hash="old"
        )
        mgr._compute_brain_hash = AsyncMock(return_value="new")  # type: ignore[method-assign]
        assert await detect_staleness(mgr)

    @pytest.mark.asyncio
    async def test_hash_match_not_stale(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from neural_memory.cache.invalidation import detect_staleness
        from neural_memory.cache.manager import ActivationCacheManager

        now = datetime(2026, 1, 1, 12, 0, 0)
        mgr = ActivationCacheManager(MagicMock())
        mgr._loaded_cache = ActivationCache(
            brain_id="b", brain_name="t", cached_at=now, brain_hash="same"
        )
        mgr._compute_brain_hash = AsyncMock(return_value="same")  # type: ignore[method-assign]
        assert not await detect_staleness(mgr)


class TestSelectiveWarmActivations:
    """Tests for ActivationCacheManager.get_warm_activations_selective."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_cache(self) -> None:
        from unittest.mock import MagicMock

        from neural_memory.cache.manager import ActivationCacheManager

        mgr = ActivationCacheManager(MagicMock())
        result = await mgr.get_warm_activations_selective(query="q", embedding_provider=None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_when_expired(self) -> None:
        from unittest.mock import MagicMock

        from neural_memory.cache.manager import ActivationCacheManager

        past = datetime(2024, 1, 1, 12, 0, 0)
        mgr = ActivationCacheManager(MagicMock())
        mgr._loaded_cache = ActivationCache(
            brain_id="b",
            brain_name="t",
            cached_at=past,
            ttl_hours=1,
            entries=(CachedState(neuron_id="a", activation_level=0.5),),
        )
        result = await mgr.get_warm_activations_selective(query="q", embedding_provider=None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_reflex_pipeline_exposes_embedding_provider(self) -> None:
        """Regression: recall_handler relies on pipeline._embedding_provider.

        If ReflexPipeline renames or drops the private attribute, warm-start
        SSC-lite silently degrades to activation-ranked fallback. Assert the
        attribute exists on a freshly constructed pipeline so upstream
        renames are caught at test time.
        """
        from neural_memory.core.brain import BrainConfig
        from neural_memory.engine.retrieval import ReflexPipeline

        class _Stub:
            brain_id = ""

        pipeline = ReflexPipeline(_Stub(), BrainConfig())
        assert hasattr(pipeline, "_embedding_provider")

    @pytest.mark.asyncio
    async def test_manager_clamps_top_k_and_similarity(self) -> None:
        """get_warm_activations_selective must clamp extreme arguments."""
        from unittest.mock import MagicMock

        from neural_memory.cache.manager import ActivationCacheManager
        from neural_memory.utils.timeutils import utcnow

        mgr = ActivationCacheManager(MagicMock(), max_entries=50)
        entries = tuple(CachedState(neuron_id=f"n{i}", activation_level=0.01 * i) for i in range(5))
        mgr._loaded_cache = ActivationCache(
            brain_id="b", brain_name="t", cached_at=utcnow(), entries=entries
        )
        # top_k=-5 clamps to 1, min_similarity=2.0 clamps to 1.0
        result = await mgr.get_warm_activations_selective(
            query="q", embedding_provider=None, top_k=-5, min_similarity=2.0
        )
        assert len(result) == 1
        # top_k=10_000 clamps to max_entries (50) — but only 5 entries exist
        result = await mgr.get_warm_activations_selective(
            query="q", embedding_provider=None, top_k=10_000
        )
        assert len(result) == 5

    @pytest.mark.asyncio
    async def test_selective_returns_top_k(self) -> None:
        from unittest.mock import MagicMock

        from neural_memory.cache.manager import ActivationCacheManager
        from neural_memory.utils.timeutils import utcnow

        mgr = ActivationCacheManager(MagicMock())
        entries = tuple(CachedState(neuron_id=f"n{i}", activation_level=0.1 * i) for i in range(5))
        mgr._loaded_cache = ActivationCache(
            brain_id="b", brain_name="t", cached_at=utcnow(), entries=entries
        )
        result = await mgr.get_warm_activations_selective(
            query="q", embedding_provider=None, top_k=2
        )
        # With no provider → fallback to activation-ranked top-K
        assert len(result) == 2
        assert "n4" in result
        assert "n3" in result
