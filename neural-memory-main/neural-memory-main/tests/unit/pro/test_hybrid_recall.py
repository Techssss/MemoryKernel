"""Tests for HNSW hybrid recall strategy."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.engine.activation import ActivationResult
from neural_memory.pro.retrieval.hybrid_recall import (
    FALLBACK_THRESHOLD,
    hnsw_hybrid_recall,
)


def _make_hnsw_results(n: int) -> list[tuple[str, float]]:
    """Create n fake HNSW results with decreasing similarity."""
    return [(f"neuron-{i}", 1.0 - i * 0.01) for i in range(n)]


def _make_storage(
    hnsw_results: list[tuple[str, float]], bm25_results: list[tuple[str, float]] | None = None
) -> Any:
    """Create a mock storage with knn_search and optional text_search."""
    storage = AsyncMock()
    storage.knn_search = AsyncMock(return_value=hnsw_results)
    if bm25_results is not None:
        storage.text_search = AsyncMock(return_value=bm25_results)
    return storage


def _make_activator(activations: dict[str, ActivationResult] | None = None) -> Any:
    """Create a mock activator."""
    if activations is None:
        activations = {
            "neuron-0": ActivationResult(
                neuron_id="neuron-0",
                activation_level=0.9,
                hop_distance=0,
                path=["neuron-0"],
                source_anchor="neuron-0",
            ),
            "neuron-1": ActivationResult(
                neuron_id="neuron-1",
                activation_level=0.7,
                hop_distance=1,
                path=["neuron-0", "neuron-1"],
                source_anchor="neuron-0",
            ),
        }
    activator = MagicMock()
    activator.activate_from_multiple = AsyncMock(return_value=(activations, ["neuron-0"]))
    return activator


class TestHybridRecall:
    """Tests for hnsw_hybrid_recall function."""

    @pytest.mark.asyncio
    async def test_returns_none_without_knn_search(self) -> None:
        storage = AsyncMock(spec=[])  # No knn_search attr
        activator = _make_activator()
        result = await hnsw_hybrid_recall(
            [0.1] * 384,
            storage,
            activator,
            [["a-1"]],
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_below_threshold(self) -> None:
        storage = _make_storage(_make_hnsw_results(FALLBACK_THRESHOLD - 1))
        activator = _make_activator()
        result = await hnsw_hybrid_recall(
            [0.1] * 384,
            storage,
            activator,
            [["a-1"]],
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_activations_above_threshold(self) -> None:
        storage = _make_storage(_make_hnsw_results(20))
        activator = _make_activator()
        result = await hnsw_hybrid_recall(
            [0.1] * 384,
            storage,
            activator,
            [["a-1"]],
        )
        assert result is not None
        activations, intersections, scope = result
        assert "neuron-0" in activations
        assert len(scope) >= 20  # At least HNSW results in scope

    @pytest.mark.asyncio
    async def test_scope_includes_anchors(self) -> None:
        storage = _make_storage(_make_hnsw_results(20))
        activator = _make_activator()
        result = await hnsw_hybrid_recall(
            [0.1] * 384,
            storage,
            activator,
            [["anchor-1", "anchor-2"]],
        )
        assert result is not None
        _, _, scope = result
        assert "anchor-1" in scope
        assert "anchor-2" in scope

    @pytest.mark.asyncio
    async def test_bm25_fusion_expands_scope(self) -> None:
        bm25 = [("bm25-neuron-1", 5.0), ("bm25-neuron-2", 3.0)]
        storage = _make_storage(_make_hnsw_results(20), bm25_results=bm25)
        activator = _make_activator()
        result = await hnsw_hybrid_recall(
            [0.1] * 384,
            storage,
            activator,
            [["a-1"]],
            bm25_query="test query",
        )
        assert result is not None
        _, _, scope = result
        assert "bm25-neuron-1" in scope
        assert "bm25-neuron-2" in scope

    @pytest.mark.asyncio
    async def test_activator_called_with_scope(self) -> None:
        storage = _make_storage(_make_hnsw_results(20))
        activator = _make_activator()
        await hnsw_hybrid_recall(
            [0.1] * 384,
            storage,
            activator,
            [["a-1"]],
        )
        call_args = activator.activate_from_multiple.call_args
        assert call_args.kwargs.get("scope") is not None
        assert isinstance(call_args.kwargs["scope"], set)

    @pytest.mark.asyncio
    async def test_hnsw_rank_boost_applied(self) -> None:
        # neuron-0 is rank 0 in HNSW, should get highest boost
        hnsw = _make_hnsw_results(20)
        activations = {
            "neuron-0": ActivationResult(
                neuron_id="neuron-0",
                activation_level=0.5,
                hop_distance=0,
                path=["neuron-0"],
                source_anchor="neuron-0",
            ),
            "neuron-10": ActivationResult(
                neuron_id="neuron-10",
                activation_level=0.5,
                hop_distance=0,
                path=["neuron-10"],
                source_anchor="neuron-10",
            ),
        }
        storage = _make_storage(hnsw)
        activator = _make_activator(activations)
        result = await hnsw_hybrid_recall(
            [0.1] * 384,
            storage,
            activator,
            [["a-1"]],
        )
        assert result is not None
        acts, _, _ = result
        # neuron-0 (rank 0) should have higher boost than neuron-10 (rank 10)
        assert acts["neuron-0"].activation_level > acts["neuron-10"].activation_level

    @pytest.mark.asyncio
    async def test_candidate_k_passed_to_knn(self) -> None:
        storage = _make_storage(_make_hnsw_results(50))
        activator = _make_activator()
        await hnsw_hybrid_recall(
            [0.1] * 384,
            storage,
            activator,
            [["a-1"]],
            candidate_k=75,
        )
        storage.knn_search.assert_called_once_with([0.1] * 384, k=75)

    @pytest.mark.asyncio
    async def test_bm25_failure_non_fatal(self) -> None:
        storage = _make_storage(_make_hnsw_results(20))
        storage.text_search = AsyncMock(side_effect=RuntimeError("BM25 broken"))
        activator = _make_activator()
        result = await hnsw_hybrid_recall(
            [0.1] * 384,
            storage,
            activator,
            [["a-1"]],
            bm25_query="test",
        )
        # Should still succeed despite BM25 failure
        assert result is not None
