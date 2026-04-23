"""Hybrid Recall Strategy for InfinityDB.

HNSW-first recall: use vector search to narrow candidates, then run
cognitive layer (spreading activation + fibers) only on the candidate set.

Two-phase approach:
  Phase 1 (HNSW, <5ms): knn_query → candidate neuron IDs (50-100)
  Phase 2 (cognitive, <15ms): scoped activation within candidate subgraph

This preserves NM's cognitive depth while bounding the search space to O(K)
instead of O(N) full-graph traversal.
"""

from __future__ import annotations

import logging
from typing import Any

from neural_memory.engine.activation import ActivationResult

logger = logging.getLogger(__name__)

# Default candidate pool size — 3x typical result count for margin
DEFAULT_CANDIDATE_K = 100
# Minimum HNSW results before falling back to full pipeline
FALLBACK_THRESHOLD = 10


async def hnsw_hybrid_recall(
    query_embedding: list[float],
    storage: Any,
    activator: Any,
    anchor_sets: list[list[str]],
    *,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    max_hops: int = 2,
    anchor_activations: dict[str, float] | None = None,
    bm25_query: str | None = None,
    bm25_limit: int = 30,
) -> tuple[dict[str, ActivationResult], list[str], set[str]] | None:
    """Execute HNSW-first hybrid recall.

    Args:
        query_embedding: Query vector for HNSW search.
        storage: InfinityDBStorage adapter (must have knn_search).
        activator: SpreadingActivation instance.
        anchor_sets: Pre-computed anchor sets from retrieval pipeline.
        candidate_k: Number of HNSW candidates to fetch.
        max_hops: Max BFS hops within scoped activation.
        anchor_activations: RRF-fused initial activation levels.
        bm25_query: Optional text query for BM25 fusion.
        bm25_limit: Max BM25 results to fuse.

    Returns:
        (activations, intersections, scope) if successful,
        None if fallback to full pipeline is needed.
    """
    # Phase 1: HNSW candidate set
    if not hasattr(storage, "knn_search"):
        return None

    hnsw_results = await storage.knn_search(query_embedding, k=candidate_k)
    if len(hnsw_results) < FALLBACK_THRESHOLD:
        logger.debug(
            "HNSW returned %d results (< %d threshold), falling back to full pipeline",
            len(hnsw_results),
            FALLBACK_THRESHOLD,
        )
        return None

    # Build candidate scope from HNSW results
    scope: set[str] = {nid for nid, _sim in hnsw_results}

    # Fuse BM25 results into scope if available
    if bm25_query and hasattr(storage, "text_search"):
        try:
            bm25_results = await storage.text_search(bm25_query, limit=bm25_limit)
            for nid, _score in bm25_results:
                scope.add(nid)
        except Exception:
            logger.debug("BM25 fusion failed in hybrid recall (non-critical)", exc_info=True)

    # Include anchors in scope so activation can start from them
    for anchor_list in anchor_sets:
        for nid in anchor_list:
            scope.add(nid)

    logger.debug(
        "Hybrid recall: %d HNSW candidates + anchors → %d scope neurons",
        len(hnsw_results),
        len(scope),
    )

    # Phase 2: Scoped activation — BFS only within candidate set
    activations, intersections = await activator.activate_from_multiple(
        anchor_sets,
        max_hops=max_hops,
        anchor_activations=anchor_activations,
        scope=scope,
    )

    # Boost HNSW-ranked neurons that survived activation
    hnsw_rank = {nid: rank for rank, (nid, _sim) in enumerate(hnsw_results)}
    for nid, result in activations.items():
        if nid in hnsw_rank:
            # RRF-style boost: closer HNSW rank → higher boost
            rank_boost = 1.0 / (60 + hnsw_rank[nid] + 1)
            boosted_level = min(1.0, result.activation_level + rank_boost)
            activations[nid] = ActivationResult(
                neuron_id=nid,
                activation_level=boosted_level,
                hop_distance=result.hop_distance,
                path=result.path,
                source_anchor=result.source_anchor,
            )

    return activations, intersections, scope
