"""Sparse Selective Restore (SSC-lite) for activation cache.

Rather than restoring all cached activation states on every recall, we use
the query's embedding to cosine-rank cached neurons and keep only the top-K
most relevant entries. This reduces noise from stale warm activations and
keeps the warm boost focused on neurons likely to be re-activated.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING

from neural_memory.cache.models import CachedState

if TYPE_CHECKING:
    from neural_memory.engine.embedding.provider import EmbeddingProvider
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 20
DEFAULT_MIN_SIMILARITY = 0.3
# Cap on concurrent storage reads during embedding fetch.
_FETCH_CONCURRENCY = 20


def _cosine(vec_a: list[float], vec_b: list[float]) -> float:
    """Local cosine similarity — avoids awaiting provider.similarity in a tight loop."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b, strict=True):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


async def select_relevant(
    cached_states: list[CachedState],
    query: str,
    embedding_provider: EmbeddingProvider | None,
    storage: NeuralStorage,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> list[CachedState]:
    """Rank cached states by cosine similarity to query and return top-K.

    Falls back to activation-ranked top-K when embeddings or provider are
    unavailable, so callers always get a usable warm subset.

    Args:
        cached_states: All cached activation states (from load_snapshot)
        query: User recall query
        embedding_provider: Active embedding provider (may be None)
        storage: NeuralStorage to fetch neuron embeddings
        top_k: Max entries to return (default 20)
        min_similarity: Minimum cosine score (default 0.3)

    Returns:
        Top-K CachedState entries ranked by relevance, or activation-ranked
        fallback when ranking cannot be computed.
    """
    if not cached_states:
        return []

    top_k = max(1, top_k)

    if embedding_provider is None or not query:
        return _fallback_top_k(cached_states, top_k)

    try:
        query_vec = await embedding_provider.embed(query)
    except Exception:
        logger.debug("Selector: query embed failed, using activation fallback", exc_info=True)
        return _fallback_top_k(cached_states, top_k)

    if not query_vec:
        return _fallback_top_k(cached_states, top_k)

    neuron_ids = [s.neuron_id for s in cached_states]
    embeddings = await _fetch_embeddings(storage, neuron_ids)

    if not embeddings:
        return _fallback_top_k(cached_states, top_k)

    scored: list[tuple[CachedState, float]] = []
    dim_mismatch_count = 0
    query_dim = len(query_vec)
    for state in cached_states:
        vec = embeddings.get(state.neuron_id)
        if vec is None:
            continue
        if len(vec) != query_dim:
            dim_mismatch_count += 1
            continue
        sim = _cosine(query_vec, vec)
        if sim >= min_similarity:
            scored.append((state, sim))

    if dim_mismatch_count and dim_mismatch_count == len(embeddings):
        logger.warning(
            "SSC-lite: all %d cached embeddings have dim != query dim (%d). "
            "Embedding model may have changed. Falling back to activation ranking.",
            dim_mismatch_count,
            query_dim,
        )

    if not scored:
        return _fallback_top_k(cached_states, top_k)

    scored.sort(key=lambda x: x[1], reverse=True)
    return [state for state, _ in scored[:top_k]]


def _fallback_top_k(cached_states: list[CachedState], top_k: int) -> list[CachedState]:
    """Activation-ranked fallback when embedding ranking is unavailable."""
    return sorted(cached_states, key=lambda s: s.activation_level, reverse=True)[:top_k]


async def _fetch_embeddings(
    storage: NeuralStorage,
    neuron_ids: list[str],
) -> dict[str, list[float]]:
    """Fetch embeddings for neurons via metadata.

    Embeddings are stored in `neuron.metadata["_embedding"]`. Missing or
    malformed entries are skipped silently — selector just uses what it has.
    """
    if not neuron_ids:
        return {}

    get_neuron = getattr(storage, "get_neuron", None)
    if get_neuron is None:
        return {}

    semaphore = asyncio.Semaphore(_FETCH_CONCURRENCY)

    async def _fetch_one(nid: str) -> tuple[str, list[float] | None]:
        async with semaphore:
            try:
                neuron = await get_neuron(nid)
            except Exception:
                return (nid, None)
        if neuron is None:
            return (nid, None)
        meta = getattr(neuron, "metadata", None) or {}
        raw = meta.get("_embedding")
        if isinstance(raw, list) and raw and all(isinstance(x, (int, float)) for x in raw):
            return (nid, [float(x) for x in raw])
        return (nid, None)

    results = await asyncio.gather(*[_fetch_one(nid) for nid in neuron_ids])
    return {nid: vec for nid, vec in results if vec is not None}


def warm_activations_from_states(states: list[CachedState]) -> dict[str, float]:
    """Convenience: convert CachedState list to {neuron_id: level} map."""
    return {s.neuron_id: s.activation_level for s in states}
