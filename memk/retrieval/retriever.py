"""
memk.retrieval.retriever
========================
Three retrieval strategies sharing the RetrievedItem contract:

  KeywordRetriever  — original SQL LIKE retriever (backward-compatible, unchanged).
  HybridRetriever   — semantic + keyword: score = w1*vec + w2*kw (v0.2).
  ScoredRetriever   — full 5-dimensional scorer: vec + keyword + importance +
                      recency + confidence (v0.3, recommended default).

ScoredRetriever algorithm
--------------------------
For query Q and candidate item C:

    score(C) = scorer.score(
        vector_similarity = cosine_norm(embed(Q), embed(C)),
        keyword_score     = 1.0 if keyword_match(C, Q) else 0.0,
        importance        = C.importance,
        created_at        = C.created_at,     # → recency via decay
        confidence        = C.confidence,
        is_fact           = C.item_type == "fact",
    ).final_score

Facts receive a fact_multiplier boost (default 1.3×) via ScoringWeights.

Access tracking
---------------
Both ScoredRetriever and HybridRetriever accept `track_access=True` (default).
When enabled, every returned item increments access_count + last_accessed_at
in the database. This feeds future frequency-based ranking.

Performance
-----------
Vector scan is O(N) — suitable for <100k rows.
Embedding model is lazy-loaded once per process on first retrieve() call.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from pydantic import BaseModel

from memk.storage.db import MemoryDB, _decode_blob
from memk.core.scorer import MemoryScorer, ScoringWeights, ScoreBreakdown

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data contract
# ---------------------------------------------------------------------------

class RetrievedItem(BaseModel):
    """
    Normalized retrieval result for any item type.
    breakdown is populated by ScoredRetriever, None for KeywordRetriever.
    """
    item_type:    str           # "fact" | "memory"
    id:           str
    content:      str
    created_at:   str
    score:        float
    importance:   float = 0.5
    confidence:   float = 1.0
    access_count: int   = 0
    decay_score:  float = 1.0
    breakdown:    Optional[ScoreBreakdown] = None


# ---------------------------------------------------------------------------
# 1. KeywordRetriever — backward-compatible (unchanged API)
# ---------------------------------------------------------------------------

class KeywordRetriever:
    """
    Original SQL LIKE-based retriever.
    Facts score 2.0, memories score 1.0.
    No scoring formula, no embeddings.
    """

    def __init__(self, db: MemoryDB):
        self.db = db

    def retrieve(self, query: str, limit: int = 10) -> List[RetrievedItem]:
        query = query.strip()
        if not query:
            return []

        results: List[RetrievedItem] = []

        for row in self.db.search_facts(keyword=query):
            results.append(RetrievedItem(
                item_type="fact",
                id=row["id"],
                content=f"{row['subject']} {row['predicate']} {row['object']}",
                created_at=row["created_at"],
                score=2.0,
                importance=float(row.get("importance") or 0.5),
                confidence=float(row.get("confidence") or 1.0),
                access_count=int(row.get("access_count") or 0),
            ))

        for row in self.db.search_memory(keyword=query):
            results.append(RetrievedItem(
                item_type="memory",
                id=row["id"],
                content=row["content"],
                created_at=row["created_at"],
                score=1.0,
                importance=float(row.get("importance") or 0.5),
                confidence=float(row.get("confidence") or 1.0),
                access_count=int(row.get("access_count") or 0),
            ))

        results.sort(key=lambda x: (x.score, x.created_at), reverse=True)
        return results[:limit]


# ---------------------------------------------------------------------------
# 2. HybridRetriever — semantic + keyword (v0.2, kept for compatibility)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Two-dimensional scorer: alpha * keyword + beta * vector_similarity.
    Suitable when you only want to tune two numbers and skip metadata scoring.
    """

    FACT_MULTIPLIER = 1.5

    def __init__(
        self,
        db: MemoryDB,
        embedder=None,
        alpha: float = 0.3,
        beta: float = 0.7,
        score_threshold: float = 0.0,
        track_access: bool = True,
    ):
        self.db = db
        self._embedder = embedder
        self.alpha = alpha
        self.beta = beta
        self.score_threshold = score_threshold
        self.track_access = track_access

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, limit: int = 10) -> List[RetrievedItem]:
        query = query.strip()
        if not query:
            return []

        embedder = self._get_embedder()
        q_vec = embedder.embed(query)

        keyword_fact_ids = {r["id"] for r in self.db.search_facts(keyword=query)}
        keyword_mem_ids  = {r["id"] for r in self.db.search_memory(keyword=query)}

        self._backfill_embeddings(embedder)

        results: List[RetrievedItem] = []

        for row in self.db.get_all_active_facts():
            content_str = f"{row['subject']} {row['predicate']} {row['object']}"
            vec_score = _cosine_score(q_vec, row.get("embedding"))
            kw_score  = 1.0 if row["id"] in keyword_fact_ids else 0.0
            combined  = (self.alpha * kw_score + self.beta * vec_score) * self.FACT_MULTIPLIER

            if combined >= self.score_threshold:
                results.append(RetrievedItem(
                    item_type="fact",
                    id=row["id"],
                    content=content_str,
                    created_at=row["created_at"],
                    score=combined,
                    importance=float(row.get("importance") or 0.5),
                    confidence=float(row.get("confidence") or 1.0),
                    access_count=int(row.get("access_count") or 0),
                ))

        for row in self.db.get_all_memories():
            vec_score = _cosine_score(q_vec, row.get("embedding"))
            kw_score  = 1.0 if row["id"] in keyword_mem_ids else 0.0
            combined  = self.alpha * kw_score + self.beta * vec_score

            if combined >= self.score_threshold:
                results.append(RetrievedItem(
                    item_type="memory",
                    id=row["id"],
                    content=row["content"],
                    created_at=row["created_at"],
                    score=combined,
                    importance=float(row.get("importance") or 0.5),
                    confidence=float(row.get("confidence") or 1.0),
                    access_count=int(row.get("access_count") or 0),
                ))

        results.sort(key=lambda x: (x.score, x.created_at), reverse=True)
        top = results[:limit]

        if self.track_access:
            self._track(top)

        return top

    def backfill_all_embeddings(self) -> int:
        return self._backfill_embeddings(self._get_embedder())

    def _get_embedder(self):
        if self._embedder is None:
            from memk.core.embedder import get_default_embedder
            self._embedder = get_default_embedder()
        return self._embedder

    def _backfill_embeddings(self, embedder, force: bool = False) -> int:
        updated = 0
        missing_mems = self.db.get_memories_without_embedding()
        if missing_mems:
            vecs = embedder.embed_batch([r["content"] for r in missing_mems])
            for row, vec in zip(missing_mems, vecs):
                self.db.update_memory_embedding(row["id"], vec)
            updated += len(missing_mems)

        missing_facts = self.db.get_facts_without_embedding()
        if missing_facts:
            texts = [f"{r['subject']} {r['predicate']} {r['object']}" for r in missing_facts]
            vecs = embedder.embed_batch(texts)
            for row, vec in zip(missing_facts, vecs):
                self.db.update_fact_embedding(row["id"], vec)
            updated += len(missing_facts)

        return updated

    def _track(self, items: List[RetrievedItem]) -> None:
        for item in items:
            try:
                if item.item_type == "fact":
                    self.db.touch_fact(item.id)
                else:
                    self.db.touch_memory(item.id)
            except Exception as exc:
                logger.warning(f"Failed to track access for {item.id}: {exc}")


# ---------------------------------------------------------------------------
# 3. CandidateFirstRetriever — low-RAM FTS5 + lightweight rerank
# ---------------------------------------------------------------------------

class CandidateFirstRetriever:
    """
    Low-RAM retriever for local agent workflows.

    Instead of hydrating every vector into RAM, this retriever asks SQLite FTS5
    for a small candidate set, reranks only those candidates with the configured
    embedder, then applies the existing metadata scorer.
    """

    def __init__(
        self,
        db: MemoryDB,
        embedder=None,
        weights: Optional[ScoringWeights] = None,
        half_life_days: float = 30.0,
        score_threshold: float = 0.0,
        track_access: bool = True,
        candidate_limit: int = 200,
    ):
        self.db = db
        self._embedder = embedder
        self.scorer = MemoryScorer(weights=weights, half_life_days=half_life_days)
        self.score_threshold = score_threshold
        self.track_access = track_access
        self.candidate_limit = max(10, int(candidate_limit))

    def retrieve(self, query: str, limit: int = 10) -> List[RetrievedItem]:
        query = query.strip()
        if not query:
            return []

        embedder = self._get_embedder()
        q_vec = embedder.embed(query)

        per_type_limit = max(limit, self.candidate_limit // 2)
        fact_rows = self.db.search_facts_fts(query, limit=per_type_limit)
        memory_rows = self.db.search_memory_fts(query, limit=per_type_limit)

        results: List[RetrievedItem] = []
        seen: set[tuple[str, str]] = set()

        for row in fact_rows:
            key = ("fact", row["id"])
            if key in seen:
                continue
            seen.add(key)
            content = f"{row['subject']} {row['predicate']} {row['object']}"
            breakdown = self.scorer.score(
                vector_similarity=_candidate_vector_score(
                    q_vec, row.get("embedding"), content, embedder
                ),
                keyword_score=1.0,
                importance=float(row.get("importance") or 0.5),
                created_at=row["created_at"],
                confidence=float(row.get("confidence") or 1.0),
                is_fact=True,
            )
            if breakdown.final_score >= self.score_threshold:
                results.append(RetrievedItem(
                    item_type="fact",
                    id=row["id"],
                    content=content,
                    created_at=row["created_at"],
                    score=breakdown.final_score,
                    importance=float(row.get("importance") or 0.5),
                    confidence=float(row.get("confidence") or 1.0),
                    access_count=int(row.get("access_count") or 0),
                    decay_score=float(row.get("decay_score") or 1.0),
                    breakdown=breakdown,
                ))

        for row in memory_rows:
            key = ("memory", row["id"])
            if key in seen:
                continue
            seen.add(key)
            content = row["content"]
            breakdown = self.scorer.score(
                vector_similarity=_candidate_vector_score(
                    q_vec, row.get("embedding"), content, embedder
                ),
                keyword_score=1.0,
                importance=float(row.get("importance") or 0.5),
                created_at=row["created_at"],
                confidence=float(row.get("confidence") or 1.0),
                is_fact=False,
            )
            if breakdown.final_score >= self.score_threshold:
                results.append(RetrievedItem(
                    item_type="memory",
                    id=row["id"],
                    content=content,
                    created_at=row["created_at"],
                    score=breakdown.final_score,
                    importance=float(row.get("importance") or 0.5),
                    confidence=float(row.get("confidence") or 1.0),
                    access_count=int(row.get("access_count") or 0),
                    decay_score=float(row.get("decay_score") or 1.0),
                    breakdown=breakdown,
                ))

        results.sort(key=lambda x: (x.score, x.created_at), reverse=True)
        top = results[:limit]

        if self.track_access:
            self._track(top)

        return top

    def _get_embedder(self):
        if self._embedder is None:
            from memk.core.embedder import get_default_embedder
            self._embedder = get_default_embedder()
        return self._embedder

    def _track(self, items: List[RetrievedItem]) -> None:
        for item in items:
            try:
                if item.item_type == "fact":
                    self.db.touch_fact(item.id)
                else:
                    self.db.touch_memory(item.id)
            except Exception as exc:
                logger.warning(f"Access tracking failed for {item.id}: {exc}")


# ---------------------------------------------------------------------------
# 4. ScoredRetriever — full 5-D scorer (v0.3, recommended)
# ---------------------------------------------------------------------------

class ScoredRetriever:
    """
    Full memory scoring and ranking system.

    Scoring formula (all components normalized to [0, 1]):
        final_score = w1 * vector_similarity
                    + w2 * keyword_score
                    + w3 * importance
                    + w4 * recency          ← exponential decay from created_at
                    + w5 * confidence
                    [× fact_multiplier if item is a fact]

    The retriever:
      1. Lazy-loads the embedding model on first call.
      2. Backfills missing embeddings in batch before scoring.
      3. Tracks access (access_count, last_accessed_at) for returned items.
      4. Attaches a ScoreBreakdown to every RetrievedItem for full transparency.

    Parameters
    ----------
    db              : MemoryDB instance.
    embedder        : Optional custom embedder; defaults to SentenceTransformer.
    weights         : ScoringWeights for tuning the formula.
    half_life_days  : Recency decay parameter. Default 30 days.
    score_threshold : Exclude items scoring below this floor.
    track_access    : If True, update access stats in the DB after retrieval.
    """

    def __init__(
        self,
        db: MemoryDB,
        embedder=None,
        weights: Optional[ScoringWeights] = None,
        half_life_days: float = 30.0,
        score_threshold: float = 0.0,
        track_access: bool = True,
        index=None,
        cache=None,
    ):
        self.db = db
        self._embedder = embedder
        self.scorer = MemoryScorer(weights=weights, half_life_days=half_life_days)
        self.score_threshold = score_threshold
        self.track_access = track_access
        self.index = index
        self.cache = cache

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, limit: int = 10) -> List[RetrievedItem]:
        """
        Return the top-`limit` items ranked by final_score.

        Missing embeddings are backfilled on first call. Returned items carry
        a populated .breakdown attribute for score introspection.
        """
        query = query.strip()
        if not query:
            return []

        embedder = self._get_embedder()
        
        # Layer 1 Cache: Query Embedding
        q_vec = None
        if self.cache:
            q_vec = self.cache.embeddings.get(query)
            
        if q_vec is None:
            q_vec = embedder.embed(query)
            if self.cache:
                self.cache.embeddings.set(query, q_vec)

        # Keyword hit sets (O(1) lookup during scoring loop)
        keyword_fact_ids = {r["id"] for r in self.db.search_facts(keyword=query)}
        keyword_mem_ids  = {r["id"] for r in self.db.search_memory(keyword=query)}

        results: List[RetrievedItem] = []

        if self.index:
            # --- RAM-FIRST TOP-K SEARCH (Sub-millisecond) ---
            # Search returns (IndexEntry, similarity_score)
            index_hits = self.index.search(q_vec, top_k=limit * 5)
            for entry, sim in index_hits:
                breakdown = self.scorer.score(
                    vector_similarity=sim,
                    keyword_score=1.0 if entry.id in keyword_fact_ids or entry.id in keyword_mem_ids else 0.0,
                    importance=entry.importance,
                    created_at=entry.created_at,
                    confidence=entry.confidence,
                    is_fact=(entry.item_type == "fact"),
                )
                if breakdown.final_score >= self.score_threshold:
                    results.append(RetrievedItem(
                        item_type=entry.item_type,
                        id=entry.id,
                        content=entry.content,
                        created_at=entry.created_at,
                        score=breakdown.final_score,
                        importance=entry.importance,
                        confidence=entry.confidence,
                        access_count=entry.access_count,
                        decay_score=entry.decay_score,
                        breakdown=breakdown,
                    ))
        else:
            # --- FULL SQLITE SCAN (Legacy fallback) ---
            for row in self.db.get_all_active_facts():
                content = f"{row['subject']} {row['predicate']} {row['object']}"
                breakdown = self.scorer.score(
                    vector_similarity=_cosine_score(q_vec, row.get("embedding")),
                    keyword_score=1.0 if row["id"] in keyword_fact_ids else 0.0,
                    importance=float(row.get("importance") or 0.5),
                    created_at=row["created_at"],
                    confidence=float(row.get("confidence") or 1.0),
                    is_fact=True,
                )
                if breakdown.final_score >= self.score_threshold:
                    results.append(RetrievedItem(
                        item_type="fact", id=row["id"], content=content,
                        created_at=row["created_at"], score=breakdown.final_score,
                        importance=float(row.get("importance") or 0.5),
                        confidence=float(row.get("confidence") or 1.0),
                        access_count=int(row.get("access_count") or 0),
                        decay_score=float(row.get("decay_score") or 1.0),
                        breakdown=breakdown,
                    ))

            for row in self.db.get_all_memories():
                breakdown = self.scorer.score(
                    vector_similarity=_cosine_score(q_vec, row.get("embedding")),
                    keyword_score=1.0 if row["id"] in keyword_mem_ids else 0.0,
                    importance=float(row.get("importance") or 0.5),
                    created_at=row["created_at"],
                    confidence=float(row.get("confidence") or 1.0),
                    is_fact=False,
                )
                if breakdown.final_score >= self.score_threshold:
                    results.append(RetrievedItem(
                        item_type="memory", id=row["id"], content=row["content"],
                        created_at=row["created_at"], score=breakdown.final_score,
                        importance=float(row.get("importance") or 0.5),
                        confidence=float(row.get("confidence") or 1.0),
                        access_count=int(row.get("access_count") or 0),
                        decay_score=float(row.get("decay_score") or 1.0),
                        breakdown=breakdown,
                    ))

        # Sort primary: final_score DESC; tiebreaker: recency DESC
        results.sort(key=lambda x: (x.score, x.created_at), reverse=True)
        top = results[:limit]

        if self.track_access:
            self._track(top)

        return top

    def rank_candidates(self, query: str, q_vec: np.ndarray, index_hits: List[Tuple], limit: int, graph_index=None) -> List[RetrievedItem]:
        """
        Pure ranking logic separated from retrieval.
        Used by the service layer for deadline-aware pipelines.
        Includes graph propagation if graph_index is provided.
        """
        # Keyword hit sets (O(1) lookup during scoring loop)
        keyword_fact_ids = {r["id"] for r in self.db.search_facts(keyword=query)}
        keyword_mem_ids  = {r["id"] for r in self.db.search_memory(keyword=query)}

        # Phase 1: Base scoring
        base_scores = []
        for entry, sim in index_hits:
            breakdown = self.scorer.score(
                vector_similarity=sim,
                keyword_score=1.0 if entry.id in keyword_fact_ids or entry.id in keyword_mem_ids else 0.0,
                importance=entry.importance,
                created_at=entry.created_at,
                confidence=entry.confidence,
                is_fact=(entry.item_type == "fact"),
            )
            base_scores.append((entry, breakdown))

        # Phase 2: Graph Propagation (extract seeds -> spread -> gather bonus)
        graph_bonus = {}
        if graph_index is not None and graph_index.num_entities > 0:
            from memk.core.graph_propagation import propagate_ppnp
            
            # Sort base hits to get valid seeds
            base_scores.sort(key=lambda x: x[1].final_score, reverse=True)
            top_seeds = base_scores[:limit]
            
            seed_scores = {}
            for entry, bd in top_seeds:
                if bd.final_score <= 0: continue
                # We map memory_ids -> entities
                m_int = graph_index.memory_id_map.get(entry.id)
                if m_int is not None:
                    start, end = graph_index.m2e_indptr[m_int], graph_index.m2e_indptr[m_int+1]
                    for e_int in graph_index.m2e_indices[start:end]:
                        seed_scores[e_int] = max(seed_scores.get(e_int, 0.0), bd.final_score)
            
            if seed_scores:
                activated_entities = propagate_ppnp(
                    seed_scores=seed_scores,
                    indptr=graph_index.e2e_indptr,
                    indices=graph_index.e2e_indices,
                    weights=graph_index.e2e_weights,
                    num_entities=graph_index.num_entities,
                    alpha=0.3,
                    steps=3,
                    max_active_entities=50
                )
                
                # Project network activation back to memory IDs
                for e_int, act_score in activated_entities.items():
                    start, end = graph_index.e2m_indptr[e_int], graph_index.e2m_indptr[e_int+1]
                    for m_int in graph_index.e2m_indices[start:end]:
                        mem_id = graph_index.memory_ids[m_int]
                        graph_bonus[mem_id] = max(graph_bonus.get(mem_id, 0.0), act_score)

        # Phase 3: Final Assembly and Rescoring
        results: List[RetrievedItem] = []
        for entry, base_bd in base_scores:
            g_score = graph_bonus.get(entry.id, 0.0)
            if g_score > 0.0:
                final_bd = self.scorer.score(
                    vector_similarity=base_bd.vector_similarity,
                    keyword_score=base_bd.keyword_score,
                    importance=base_bd.importance,
                    created_at=entry.created_at,
                    confidence=base_bd.confidence,
                    graph_score=g_score,
                    is_fact=(entry.item_type == "fact"),
                )
            else:
                final_bd = base_bd
                
            if final_bd.final_score >= self.score_threshold:
                results.append(RetrievedItem(
                    item_type=entry.item_type,
                    id=entry.id,
                    content=entry.content,
                    created_at=entry.created_at,
                    score=final_bd.final_score,
                    importance=entry.importance,
                    confidence=entry.confidence,
                    access_count=entry.access_count,
                    decay_score=entry.decay_score,
                    breakdown=final_bd,
                ))

        results.sort(key=lambda x: (x.score, x.created_at), reverse=True)
        top = results[:limit]

        if self.track_access:
            self._track(top)
        
        return top

    def backfill_all_embeddings(self) -> int:
        """Explicitly embed all rows lacking a vector. Returns count updated."""
        return self._backfill_embeddings(self._get_embedder())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_embedder(self):
        if self._embedder is None:
            from memk.core.embedder import get_default_embedder
            self._embedder = get_default_embedder()
        return self._embedder

    def _backfill_embeddings(self, embedder) -> int:
        updated = 0
        missing_mems = self.db.get_memories_without_embedding()
        if missing_mems:
            vecs = embedder.embed_batch([r["content"] for r in missing_mems])
            for row, vec in zip(missing_mems, vecs):
                self.db.update_memory_embedding(row["id"], vec)
            updated += len(missing_mems)
            logger.debug(f"Backfilled embeddings: {len(missing_mems)} memories.")

        missing_facts = self.db.get_facts_without_embedding()
        if missing_facts:
            texts = [f"{r['subject']} {r['predicate']} {r['object']}" for r in missing_facts]
            vecs = embedder.embed_batch(texts)
            for row, vec in zip(missing_facts, vecs):
                self.db.update_fact_embedding(row["id"], vec)
            updated += len(missing_facts)
            logger.debug(f"Backfilled embeddings: {len(missing_facts)} facts.")

        return updated

    def _track(self, items: List[RetrievedItem]) -> None:
        for item in items:
            try:
                if item.item_type == "fact":
                    self.db.touch_fact(item.id)
                else:
                    self.db.touch_memory(item.id)
            except Exception as exc:
                logger.warning(f"Access tracking failed for {item.id}: {exc}")


# ---------------------------------------------------------------------------
# Module-level helper (shared by all retrievers)
# ---------------------------------------------------------------------------

def _cosine_score(q_vec: np.ndarray, emb_blob: Optional[bytes]) -> float:
    """
    Cosine similarity normalized from [-1,1] to [0,1].
    Returns 0.0 safely if no embedding is stored.
    """
    if emb_blob is None:
        return 0.0
    try:
        c_vec = _decode_blob(emb_blob)
        denom = np.linalg.norm(q_vec) * np.linalg.norm(c_vec)
        if denom < 1e-10:
            return 0.0
        raw = float(np.dot(q_vec, c_vec) / denom)
        return (raw + 1.0) / 2.0
    except Exception as exc:
        logger.warning(f"cosine_score failed: {exc}")
        return 0.0


def _candidate_vector_score(q_vec: np.ndarray, emb_blob: Optional[bytes], content: str, embedder) -> float:
    """Score a candidate using a stored vector when dimensions match, else embed on demand."""
    if emb_blob is not None and len(emb_blob) // 4 == int(q_vec.shape[0]):
        return _cosine_score(q_vec, emb_blob)
    try:
        c_vec = embedder.embed(content)
        return _cosine_vec_score(q_vec, c_vec)
    except Exception as exc:
        logger.debug("candidate vector scoring skipped: %s", exc)
        return 0.0


def _cosine_vec_score(q_vec: np.ndarray, c_vec: np.ndarray) -> float:
    denom = np.linalg.norm(q_vec) * np.linalg.norm(c_vec)
    if denom < 1e-10:
        return 0.0
    raw = float(np.dot(q_vec, c_vec) / denom)
    return (raw + 1.0) / 2.0
