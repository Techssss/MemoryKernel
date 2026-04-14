"""
memk.core.service
=================
Transport-agnostic service layer with full per-request tracing,
latency guards, and graceful degradation.

v0.5: Uses EmbeddingPipeline for cached, pooled, telemetered embedding.
      Embedding cache is now INTERNAL to the pipeline — no manual cache
      lookups needed in the service layer.
"""

import logging
import datetime
import asyncio
from typing import List, Dict, Any, Optional

from memk.core.runtime import get_runtime
from memk.core.tracing import TraceContext, get_collector
from memk.retrieval.index import IndexEntry

logger = logging.getLogger("memk.service")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MAX_CANDIDATES = 50       # Max items to retrieve before ranking
MAX_RETURN_LIMIT = 25     # Max items to return (user-facing cap)
DEADLINE_MS = 20.0        # Deadline — degrade to partial results if exceeded
SOFT_LIMIT_MS = 15.0      # When to start skipping optional steps
SLOW_THRESHOLD_MS = 15.0  # Log breakdown above this threshold


class MemoryKernelService:
    """
    Transport-agnostic service layer for MemoryKernel.
    v0.5: EmbeddingPipeline integration (cache + pool + telemetry).
    """

    def __init__(self, deadline_ms: float = DEADLINE_MS):
        self.runtime = get_runtime()
        self.deadline_ms = deadline_ms
        self.collector = get_collector(slow_threshold_ms=SLOW_THRESHOLD_MS)

    def ensure_initialized(self):
        if not self.runtime._is_initialized:
            self.runtime.initialize()

    # ------------------------------------------------------------------
    # add_memory — write path
    # ------------------------------------------------------------------

    async def add_memory(
        self,
        content: str,
        importance: float = 0.5,
        confidence: float = 1.0,
        workspace_id: str = "default",
    ) -> Dict[str, Any]:
        self.ensure_initialized()

        with TraceContext("add_memory") as tc:
            # 1. Embed via pipeline (auto-cached)
            with tc.span("embed", content_len=len(content)):
                # Async offload to pool
                try:
                    q_fut = self.runtime.embedder.embed_async(content)
                    emb = await asyncio.wrap_future(q_fut)
                except RuntimeError:
                    # Rare backpressure on write — we must wait or fail
                    # For now, we block on the sync embed if queue is full 
                    # better than skipping a write-path embedding
                    tc.mark_degraded("embedding queue full (sync fallback)")
                    emb = self.runtime.embedder.embed(content)

            # 2. Persist to SQLite
            with tc.span("db_persist"):
                # DB is synchronous for now, wrap in to_thread if bottlenecked
                mem_id = self.runtime.db.insert_memory(
                    content, embedding=emb,
                    importance=importance, confidence=confidence,
                )

            # 3. Sync to RAM index
            with tc.span("index_sync"):
                now_str = datetime.datetime.now().isoformat()
                self.runtime.index.add_entry(
                    IndexEntry(
                        id=mem_id, item_type="memory", content=content,
                        importance=importance, confidence=confidence,
                        created_at=now_str, decay_score=1.0, access_count=0,
                    ),
                    emb,
                )

            # 4. Extract structured facts
            with tc.span("extract_facts"):
                facts = self.runtime.extractor.extract_facts(content)

            extracted = []
            if facts:
                if tc.elapsed_ms() < self.deadline_ms:
                    with tc.span("embed_facts", fact_count=len(facts)):
                        fact_texts = [f"{f.subject} {f.relation} {f.object}" for f in facts]
                        # Use sync embed_batch as this is already in a thread-like context 
                        # or async wait for batch
                        try:
                            f_fut = self.runtime.embedder.embed_batch_async(fact_texts)
                            fact_embs = await asyncio.wrap_future(f_fut)
                        except RuntimeError:
                            fact_embs = self.runtime.embedder.embed_batch(fact_texts)

                    with tc.span("persist_facts"):
                        for f, f_emb in zip(facts, fact_embs):
                            f_id = self.runtime.db.insert_fact(
                                f.subject, f.relation, f.object,
                                embedding=f_emb,
                                importance=importance, confidence=confidence,
                            )
                            t_str = f"{f.subject} {f.relation} {f.object}"
                            self.runtime.index.add_entry(
                                IndexEntry(
                                    id=f_id, item_type="fact", content=t_str,
                                    importance=importance, confidence=confidence,
                                    created_at=now_str, decay_score=1.0, access_count=0,
                                ),
                                f_emb,
                            )
                            extracted.append({"id": f_id, "triplet": t_str})
                else:
                    tc.mark_degraded()
                    logger.info("Deadline pressure: deferring fact embedding.")

            with tc.span("cache_invalidate"):
                self.runtime.cache.invalidate_structural()

            tc.set_item_count(1 + len(extracted))

        self.collector.record(tc.trace)
        return {"id": mem_id, "extracted_facts": extracted}

    # ------------------------------------------------------------------
    # search — read path
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
        workspace_id: str = "default",
    ) -> List[Dict[str, Any]]:
        self.ensure_initialized()
        limit = min(limit, MAX_RETURN_LIMIT)

        with TraceContext("search") as tc:
            # Layer 1: Full-result cache
            with tc.span("cache_check"):
                cache_key = (workspace_id, query, limit)
                cached = self.runtime.cache.search_results.get(cache_key)

            if cached is not None:
                tc.mark_cache_hit()
                tc.set_item_count(len(cached))
                self.collector.record(tc.trace)
                return cached

            # Layer 2: Embed via pipeline (with backpressure check)
            q_vec = None
            if tc.elapsed_ms() < SOFT_LIMIT_MS:
                with tc.span("embed", query_len=len(query)):
                    try:
                        q_fut = self.runtime.embedder.embed_async(query)
                        q_vec = await asyncio.wrap_future(q_fut)
                    except RuntimeError:
                        # Backpressure triggered: Fallback to lexical immediately
                        tc.mark_degraded("fast backpressure (queue full)")
                        q_vec = None
            else:
                tc.mark_degraded("skipped embedding (latency budget)")

            # Layer 3: RAM index search (vector or lexical)
            with tc.span("retrieve", index_size=len(self.runtime.index)):
                max_candidates = min(limit * 3, MAX_CANDIDATES)
                results = self._retrieve_with_deadline(
                    tc, query, q_vec, limit, max_candidates,
                )

            tc.set_item_count(len(results))

            with tc.span("serialize"):
                serialized = [item.__dict__ for item in results]
                self.runtime.cache.search_results.set(cache_key, serialized)

        self.collector.record(tc.trace)
        return serialized

    # ------------------------------------------------------------------
    # build_context — compound read path
    # ------------------------------------------------------------------

    async def build_context(
        self,
        query: str,
        max_chars: int = 500,
        threshold: float = 0.3,
        workspace_id: str = "default",
    ) -> str:
        self.ensure_initialized()

        with TraceContext("build_context") as tc:
            with tc.span("cache_check"):
                cache_key = (workspace_id, query, max_chars, threshold)
                cached = self.runtime.cache.contexts.get(cache_key)

            if cached is not None:
                tc.mark_cache_hit()
                self.collector.record(tc.trace)
                return cached

            # Embed via pipeline
            q_vec = None
            if tc.elapsed_ms() < SOFT_LIMIT_MS:
                with tc.span("embed", query_len=len(query)):
                    try:
                        q_fut = self.runtime.embedder.embed_async(query)
                        q_vec = await asyncio.wrap_future(q_fut)
                    except RuntimeError:
                        # Backpressure triggered: Fallback to lexical
                        tc.mark_degraded("fast backpressure (queue full)")
                        q_vec = None
            else:
                tc.mark_degraded("skipped embedding (latency budget)")

            with tc.span("retrieve", index_size=len(self.runtime.index)):
                max_candidates = min(MAX_RETURN_LIMIT * 3, MAX_CANDIDATES)
                all_items = self._retrieve_with_deadline(
                    tc, query, q_vec, MAX_RETURN_LIMIT, max_candidates,
                )
                items = [i for i in all_items if i.score >= threshold]

            tc.set_item_count(len(items))

            # Skip conflict resolution if budget is tight
            conflicts = []
            if tc.elapsed_ms() < SOFT_LIMIT_MS:
                with tc.span("conflict_resolution"):
                    active_fact_ids = [i.id for i in items if i.item_type == "fact"]
                    conflicts = self.runtime.db.get_fact_conflicts(active_fact_ids)
            else:
                tc.mark_degraded("skipped conflict resolution")

            with tc.span("assemble", max_chars=max_chars):
                self.runtime.builder.max_chars = max_chars
                context_str = self.runtime.builder.build_context(items, conflicts=conflicts)

            with tc.span("cache_set"):
                self.runtime.cache.contexts.set(cache_key, context_str)

        self.collector.record(tc.trace)
        return context_str

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self) -> Dict[str, Any]:
        self.ensure_initialized()
        stats = self.runtime.db.get_stats()
        from memk.core.forgetting import ForgettingEngine
        engine = ForgettingEngine()
        states = self.runtime.db.get_state_counts(engine.cold_threshold, engine.warm_threshold)
        return {
            "db_stats": stats,
            "memory_health": states,
            "runtime": self.runtime.get_diagnostics(),
        }

    def get_tail_latency_report(self) -> Dict[str, Any]:
        return self.collector.get_report()

    def get_embedding_telemetry(self) -> Dict[str, Any]:
        """Return detailed embedding pipeline telemetry."""
        if self.runtime.embedder:
            return self.runtime.embedder.get_telemetry()
        return {}

    def submit_job(self, job_type: str, job_func: str = "", **kwargs) -> str:
        self.ensure_initialized()
        if job_type == "synthesize":
            def task():
                from memk.synthesis.synthesizer import KnowledgeSynthesizer
                return KnowledgeSynthesizer(self.runtime.db).synthesize_all()
            return self.runtime.jobs.submit(job_type, task)
        return "unsupported"

    # ------------------------------------------------------------------
    # Private: deadline-aware retrieval
    # ------------------------------------------------------------------

    def _retrieve_with_deadline(self, tc, query, q_vec, limit, max_candidates):
        from memk.retrieval.retriever import RetrievedItem

        if self.runtime.index and len(self.runtime.index) > 0:
            # Fast path: RAM index search
            with tc.span("index_search", max_candidates=max_candidates):
                if q_vec is not None:
                    index_hits = self.runtime.index.search(q_vec, top_k=max_candidates)
                else:
                    # Lexical fallback when embedding is skipped or too slow
                    index_hits = self.runtime.index.search_lexical(query, top_k=max_candidates)

            if tc.elapsed_ms() > self.deadline_ms:
                tc.mark_degraded("deadline reached in index search")
                results = []
                for entry, sim in index_hits[:limit]:
                    results.append(RetrievedItem(
                        item_type=entry.item_type,
                        id=entry.id,
                        content=entry.content,
                        created_at=entry.created_at,
                        score=float(sim),
                        importance=entry.importance,
                        confidence=entry.confidence,
                        access_count=entry.access_count,
                        decay_score=entry.decay_score,
                    ))
                return results

            # Full scoring in RAM
            with tc.span("rank", candidates=len(index_hits)):
                query_lower = query.lower()
                results = []
                for entry, sim in index_hits:
                    kw_match = 1.0 if query_lower in entry.content.lower() else 0.0
                    breakdown = self.runtime.retriever.scorer.score(
                        vector_similarity=sim,
                        keyword_score=kw_match,
                        importance=entry.importance,
                        created_at=entry.created_at,
                        confidence=entry.confidence,
                        is_fact=(entry.item_type == "fact"),
                    )
                    if breakdown.final_score >= self.runtime.retriever.score_threshold:
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

                results.sort(key=lambda x: (x.score, x.created_at), reverse=True)
                return results[:limit]
        else:
            return self.runtime.retriever.retrieve(query, limit=limit)
