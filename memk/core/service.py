"""
memk.core.service
=================
Transport-agnostic service layer with full per-request tracing,
latency guards, graceful degradation, and generation-based consistency.

v1.0: Workspace-aware and Single-writer enforced.
v1.1: Generation tracking for stale-context detection.
"""

import logging
import datetime
import asyncio
import os
from typing import List, Dict, Any, Optional

from memk.core.runtime import get_runtime, WorkspaceRuntime
from memk.core.tracing import TraceContext, get_collector
from memk.retrieval.index import IndexEntry
from memk.workspace.schema import ResponseMetadata

logger = logging.getLogger("memk.service")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MAX_CANDIDATES = 50       # Max items to retrieve before ranking
MAX_RETURN_LIMIT = 25     # Max items to return (user-facing cap)
DEADLINE_MS = 5000.0      # Deadline — degrade to partial results if exceeded
SOFT_LIMIT_MS = 2000.0    # When to start skipping optional steps
SLOW_THRESHOLD_MS = 15.0  # Log breakdown above this threshold


class MemoryKernelService:
    """
    Transport-agnostic service layer for MemoryKernel.
    v1.0: Multi-workspace support via RuntimeManager.
    """

    def __init__(self, deadline_ms: float = DEADLINE_MS):
        self.global_runtime = get_runtime()
        self.deadline_ms = deadline_ms
        self.collector = get_collector(slow_threshold_ms=SLOW_THRESHOLD_MS)

    def _get_runtime(self, workspace_id: str) -> WorkspaceRuntime:
        return self.global_runtime.get_workspace_runtime(workspace_id)

    def _ensure_daemon_writability(self):
        """
        Policy check: only the daemon (identified by a specific environment flag or context)
        is allowed to perform writes in v1.0 multi-workspace mode.
        """
        if not os.getenv("MEMK_DAEMON_MODE"):
            raise PermissionError(
                "Direct write access to MemoryKernel storage is disabled in multi-workspace mode. "
                "Please start the daemon using 'memk serve' and ensure it is running."
            )

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
        runtime = self._get_runtime(workspace_id)
        self._ensure_daemon_writability()

        with TraceContext("add_memory") as tc:
            # 1. Embed via global pipeline (shared across all workspaces)
            with tc.span("embed", content_len=len(content)):
                try:
                    q_fut = self.global_runtime.embedder_pipeline.embed_async(content)
                    emb = await asyncio.wrap_future(q_fut)
                except (RuntimeError, AttributeError):
                    emb = self.global_runtime.shared_embedder.embed(content)

            # 2. Persist to SQLite (Workspace specific)
            with tc.span("db_persist"):
                mem_id = runtime.db.insert_memory(
                    content, embedding=emb,
                    importance=importance, confidence=confidence,
                )

            # 3. Sync to RAM index (Workspace specific)
            with tc.span("index_sync"):
                now_str = datetime.datetime.now().isoformat()
                runtime.index.add_entry(
                    IndexEntry(
                        id=mem_id, item_type="memory", content=content,
                        importance=importance, confidence=confidence,
                        created_at=now_str, decay_score=1.0, access_count=0,
                    ),
                    emb,
                )

            with tc.span("extract_facts"):
                facts = runtime.extractor.extract_facts(content)

            extracted = []
            if facts:
                if tc.elapsed_ms() < self.deadline_ms:
                    with tc.span("embed_facts", fact_count=len(facts)):
                        fact_texts = [f"{f.subject} {f.relation} {f.object}" for f in facts]
                        try:
                            f_fut = self.global_runtime.embedder_pipeline.embed_batch_async(fact_texts)
                            fact_embs = await asyncio.wrap_future(f_fut)
                        except (RuntimeError, AttributeError):
                            fact_embs = [self.global_runtime.shared_embedder.embed(t) for t in fact_texts]

                    with tc.span("persist_facts"):
                        for f, f_emb in zip(facts, fact_embs):
                            f_id = runtime.db.insert_fact(
                                f.subject, f.relation, f.object,
                                embedding=f_emb,
                                importance=importance, confidence=confidence,
                            )
                            t_str = f"{f.subject} {f.relation} {f.object}"
                            runtime.index.add_entry(
                                IndexEntry(
                                    id=f_id, item_type="fact", content=t_str,
                                    importance=importance, confidence=confidence,
                                    created_at=now_str, decay_score=1.0, access_count=0,
                                ),
                                f_emb,
                            )
                            extracted.append({"id": f_id, "triplet": t_str})

            # 4. Bump generation and invalidate cache
            with tc.span("generation_bump"):
                new_generation = runtime.bump_generation()

            tc.set_item_count(1 + len(extracted))

        self.collector.record(tc.trace)
        
        metadata = ResponseMetadata(
            workspace_id=workspace_id,
            generation=new_generation,
        )
        
        return {
            "id": mem_id,
            "extracted_facts": extracted,
            "metadata": metadata.dict(),
        }

    # ------------------------------------------------------------------
    # search — read path
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
        workspace_id: str = "default",
        client_generation: Optional[int] = None,
    ) -> Dict[str, Any]:
        limit = min(limit, MAX_RETURN_LIMIT)
        runtime = self._get_runtime(workspace_id)

        with TraceContext("search") as tc:
            # Sync cache with current generation
            with tc.span("generation_check"):
                runtime.sync_cache_generation()
                current_gen = runtime.get_generation()
                
                # Detect stale context
                stale_warning = None
                if client_generation is not None and client_generation < current_gen:
                    stale_warning = f"Client context is stale (client: {client_generation}, current: {current_gen})"
                    logger.warning(f"[{workspace_id}] {stale_warning}")

            # Layer 1: Full-result cache
            with tc.span("cache_check"):
                cache_key = (workspace_id, query, limit)
                cached = runtime.cache.search_results.get(cache_key)

            cache_hit = False
            if cached is not None:
                tc.mark_cache_hit()
                tc.set_item_count(len(cached))
                cache_hit = True
                serialized = cached
            else:
                # Layer 2: Embed via global pipeline
                q_vec = None
                if tc.elapsed_ms() < SOFT_LIMIT_MS:
                    with tc.span("embed", query_len=len(query)):
                        try:
                            q_fut = self.global_runtime.embedder_pipeline.embed_async(query)
                            q_vec = await asyncio.wrap_future(q_fut)
                        except (RuntimeError, AttributeError):
                            q_vec = self.global_runtime.shared_embedder.embed(query)
                else:
                    tc.mark_degraded("skipped embedding (latency budget)")

                # Layer 3: RAM index search
                with tc.span("retrieve", index_size=len(runtime.index)):
                    max_candidates = min(limit * 3, MAX_CANDIDATES)
                    results = self._retrieve_with_deadline(
                        tc, runtime, query, q_vec, limit, max_candidates,
                    )

                tc.set_item_count(len(results))

                with tc.span("serialize"):
                    serialized = [item.__dict__ for item in results]
                    runtime.cache.search_results.set(cache_key, serialized)

        self.collector.record(tc.trace)
        
        metadata = ResponseMetadata(
            workspace_id=workspace_id,
            generation=current_gen,
            cache_hit=cache_hit,
            degraded=tc.trace.degraded,
            stale_warning=stale_warning,
        )
        
        return {
            "results": serialized,
            "metadata": metadata.dict(),
        }

    # ------------------------------------------------------------------
    # build_context — compound read path
    # ------------------------------------------------------------------

    async def build_context(
        self,
        query: str,
        max_chars: int = 500,
        threshold: float = 0.3,
        workspace_id: str = "default",
        client_generation: Optional[int] = None,
    ) -> Dict[str, Any]:
        runtime = self._get_runtime(workspace_id)

        with TraceContext("build_context") as tc:
            # Sync cache with current generation
            with tc.span("generation_check"):
                runtime.sync_cache_generation()
                current_gen = runtime.get_generation()
                
                # Detect stale context
                stale_warning = None
                if client_generation is not None and client_generation < current_gen:
                    stale_warning = f"Client context is stale (client: {client_generation}, current: {current_gen})"
                    logger.warning(f"[{workspace_id}] {stale_warning}")

            with tc.span("cache_check"):
                cache_key = (workspace_id, query, max_chars, threshold)
                cached = runtime.cache.contexts.get(cache_key)

            cache_hit = False
            if cached is not None:
                tc.mark_cache_hit()
                cache_hit = True
                context_str = cached
            else:
                q_vec = None
                if tc.elapsed_ms() < SOFT_LIMIT_MS:
                    with tc.span("embed", query_len=len(query)):
                        try:
                            q_fut = self.global_runtime.embedder_pipeline.embed_async(query)
                            q_vec = await asyncio.wrap_future(q_fut)
                        except (RuntimeError, AttributeError):
                            q_vec = self.global_runtime.shared_embedder.embed(query)

                with tc.span("retrieve", index_size=len(runtime.index)):
                    max_candidates = min(MAX_RETURN_LIMIT * 3, MAX_CANDIDATES)
                    all_items = self._retrieve_with_deadline(
                        tc, runtime, query, q_vec, MAX_RETURN_LIMIT, max_candidates,
                    )
                    items = [i for i in all_items if i.score >= threshold]

                tc.set_item_count(len(items))

                conflicts = []
                if tc.elapsed_ms() < SOFT_LIMIT_MS:
                    with tc.span("conflict_resolution"):
                        active_fact_ids = [i.id for i in items if i.item_type == "fact"]
                        conflicts = runtime.db.get_fact_conflicts(active_fact_ids)

                with tc.span("assemble", max_chars=max_chars):
                    runtime.builder.max_chars = max_chars
                    context_str = runtime.builder.build_context(items, conflicts=conflicts)

                with tc.span("cache_set"):
                    runtime.cache.contexts.set(cache_key, context_str)

        self.collector.record(tc.trace)
        
        metadata = ResponseMetadata(
            workspace_id=workspace_id,
            generation=current_gen,
            cache_hit=cache_hit,
            degraded=tc.trace.degraded,
            stale_warning=stale_warning,
        )
        
        return {
            "context": context_str,
            "metadata": metadata.dict(),
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_diagnostics(self, workspace_id: str = "default") -> Dict[str, Any]:
        runtime = self._get_runtime(workspace_id)
        stats = runtime.db.get_stats()
        from memk.core.forgetting import ForgettingEngine
        engine = ForgettingEngine()
        states = runtime.db.get_state_counts(engine.cold_threshold, engine.warm_threshold)
        return {
            "db_stats": stats,
            "memory_health": states,
            "runtime": runtime.get_diagnostics(),
            "global": self.global_runtime.get_diagnostics()["global"]
        }

    def get_tail_latency_report(self) -> Dict[str, Any]:
        return self.collector.get_report()

    def submit_job(self, workspace_id: str, job_type: str, **kwargs) -> str:
        runtime = self._get_runtime(workspace_id)
        if job_type == "synthesize":
            def task():
                from memk.synthesis.synthesizer import KnowledgeSynthesizer
                return KnowledgeSynthesizer(runtime.db).synthesize_all()
            return runtime.jobs.submit(job_type, task)
        return "unsupported"

    # ------------------------------------------------------------------
    # Private: deadline-aware retrieval
    # ------------------------------------------------------------------

    def _retrieve_with_deadline(self, tc, runtime, query, q_vec, limit, max_candidates):
        from memk.retrieval.retriever import RetrievedItem

        if runtime.index and len(runtime.index) > 0:
            with tc.span("index_search", max_candidates=max_candidates):
                if q_vec is not None:
                    index_hits = runtime.index.search(q_vec, top_k=max_candidates)
                else:
                    index_hits = runtime.index.search_lexical(query, top_k=max_candidates)

            if tc.elapsed_ms() > self.deadline_ms:
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

            with tc.span("rank", candidates=len(index_hits)):
                return runtime.retriever.rank_candidates(query, q_vec, index_hits, limit)
        
        return []
