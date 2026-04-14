"""
memk.core.runtime
=================
Singleton RuntimeManager — bootstraps all subsystems.

v0.5: Uses EmbeddingPipeline for cached, pooled embedding with pre-warming.
"""

import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager

from memk.storage.db import MemoryDB
from memk.core.embedder import (
    get_default_embedder, get_default_pipeline,
    BaseEmbedder, EmbeddingPipeline, decode_embedding,
)
from memk.retrieval.retriever import ScoredRetriever
from memk.context.builder import ContextBuilder
from memk.extraction.extractor import RuleBasedExtractor
from memk.retrieval.index import VectorIndex, IndexEntry
from memk.core.cache import MemoryCacheManager
from memk.core.jobs import BackgroundJobManager

logger = logging.getLogger("memk.runtime")


@dataclass
class TelemetryData:
    startup_time_ms: float = 0.0
    model_load_time_ms: float = 0.0
    warmup_complete: bool = False
    db_connected: bool = False
    index_size: int = 0
    last_query_time_ms: float = 0.0
    total_requests: int = 0


class RuntimeManager:
    _instance: Optional['RuntimeManager'] = None

    def __init__(self):
        self.db: Optional[MemoryDB] = None
        self.embedder: Optional[EmbeddingPipeline] = None
        self._raw_embedder: Optional[BaseEmbedder] = None
        self.retriever: Optional[ScoredRetriever] = None
        self.builder: Optional[ContextBuilder] = None
        self.extractor: Optional[RuleBasedExtractor] = None
        self.index: Optional[VectorIndex] = None
        self.cache: Optional[MemoryCacheManager] = None
        self.jobs: Optional[BackgroundJobManager] = None
        self.telemetry = TelemetryData()
        self._is_initialized = False

    @classmethod
    def get_instance(cls) -> 'RuntimeManager':
        if cls._instance is None:
            cls._instance = RuntimeManager()
        return cls._instance

    def initialize(self):
        if self._is_initialized:
            return

        start_all = time.perf_counter()

        # 1. Database
        self.db = MemoryDB()
        self.db.init_db()
        self.telemetry.db_connected = True

        # 2. Embedding Pipeline (model load + pipeline wrapping)
        model_start = time.perf_counter()
        self._raw_embedder = get_default_embedder()
        self.embedder = get_default_pipeline(self._raw_embedder)
        self.telemetry.model_load_time_ms = (time.perf_counter() - model_start) * 1000

        # 3. RAM Vector Index
        self.index = VectorIndex(dim=self._raw_embedder.dim)
        self._hydrate_index()

        # 4. Cache & Jobs
        self.cache = MemoryCacheManager()
        self.jobs = BackgroundJobManager()

        # 5. Model warmup (ensures JIT/lazy-init is done)
        self.embedder.embed("warmup")
        self.telemetry.warmup_complete = True

        # 6. High-level services
        self.retriever = ScoredRetriever(
            self.db, embedder=self._raw_embedder,
            index=self.index, cache=self.cache,
        )
        self.builder = ContextBuilder()
        self.extractor = RuleBasedExtractor()

        # 7. Pre-warm common query embeddings
        self._prewarm_common_queries()

        self.telemetry.startup_time_ms = (time.perf_counter() - start_all) * 1000
        self._is_initialized = True
        logger.info(
            f"Runtime READY in {self.telemetry.startup_time_ms:.0f}ms. "
            f"Index: {self.telemetry.index_size} entries. "
            f"Pipeline: pooled + cached."
        )

    def _hydrate_index(self):
        """Load all existing embeddings from DB into the RAM index."""
        facts = self.db.get_all_active_facts()
        for r in facts:
            if r["embedding"]:
                entry = IndexEntry(
                    id=r["id"], item_type="fact",
                    content=f"{r['subject']} {r['predicate']} {r['object']}",
                    importance=float(r.get("importance", 0.5)),
                    confidence=float(r.get("confidence", 1.0)),
                    created_at=r["created_at"],
                    decay_score=float(r.get("decay_score", 1.0)),
                    access_count=int(r.get("access_count", 0)),
                )
                self.index.add_entry(entry, decode_embedding(r["embedding"]))

        mems = self.db.get_all_memories()
        for r in mems:
            if r["embedding"]:
                entry = IndexEntry(
                    id=r["id"], item_type="memory", content=r["content"],
                    importance=float(r.get("importance", 0.5)),
                    confidence=float(r.get("confidence", 1.0)),
                    created_at=r["created_at"],
                    decay_score=float(r.get("decay_score", 1.0)),
                    access_count=int(r.get("access_count", 0)),
                )
                self.index.add_entry(entry, decode_embedding(r["embedding"]))

        self.telemetry.index_size = len(self.index)

    def _prewarm_common_queries(self):
        """
        Pre-warm the embedding cache with common query patterns.
        These are typical queries AI agents make repeatedly.
        """
        common_queries = [
            "performance", "latency", "system", "architecture",
            "memory", "facts", "context", "user preferences",
            "AI agents", "critical", "embedding", "search",
        ]
        self.embedder.prewarm(common_queries)

    def get_diagnostics(self) -> Dict[str, Any]:
        result = {
            "initialized": self._is_initialized,
            "index_entries": len(self.index) if self.index else 0,
            "cache": self.cache.get_stats() if self.cache else {},
            "active_jobs": 0,
            "telemetry": asdict(self.telemetry),
        }
        if self.jobs:
            result["active_jobs"] = len([
                j for j in self.jobs.jobs.values() if j.status == "running"
            ])
        if self.embedder:
            result["embedding_pipeline"] = self.embedder.get_telemetry()
        return result

    @contextmanager
    def track_latency(self):
        start = time.perf_counter()
        self.telemetry.total_requests += 1
        try:
            yield
        finally:
            self.telemetry.last_query_time_ms = (time.perf_counter() - start) * 1000


def get_runtime() -> RuntimeManager:
    return RuntimeManager.get_instance()
