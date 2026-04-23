"""
memk.core.runtime
=================
Architecture for project-scoped runtimes with a shared global manager.

GlobalRuntime: Shared resources (Embedding model, etc.)
WorkspaceRuntime: Per-project state (DB, Index, Cache, Jobs, Generation tracking)
"""

import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import os

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
from memk.storage.graph_repository import GraphRepository

logger = logging.getLogger("memk.runtime")

@dataclass
class TelemetryData:
    startup_time_ms: float = 0.0
    db_connected: bool = False
    index_size: int = 0
    total_requests: int = 0

class WorkspaceRuntime:
    """Encapsulates the isolated state of a single project/brain with generation tracking."""
    
    def __init__(self, workspace_id: str, db_path: str, embedder: BaseEmbedder, workspace_manager=None):
        self.workspace_id = workspace_id
        self.db_path = db_path
        self._raw_embedder = embedder
        self.workspace_manager = workspace_manager
        
        self.db: MemoryDB = MemoryDB(db_path=db_path)
        self.db.init_db()
        
        self.index = VectorIndex(dim=embedder.dim)
        self.cache = MemoryCacheManager()
        self.jobs = BackgroundJobManager()
        
        # High-level services (linked to this workspace's state)
        self.retriever = ScoredRetriever(
            self.db, embedder=self._raw_embedder,
            index=self.index, cache=self.cache,
        )
        self.builder = ContextBuilder()
        self.extractor = self._create_extractor()
        
        # Graph sidecar — safe init (None if tables don't exist)
        self.graph_repo = self._create_graph_repo()
        
        self.telemetry = TelemetryData()
        self.last_active = time.time()
        
        # Initialize cache with current generation
        if self.workspace_manager:
            current_gen = self.workspace_manager.get_generation()
            self.cache.set_generation(current_gen)
        
        self._hydrate_index()

    @staticmethod
    def _create_extractor():
        """Create best available extractor: SpaCyExtractor > RuleBasedExtractor."""
        try:
            from memk.extraction.spacy_extractor import SpaCyExtractor
            ext = SpaCyExtractor()
            # Probe model availability without full load
            if ext._ensure_model():
                logger.info("Using SpaCyExtractor for fact extraction")
                return ext
        except ImportError:
            pass
        logger.info("Falling back to RuleBasedExtractor")
        return RuleBasedExtractor()

    def _create_graph_repo(self):
        """Create GraphRepository if V5 schema tables exist, otherwise None."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            conn.close()
            if "entity" in tables and "edge" in tables and "mention" in tables:
                logger.info(f"[{self.workspace_id}] Graph repository initialized")
                return GraphRepository(self.db_path)
        except Exception as e:
            logger.debug(f"[{self.workspace_id}] Graph repo init skipped: {e}")
        return None

    def get_generation(self) -> int:
        """Get current generation from workspace manifest."""
        if self.workspace_manager:
            return self.workspace_manager.get_generation()
        return 0

    def bump_generation(self) -> int:
        """
        Increment generation and invalidate caches.
        Called after any write operation that changes knowledge state.
        """
        if self.workspace_manager:
            new_gen = self.workspace_manager.bump_generation()
            self.cache.set_generation(new_gen)
            logger.info(f"[{self.workspace_id}] Generation bumped to {new_gen}")
            return new_gen
        return 0

    def sync_cache_generation(self):
        """
        Ensure cache is synced with current workspace generation.
        Called at the start of read operations to detect stale cache.
        """
        if self.workspace_manager:
            current_gen = self.workspace_manager.get_generation()
            self.cache.set_generation(current_gen)

    def _hydrate_index(self):
        """Load project-specific embeddings into the isolated RAM index."""
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

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "generation": self.get_generation(),
            "index_entries": len(self.index),
            "cache": self.cache.get_stats(),
            "active_jobs": len([j for j in self.jobs.jobs.values() if j.status == "running"]),
            "telemetry": asdict(self.telemetry),
        }

class RuntimeManager:
    """
    Global Singleton that manages shared infrastructure and multiple WorkspaceRuntimes.
    """
    _instance: Optional['RuntimeManager'] = None

    def __init__(self):
        self.shared_embedder: Optional[BaseEmbedder] = None
        self.embedder_pipeline: Optional[EmbeddingPipeline] = None
        self.workspaces: Dict[str, WorkspaceRuntime] = {}
        self._is_global_initialized = False
        
        self.global_telemetry = {
            "model_load_time_ms": 0.0,
            "total_workspaces_active": 0,
        }

    @classmethod
    def get_instance(cls) -> 'RuntimeManager':
        if cls._instance is None:
            cls._instance = RuntimeManager()
        return cls._instance

    def initialize_global(self):
        """Load the shared embedding model (the heavy part)."""
        if self._is_global_initialized:
            return

        start = time.perf_counter()
        self.shared_embedder = get_default_embedder()
        self.embedder_pipeline = get_default_pipeline(self.shared_embedder)
        
        # Warmup
        self.embedder_pipeline.embed("warmup")
        
        self.global_telemetry["model_load_time_ms"] = (time.perf_counter() - start) * 1000
        self._is_global_initialized = True
        logger.info(f"Global Infrastructure READY. Shared model loaded in {self.global_telemetry['model_load_time_ms']:.0f}ms.")

    def get_workspace_runtime(self, workspace_id: str, db_path: Optional[str] = None) -> WorkspaceRuntime:
        """Fetch or activate a workspace runtime."""
        self.initialize_global()
        
        if workspace_id in self.workspaces:
            runtime = self.workspaces[workspace_id]
            runtime.last_active = time.time()
            return runtime
        
        # Activate new workspace
        from memk.workspace.manager import WorkspaceManager
        # If db_path not provided, try to resolve from CWD or use default naming
        workspace_manager = None
        if not db_path:
            ws_mgr = WorkspaceManager()
            if ws_mgr.is_initialized():
                db_path = ws_mgr.get_db_path()
                workspace_manager = ws_mgr
            else:
                # Fallback / Local-only fallback
                db_path = "mem.db"

        logger.info(f"Activating workspace runtime: {workspace_id} (Path: {db_path})")
        runtime = WorkspaceRuntime(workspace_id, db_path, self.shared_embedder, workspace_manager)
        self.workspaces[workspace_id] = runtime
        self.global_telemetry["total_workspaces_active"] = len(self.workspaces)
        return runtime

    def evict_idle_workspaces(self, idle_seconds: int = 3600):
        """Unload workspaces that haven't been used recently to free RAM index/cache."""
        now = time.time()
        to_remove = []
        for wid, runtime in self.workspaces.items():
            if now - runtime.last_active > idle_seconds:
                to_remove.append(wid)
        
        for wid in to_remove:
            logger.info(f"Evicting idle workspace: {wid}")
            del self.workspaces[wid]
        self.global_telemetry["total_workspaces_active"] = len(self.workspaces)

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "global": self.global_telemetry,
            "active_workspaces": {wid: r.get_diagnostics() for wid, r in self.workspaces.items()}
        }

def get_runtime() -> RuntimeManager:
    return RuntimeManager.get_instance()
