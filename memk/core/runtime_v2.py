"""
memk.core.runtime_v2
====================
Refactored runtime with Dependency Injection.

Key improvements:
- Uses DI container for all dependencies
- Cleaner separation of concerns
- Easier testing with mock injection
- Better resource management
"""

import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

from memk.core.container import get_container, DependencyContainer
from memk.core.protocols import (
    StorageProtocol,
    IndexProtocol,
    CacheProtocol,
    RetrieverProtocol,
    ContextBuilderProtocol,
    ExtractorProtocol,
    JobManagerProtocol,
    WorkspaceManagerProtocol,
)

logger = logging.getLogger("memk.runtime_v2")


@dataclass
class TelemetryData:
    """Runtime telemetry data."""
    startup_time_ms: float = 0.0
    db_connected: bool = False
    index_size: int = 0
    total_requests: int = 0


# ---------------------------------------------------------------------------
# Workspace Runtime (DI-based)
# ---------------------------------------------------------------------------

class WorkspaceRuntimeV2:
    """
    Workspace runtime with dependency injection.
    
    All dependencies are injected via the container, making this
    class easier to test and more flexible.
    """
    
    def __init__(
        self,
        workspace_id: str,
        container: DependencyContainer,
        workspace_manager: Optional[WorkspaceManagerProtocol] = None,
    ):
        self.workspace_id = workspace_id
        self.container = container
        self.workspace_manager = workspace_manager
        
        # Lazy-loaded components (injected via container)
        self._db: Optional[StorageProtocol] = None
        self._index: Optional[IndexProtocol] = None
        self._cache: Optional[CacheProtocol] = None
        self._retriever: Optional[RetrieverProtocol] = None
        self._builder: Optional[ContextBuilderProtocol] = None
        self._extractor: Optional[ExtractorProtocol] = None
        self._jobs: Optional[JobManagerProtocol] = None
        
        self.telemetry = TelemetryData()
        self.last_active = time.time()
        self._initialized = False
    
    # -----------------------------------------------------------------------
    # Property-based Lazy Loading
    # -----------------------------------------------------------------------
    
    @property
    def db(self) -> StorageProtocol:
        """Get storage instance (lazy-loaded)."""
        if self._db is None:
            db_path = self._resolve_db_path()
            self._db = self.container.get_workspace_instance(
                self.workspace_id,
                "storage",
                db_path=db_path
            )
            self.telemetry.db_connected = True
            # Initialize on first access
            if not self._initialized:
                self._initialize()
        return self._db
    
    @property
    def index(self) -> IndexProtocol:
        """Get index instance (lazy-loaded)."""
        if self._index is None:
            self._index = self.container.get_workspace_instance(
                self.workspace_id,
                "index"
            )
        return self._index
    
    @property
    def cache(self) -> CacheProtocol:
        """Get cache instance (lazy-loaded)."""
        if self._cache is None:
            self._cache = self.container.get_workspace_instance(
                self.workspace_id,
                "cache"
            )
            # Initialize with current generation
            if self.workspace_manager:
                current_gen = self.workspace_manager.get_generation()
                self._cache.set_generation(current_gen)
        return self._cache
    
    @property
    def retriever(self) -> RetrieverProtocol:
        """Get retriever instance (lazy-loaded)."""
        if self._retriever is None:
            self._retriever = self.container.get_workspace_instance(
                self.workspace_id,
                "retriever",
                db_path=self._resolve_db_path()
            )
        return self._retriever
    
    @property
    def builder(self) -> ContextBuilderProtocol:
        """Get context builder instance (lazy-loaded)."""
        if self._builder is None:
            self._builder = self.container.get_workspace_instance(
                self.workspace_id,
                "builder"
            )
        return self._builder
    
    @property
    def extractor(self) -> ExtractorProtocol:
        """Get fact extractor instance (lazy-loaded)."""
        if self._extractor is None:
            self._extractor = self.container.get_workspace_instance(
                self.workspace_id,
                "extractor"
            )
        return self._extractor
    
    @property
    def jobs(self) -> JobManagerProtocol:
        """Get job manager instance (lazy-loaded)."""
        if self._jobs is None:
            self._jobs = self.container.get_workspace_instance(
                self.workspace_id,
                "job_manager"
            )
        return self._jobs
    
    # -----------------------------------------------------------------------
    # Generation Management
    # -----------------------------------------------------------------------
    
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
    
    # -----------------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------------
    
    def _initialize(self):
        """Initialize runtime and hydrate index."""
        if self._initialized:
            return
        
        start = time.perf_counter()
        
        # Hydrate index from storage (db already loaded at this point)
        self._hydrate_index()
        
        self._initialized = True
        self.telemetry.startup_time_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"[{self.workspace_id}] Runtime initialized in "
            f"{self.telemetry.startup_time_ms:.0f}ms"
        )
    
    def _hydrate_index(self):
        """Load embeddings from storage into RAM index."""
        from memk.retrieval.index import IndexEntry
        from memk.core.embedder import decode_embedding
        
        # Load facts
        facts = self.db.get_all_active_facts()
        for r in facts:
            if r["embedding"]:
                entry = IndexEntry(
                    id=r["id"],
                    item_type="fact",
                    content=f"{r['subject']} {r['predicate']} {r['object']}",
                    importance=float(r.get("importance", 0.5)),
                    confidence=float(r.get("confidence", 1.0)),
                    created_at=r["created_at"],
                    decay_score=float(r.get("decay_score", 1.0)),
                    access_count=int(r.get("access_count", 0)),
                )
                self.index.add_entry(entry, decode_embedding(r["embedding"]))
        
        # Load memories
        mems = self.db.get_all_memories()
        for r in mems:
            if r["embedding"]:
                entry = IndexEntry(
                    id=r["id"],
                    item_type="memory",
                    content=r["content"],
                    importance=float(r.get("importance", 0.5)),
                    confidence=float(r.get("confidence", 1.0)),
                    created_at=r["created_at"],
                    decay_score=float(r.get("decay_score", 1.0)),
                    access_count=int(r.get("access_count", 0)),
                )
                self.index.add_entry(entry, decode_embedding(r["embedding"]))
        
        self.telemetry.index_size = len(self.index)
        logger.info(f"[{self.workspace_id}] Hydrated index: {self.telemetry.index_size} entries")
    
    def _resolve_db_path(self) -> str:
        """Resolve database path for this workspace."""
        if self.workspace_manager and self.workspace_manager.is_initialized():
            return self.workspace_manager.get_db_path()
        return "mem.db"  # Fallback
    
    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get runtime diagnostics."""
        return {
            "workspace_id": self.workspace_id,
            "generation": self.get_generation(),
            "index_entries": len(self.index) if self._index else 0,
            "cache": self.cache.get_stats() if self._cache else {},
            "active_jobs": len([
                j for j in self.jobs.jobs.values()
                if j.status == "running"
            ]) if self._jobs else 0,
            "telemetry": asdict(self.telemetry),
            "components_loaded": {
                "db": self._db is not None,
                "index": self._index is not None,
                "cache": self._cache is not None,
                "retriever": self._retriever is not None,
                "builder": self._builder is not None,
                "extractor": self._extractor is not None,
                "jobs": self._jobs is not None,
            },
        }


# ---------------------------------------------------------------------------
# Runtime Manager (DI-based)
# ---------------------------------------------------------------------------

class RuntimeManagerV2:
    """
    Global runtime manager with dependency injection.
    
    Manages multiple workspace runtimes using a shared DI container.
    """
    
    _instance: Optional['RuntimeManagerV2'] = None
    
    def __init__(self, container: Optional[DependencyContainer] = None):
        self.container = container or get_container()
        self.workspaces: Dict[str, WorkspaceRuntimeV2] = {}
        
        self.global_telemetry = {
            "model_load_time_ms": 0.0,
            "total_workspaces_active": 0,
        }
    
    @classmethod
    def get_instance(cls, container: Optional[DependencyContainer] = None) -> 'RuntimeManagerV2':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = RuntimeManagerV2(container)
        return cls._instance
    
    def initialize_global(self):
        """Initialize shared resources (embedder, etc.)."""
        start = time.perf_counter()
        
        # Trigger embedder initialization
        embedder = self.container.get_embedder()
        pipeline = self.container.get_embedder_pipeline()
        
        # Warmup
        if pipeline:
            pipeline.embed("warmup")
        else:
            embedder.embed("warmup")
        
        self.global_telemetry["model_load_time_ms"] = (time.perf_counter() - start) * 1000
        logger.info(
            f"Global infrastructure ready. Model loaded in "
            f"{self.global_telemetry['model_load_time_ms']:.0f}ms"
        )
    
    def get_workspace_runtime(
        self,
        workspace_id: str,
        workspace_manager: Optional[WorkspaceManagerProtocol] = None
    ) -> WorkspaceRuntimeV2:
        """Get or create workspace runtime."""
        self.initialize_global()
        
        if workspace_id in self.workspaces:
            runtime = self.workspaces[workspace_id]
            runtime.last_active = time.time()
            return runtime
        
        # Create new workspace runtime
        logger.info(f"Activating workspace runtime: {workspace_id}")
        
        # Try to resolve workspace manager if not provided
        if workspace_manager is None:
            try:
                from memk.workspace.manager import WorkspaceManager
                ws_mgr = WorkspaceManager()
                if ws_mgr.is_initialized():
                    workspace_manager = ws_mgr
            except Exception as e:
                logger.warning(f"Could not load workspace manager: {e}")
        
        runtime = WorkspaceRuntimeV2(
            workspace_id,
            self.container,
            workspace_manager
        )
        
        self.workspaces[workspace_id] = runtime
        self.global_telemetry["total_workspaces_active"] = len(self.workspaces)
        
        return runtime
    
    def evict_idle_workspaces(self, idle_seconds: int = 3600):
        """Unload idle workspaces to free resources."""
        now = time.time()
        to_remove = []
        
        for wid, runtime in self.workspaces.items():
            if now - runtime.last_active > idle_seconds:
                to_remove.append(wid)
        
        for wid in to_remove:
            logger.info(f"Evicting idle workspace: {wid}")
            self.container.clear_workspace(wid)
            del self.workspaces[wid]
        
        self.global_telemetry["total_workspaces_active"] = len(self.workspaces)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get global diagnostics."""
        return {
            "global": self.global_telemetry,
            "container": self.container.get_diagnostics(),
            "active_workspaces": {
                wid: r.get_diagnostics()
                for wid, r in self.workspaces.items()
            },
        }


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def get_runtime_v2(container: Optional[DependencyContainer] = None) -> RuntimeManagerV2:
    """Get the global runtime manager."""
    return RuntimeManagerV2.get_instance(container)
