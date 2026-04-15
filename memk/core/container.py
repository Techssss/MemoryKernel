"""
memk.core.container
===================
Dependency Injection Container for MemoryKernel.

Provides centralized dependency management with:
- Singleton pattern for shared resources
- Factory pattern for per-workspace instances
- Lazy initialization
- Easy testing with mock injection
"""

import logging
from typing import Optional, Dict, Any, Callable, TypeVar
from dataclasses import dataclass
from threading import Lock

from memk.core.protocols import (
    EmbedderProtocol,
    StorageProtocol,
    IndexProtocol,
    CacheProtocol,
    RetrieverProtocol,
    ContextBuilderProtocol,
    ExtractorProtocol,
    JobManagerProtocol,
    WorkspaceManagerProtocol,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


# ---------------------------------------------------------------------------
# Container Configuration
# ---------------------------------------------------------------------------

@dataclass
class ContainerConfig:
    """Configuration for dependency injection."""
    
    # Embedder settings
    embedder_model: str = "all-MiniLM-L6-v2"
    embedder_dim: int = 384
    
    # Cache settings
    cache_maxsize: int = 100
    cache_ttl_seconds: int = 3600
    
    # Job settings
    max_workers: int = 2
    
    # Storage settings
    db_path: Optional[str] = None
    
    # Performance settings
    enable_pipeline: bool = True
    pipeline_batch_size: int = 32


# ---------------------------------------------------------------------------
# Dependency Container
# ---------------------------------------------------------------------------

class DependencyContainer:
    """
    Centralized dependency injection container.
    
    Manages:
    - Singleton instances (shared across workspaces)
    - Factory functions (per-workspace instances)
    - Lazy initialization
    - Thread-safe access
    """
    
    def __init__(self, config: Optional[ContainerConfig] = None):
        self.config = config or ContainerConfig()
        self._lock = Lock()
        
        # Singleton instances (shared)
        self._embedder: Optional[EmbedderProtocol] = None
        self._embedder_pipeline: Optional[Any] = None
        
        # Factory functions (registered)
        self._factories: Dict[str, Callable] = {}
        
        # Per-workspace instances cache
        self._workspace_instances: Dict[str, Dict[str, Any]] = {}
        
        # Register default factories
        self._register_default_factories()
    
    # -----------------------------------------------------------------------
    # Singleton Management (Shared Resources)
    # -----------------------------------------------------------------------
    
    def get_embedder(self) -> EmbedderProtocol:
        """Get or create shared embedder (singleton)."""
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    logger.info(f"Initializing embedder: {self.config.embedder_model}")
                    from memk.core.embedder import get_default_embedder
                    self._embedder = get_default_embedder()
        return self._embedder
    
    def get_embedder_pipeline(self) -> Any:
        """Get or create shared embedder pipeline (singleton)."""
        if self._embedder_pipeline is None:
            with self._lock:
                if self._embedder_pipeline is None:
                    if self.config.enable_pipeline:
                        logger.info("Initializing embedder pipeline")
                        from memk.core.embedder import get_default_pipeline
                        self._embedder_pipeline = get_default_pipeline(
                            self.get_embedder()
                        )
                    else:
                        self._embedder_pipeline = None
        return self._embedder_pipeline
    
    def set_embedder(self, embedder: EmbedderProtocol) -> None:
        """Override embedder (useful for testing)."""
        with self._lock:
            self._embedder = embedder
    
    # -----------------------------------------------------------------------
    # Factory Registration
    # -----------------------------------------------------------------------
    
    def register_factory(self, name: str, factory: Callable) -> None:
        """Register a factory function for creating instances."""
        self._factories[name] = factory
    
    def _register_default_factories(self) -> None:
        """Register default factory functions."""
        
        # Storage factory
        def storage_factory(workspace_id: str, db_path: str) -> StorageProtocol:
            from memk.storage.db import MemoryDB
            db = MemoryDB(db_path=db_path)
            db.init_db()
            return db
        
        # Index factory
        def index_factory(workspace_id: str) -> IndexProtocol:
            from memk.retrieval.index import VectorIndex
            return VectorIndex(dim=self.config.embedder_dim)
        
        # Cache factory
        def cache_factory(workspace_id: str) -> CacheProtocol:
            from memk.core.cache import MemoryCacheManager
            return MemoryCacheManager()
        
        # Retriever factory
        def retriever_factory(
            workspace_id: str,
            storage: StorageProtocol,
            index: IndexProtocol,
            cache: CacheProtocol
        ) -> RetrieverProtocol:
            from memk.retrieval.retriever import ScoredRetriever
            return ScoredRetriever(
                db=storage,
                embedder=self.get_embedder(),
                index=index,
                cache=cache,
            )
        
        # Context builder factory
        def builder_factory(workspace_id: str) -> ContextBuilderProtocol:
            from memk.context.builder import ContextBuilder
            return ContextBuilder()
        
        # Extractor factory
        def extractor_factory(workspace_id: str) -> ExtractorProtocol:
            from memk.extraction.extractor import RuleBasedExtractor
            return RuleBasedExtractor()
        
        # Job manager factory
        def job_manager_factory(workspace_id: str) -> JobManagerProtocol:
            from memk.core.jobs import BackgroundJobManager
            return BackgroundJobManager(max_workers=self.config.max_workers)
        
        # Register all factories
        self.register_factory("storage", storage_factory)
        self.register_factory("index", index_factory)
        self.register_factory("cache", cache_factory)
        self.register_factory("retriever", retriever_factory)
        self.register_factory("builder", builder_factory)
        self.register_factory("extractor", extractor_factory)
        self.register_factory("job_manager", job_manager_factory)
    
    # -----------------------------------------------------------------------
    # Per-Workspace Instance Management
    # -----------------------------------------------------------------------
    
    def get_workspace_instance(
        self,
        workspace_id: str,
        component: str,
        **kwargs
    ) -> Any:
        """
        Get or create a workspace-specific instance.
        
        Args:
            workspace_id: Workspace identifier
            component: Component name (must have registered factory)
            **kwargs: Additional arguments for factory
        
        Returns:
            Component instance for the workspace
        """
        # Initialize workspace cache if needed
        if workspace_id not in self._workspace_instances:
            self._workspace_instances[workspace_id] = {}
        
        workspace_cache = self._workspace_instances[workspace_id]
        
        # Return cached instance if exists
        if component in workspace_cache:
            return workspace_cache[component]
        
        # Create new instance using factory
        if component not in self._factories:
            raise ValueError(f"No factory registered for component: {component}")
        
        with self._lock:
            # Double-check after acquiring lock
            if component in workspace_cache:
                return workspace_cache[component]
            
            factory = self._factories[component]
            
            # Special handling for retriever (needs other components)
            if component == "retriever":
                storage = self.get_workspace_instance(workspace_id, "storage", **kwargs)
                index = self.get_workspace_instance(workspace_id, "index")
                cache = self.get_workspace_instance(workspace_id, "cache")
                instance = factory(workspace_id, storage, index, cache)
            else:
                instance = factory(workspace_id, **kwargs)
            
            workspace_cache[component] = instance
            logger.debug(f"Created {component} for workspace: {workspace_id}")
            
            return instance
    
    def clear_workspace(self, workspace_id: str) -> None:
        """Clear all cached instances for a workspace."""
        if workspace_id in self._workspace_instances:
            del self._workspace_instances[workspace_id]
            logger.info(f"Cleared workspace instances: {workspace_id}")
    
    def get_workspace_diagnostics(self, workspace_id: str) -> Dict[str, Any]:
        """Get diagnostics for a workspace."""
        if workspace_id not in self._workspace_instances:
            return {"status": "not_loaded"}
        
        components = list(self._workspace_instances[workspace_id].keys())
        return {
            "status": "loaded",
            "components": components,
            "component_count": len(components),
        }
    
    # -----------------------------------------------------------------------
    # Global Diagnostics
    # -----------------------------------------------------------------------
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get container diagnostics."""
        return {
            "config": {
                "embedder_model": self.config.embedder_model,
                "embedder_dim": self.config.embedder_dim,
                "cache_maxsize": self.config.cache_maxsize,
                "max_workers": self.config.max_workers,
            },
            "singletons": {
                "embedder_loaded": self._embedder is not None,
                "pipeline_loaded": self._embedder_pipeline is not None,
            },
            "workspaces": {
                "active_count": len(self._workspace_instances),
                "workspace_ids": list(self._workspace_instances.keys()),
            },
            "factories": {
                "registered": list(self._factories.keys()),
            },
        }


# ---------------------------------------------------------------------------
# Global Container Instance
# ---------------------------------------------------------------------------

_global_container: Optional[DependencyContainer] = None
_container_lock = Lock()


def get_container(config: Optional[ContainerConfig] = None) -> DependencyContainer:
    """
    Get the global dependency container (singleton).
    
    Args:
        config: Optional configuration (only used on first call)
    
    Returns:
        Global DependencyContainer instance
    """
    global _global_container
    
    if _global_container is None:
        with _container_lock:
            if _global_container is None:
                _global_container = DependencyContainer(config)
                logger.info("Initialized global dependency container")
    
    return _global_container


def reset_container() -> None:
    """Reset the global container (useful for testing)."""
    global _global_container
    with _container_lock:
        _global_container = None
        logger.info("Reset global dependency container")
