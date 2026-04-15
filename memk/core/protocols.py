"""
memk.core.protocols
===================
Protocol definitions for core interfaces.
Enables type-safe dependency injection and easy mocking.
"""

from typing import Protocol, List, Optional, Dict, Any, Tuple
import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# Embedder Protocol
# ---------------------------------------------------------------------------

class EmbedderProtocol(Protocol):
    """Protocol for embedding models."""
    
    @property
    def dim(self) -> int:
        """Embedding dimension."""
        ...
    
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        ...
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed multiple texts."""
        ...


# ---------------------------------------------------------------------------
# Storage Protocol
# ---------------------------------------------------------------------------

class StorageProtocol(Protocol):
    """Protocol for storage backends."""
    
    def init_db(self) -> None:
        """Initialize database schema."""
        ...
    
    def insert_memory(
        self,
        content: str,
        *,
        embedding: Optional[np.ndarray] = None,
        importance: float = 0.5,
        confidence: float = 1.0,
    ) -> str:
        """Insert a memory and return its ID."""
        ...
    
    def insert_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        confidence: float = 1.0,
        importance: float = 0.5,
        embedding: Optional[np.ndarray] = None,
    ) -> str:
        """Insert a fact and return its ID."""
        ...
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Get all memories."""
        ...
    
    def get_all_active_facts(self) -> List[Dict[str, Any]]:
        """Get all active facts."""
        ...
    
    def search_memory(self, keyword: str) -> List[Dict[str, Any]]:
        """Search memories by keyword."""
        ...
    
    def search_facts(
        self,
        subject: Optional[str] = None,
        keyword: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search facts."""
        ...
    
    def touch_memory(self, mem_id: str) -> None:
        """Update access tracking for memory."""
        ...
    
    def touch_fact(self, fact_id: str) -> None:
        """Update access tracking for fact."""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        ...
    
    def get_fact_conflicts(self, active_fact_ids: List[str]) -> List[Dict[str, Any]]:
        """Get conflicting facts."""
        ...
    
    def get_state_counts(self, cold_th: float, warm_th: float) -> Dict[str, int]:
        """Get memory health state counts."""
        ...


# ---------------------------------------------------------------------------
# Index Protocol
# ---------------------------------------------------------------------------

class IndexProtocol(Protocol):
    """Protocol for vector indexes."""
    
    def __len__(self) -> int:
        """Number of entries in index."""
        ...
    
    def add_entry(self, entry: Any, vector: np.ndarray) -> None:
        """Add an entry with its vector."""
        ...
    
    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[Any, float]]:
        """Search for similar vectors."""
        ...
    
    def search_lexical(self, query: str, top_k: int = 10) -> List[Tuple[Any, float]]:
        """Lexical search fallback."""
        ...
    
    def clear(self) -> None:
        """Clear all entries."""
        ...


# ---------------------------------------------------------------------------
# Cache Protocol
# ---------------------------------------------------------------------------

class CacheProtocol(Protocol):
    """Protocol for cache managers."""
    
    def set_generation(self, generation: int) -> None:
        """Set current generation and invalidate if changed."""
        ...
    
    def get_generation(self) -> int:
        """Get cached generation."""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...


# ---------------------------------------------------------------------------
# Retriever Protocol
# ---------------------------------------------------------------------------

class RetrieverProtocol(Protocol):
    """Protocol for retrieval strategies."""
    
    def retrieve(self, query: str, limit: int = 10) -> List[Any]:
        """Retrieve relevant items."""
        ...
    
    def rank_candidates(
        self,
        query: str,
        q_vec: np.ndarray,
        index_hits: List[Tuple],
        limit: int
    ) -> List[Any]:
        """Rank candidate items."""
        ...


# ---------------------------------------------------------------------------
# Context Builder Protocol
# ---------------------------------------------------------------------------

class ContextBuilderProtocol(Protocol):
    """Protocol for context builders."""
    
    def build_context(
        self,
        items: List[Any],
        conflicts: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Build context string from items."""
        ...


# ---------------------------------------------------------------------------
# Extractor Protocol
# ---------------------------------------------------------------------------

class ExtractorProtocol(Protocol):
    """Protocol for fact extractors."""
    
    def extract_facts(self, text: str) -> List[Any]:
        """Extract facts from text."""
        ...


# ---------------------------------------------------------------------------
# Job Manager Protocol
# ---------------------------------------------------------------------------

class JobManagerProtocol(Protocol):
    """Protocol for background job managers."""
    
    def submit(self, job_type: str, task_fn) -> str:
        """Submit a background job."""
        ...
    
    @property
    def jobs(self) -> Dict[str, Any]:
        """Get all jobs."""
        ...


# ---------------------------------------------------------------------------
# Workspace Manager Protocol
# ---------------------------------------------------------------------------

class WorkspaceManagerProtocol(Protocol):
    """Protocol for workspace managers."""
    
    def is_initialized(self) -> bool:
        """Check if workspace is initialized."""
        ...
    
    def get_db_path(self) -> str:
        """Get database path."""
        ...
    
    def get_generation(self) -> int:
        """Get current generation."""
        ...
    
    def bump_generation(self) -> int:
        """Increment generation."""
        ...
