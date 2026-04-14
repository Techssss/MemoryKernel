import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("memk.index")

@dataclass
class IndexEntry:
    id: str
    item_type: str  # "memory" or "fact"
    content: str
    importance: float
    confidence: float
    created_at: str
    decay_score: float
    access_count: int

class VectorIndex:
    """
    In-memory vector and metadata index for sub-millisecond similarity search.
    SQLite remains the source of truth; this index is a 'hot' cache.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vectors: Optional[np.ndarray] = None  # Shape: (N, dim)
        self.metadata: List[IndexEntry] = []
        self._id_to_idx: Dict[str, int] = {}

    def clear(self):
        self.vectors = None
        self.metadata = []
        self._id_to_idx = {}

    def add_entry(self, entry: IndexEntry, vector: np.ndarray):
        """Append a new entry to the in-memory index."""
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dim mismatch: expected {self.dim}, got {vector.shape[0]}")

        # Ensure normalized
        norm = np.linalg.norm(vector)
        if norm > 1e-10:
            vector = vector / norm

        if self.vectors is None:
            self.vectors = vector.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vector.reshape(1, -1)])
        
        self._id_to_idx[entry.id] = len(self.metadata)
        self.metadata.append(entry)

    def search(self, query_vec: np.ndarray, top_k: int = 50) -> List[tuple[IndexEntry, float]]:
        """Perform exhaustive (brute force) cosine similarity in RAM using NumPy."""
        if self.vectors is None or len(self.metadata) == 0:
            return []

        # Ensure query is normalized
        norm = np.linalg.norm(query_vec)
        if norm > 1e-10:
            query_vec = query_vec / norm

        # Compute cosine similarities via dot product: (N, D) @ (D, 1) -> (N, 1)
        # Since both are normalized, dot product == cosine similarity
        similarities = np.dot(self.vectors, query_vec)
        
        # Sort and get top-k
        indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in indices:
            # Map [-1, 1] similarity to [0, 1]
            score = (similarities[idx] + 1.0) / 2.0
            results.append((self.metadata[idx], score))
            
        return results

    def search_lexical(self, query: str, top_k: int = 50) -> List[tuple[IndexEntry, float]]:
        """
        Perform fast keyword-based search in RAM metadata. 
        Zero vector cost, purely string matching.
        """
        if not self.metadata or not query:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        for entry in self.metadata:
            content_lower = entry.content.lower()
            
            # Simple scoring: 0.5 base for any word match, up to 1.0 for full phrase
            score = 0.0
            if query_lower in content_lower:
                score = 0.9  # Exact phrase match
            else:
                matches = sum(1 for word in query_words if word in content_lower)
                if matches > 0:
                    score = 0.5 + (0.3 * (matches / len(query_words)))
            
            if score > 0:
                results.append((entry, score))
        
        # Sort by score and recency
        results.sort(key=lambda x: (x[1], x[0].created_at), reverse=True)
        return results[:top_k]

    def update_entry_metadata(self, entry_id: str, **updates):
        """Update hot metadata (e.g., access_count, decay_score) directly in RAM."""
        if entry_id in self._id_to_idx:
            idx = self._id_to_idx[entry_id]
            entry = self.metadata[idx]
            for key, val in updates.items():
                if hasattr(entry, key):
                    setattr(entry, key, val)

    def __len__(self):
        return len(self.metadata)
