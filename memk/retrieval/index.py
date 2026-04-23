import logging
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from collections import defaultdict

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
    centroid_id: Optional[str] = None
    heat_tier: int = 0

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
        self._pending_vectors: List[np.ndarray] = []
        
        # IVF-like properties
        self.centroid_matrix: Optional[np.ndarray] = None
        self.centroid_ids: List[str] = []
        self._shards: Dict[str, List[int]] = defaultdict(list)

    def clear(self):
        self.vectors = None
        self.metadata = []
        self._id_to_idx = {}
        self._pending_vectors = []
        self._shards.clear()

    def set_centroids(self, centroids_map: Dict[str, np.ndarray]):
        """Load trained centroid matrix for IVF routing"""
        if not centroids_map:
            self.centroid_matrix = None
            self.centroid_ids = []
            return
            
        self.centroid_ids = list(centroids_map.keys())
        matrix = np.vstack(list(centroids_map.values()))
        
        # Normalize centroids for cosine dot scoring
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.centroid_matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms!=0)

    def _sync_vectors(self):
        """Coalesce pending vectors into the main matrix."""
        if not self._pending_vectors:
            return
        
        new_block = np.vstack(self._pending_vectors)
        if self.vectors is None:
            self.vectors = new_block
        else:
            self.vectors = np.vstack([self.vectors, new_block])
        self._pending_vectors = []
        self._rebuild_shards()
        
    def _rebuild_shards(self):
        """Map centroid IDs to row indices in self.vectors."""
        self._shards.clear()
        for idx, entry in enumerate(self.metadata):
             cid = entry.centroid_id if entry.centroid_id else "unassigned"
             self._shards[cid].append(idx)

    def add_entry(self, entry: IndexEntry, vector: np.ndarray):
        """Append a new entry to the in-memory index."""
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dim mismatch: expected {self.dim}, got {vector.shape[0]}")

        # Ensure normalized
        norm = np.linalg.norm(vector)
        if norm > 1e-10:
            vector = vector / norm
        else:
            vector = np.zeros(self.dim, dtype=np.float32)

        self._id_to_idx[entry.id] = len(self.metadata)
        self.metadata.append(entry)
        self._pending_vectors.append(vector.reshape(1, -1))
        
        # Limit pending list size to keep search speed reasonable before sync
        if len(self._pending_vectors) > 1000:
            self._sync_vectors()

    def bulk_add_entries(self, entries: List[IndexEntry], vectors: List[np.ndarray]):
        """Efficiently add multiple entries at once."""
        if not entries:
            return
        for entry, vector in zip(entries, vectors):
            self._id_to_idx[entry.id] = len(self.metadata)
            self.metadata.append(entry)
            self._pending_vectors.append(vector.reshape(1, -1))
        self._sync_vectors()

    def search(
        self, 
        query_vec: np.ndarray, 
        top_k: int = 50, 
        nprobe: int = 3, 
        min_heat: int = 0
    ) -> List[tuple[IndexEntry, float]]:
        """Perform similarity search using IVF-like shard routing if centroids are present, 
        or fallback to exhaustive brute-force. Heat tier filtering prioritizes hot/warm items."""
        self._sync_vectors()
        if self.vectors is None or len(self.metadata) == 0:
            return []

        # Ensure query is normalized
        norm = np.linalg.norm(query_vec)
        if norm > 1e-10:
            query_vec = query_vec / norm

        scan_indices = []

        # IVF Routing
        if self.centroid_matrix is not None and len(self.centroid_ids) > 0:
            # Route: Score query against all centroids (N_c x D @ D x 1) -> (N_c, 1)
            cent_sims = np.dot(self.centroid_matrix, query_vec)
            
            # Select top-nprobe centroids
            top_c_indices = np.argsort(cent_sims)[::-1][:nprobe]
            
            for c_idx in top_c_indices:
                cid = self.centroid_ids[c_idx]
                scan_indices.extend(self._shards.get(cid, []))
                
            # Always include unassigned memories so they aren't lost
            scan_indices.extend(self._shards.get("unassigned", []))
            
        else:
            # Brute force fallback
            scan_indices = list(range(len(self.metadata)))

        # Apply Heat Tier filter
        if min_heat > 0:
            scan_indices = [idx for idx in scan_indices if self.metadata[idx].heat_tier >= min_heat]

        if not scan_indices:
            return []

        # Uniquify exact search bounds to prevent duplicate scans
        scan_indices = list(set(scan_indices))
        
        # Sub-matrix slicing and parallel similarity scoring
        shard_vecs = self.vectors[scan_indices]
        similarities = np.dot(shard_vecs, query_vec)
        
        # Sort and take local top-K
        local_sort_idx = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for local_idx in local_sort_idx:
            global_idx = scan_indices[local_idx]
            score = (similarities[local_idx] + 1.0) / 2.0
            results.append((self.metadata[global_idx], score))
            
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
