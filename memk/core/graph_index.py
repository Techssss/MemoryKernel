"""
memk.core.graph_index
=====================
In-memory Graph Sidecar (Index) based purely on NumPy.

Constructs CSR-like adjacency arrays from SQLite sidecar tables for
high performance graph propagation (random walks, PageRank, etc.).

Optimized for fast reading/traversal in RAM. Writing is handled by
the GraphRepository to SQLite, and this index is rebuilt/refreshed
periodically or on-demand.
"""

import sqlite3
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class GraphIndex:
    """
    In-memory graph representation using NumPy structured as CSR (Compressed Sparse Row).

    Provides fast adjacency lookups for:
      1. Entity -> Entity (Edges)
      2. Memory -> Entity (Mentions)
      3. Entity -> Memory (Inverse Mentions)
    """

    def __init__(self, db_path: str):
        """
        Initialize an empty GraphIndex.
        
        Parameters
        ----------
        db_path : str
            Path to the MemoryKernel SQLite database.
        """
        self.db_path = db_path
        
        # Identity mappings
        self.entity_id_map: Dict[int, int] = {}  # sqlite_entity_id -> internal_id
        self.entity_ids: List[int] = []          # internal_id -> sqlite_entity_id
        
        self.memory_id_map: Dict[str, int] = {}  # sqlite_memory_id -> internal_id
        self.memory_ids: List[str] = []          # internal_id -> sqlite_memory_id

        # Structural dimensions
        self.num_entities = 0
        self.num_memories = 0

        # Adjacency: Entity -> Entity (Directed edges)
        self.e2e_indptr = np.array([0], dtype=np.int32)
        self.e2e_indices = np.array([], dtype=np.int32)
        self.e2e_weights = np.array([], dtype=np.float32)

        # Adjacency: Memory -> Entity (Mentions)
        self.m2e_indptr = np.array([0], dtype=np.int32)
        self.m2e_indices = np.array([], dtype=np.int32)

        # Adjacency: Entity -> Memory (Inverse mentions)
        self.e2m_indptr = np.array([0], dtype=np.int32)
        self.e2m_indices = np.array([], dtype=np.int32)

    def refresh(self, workspace_id: str) -> None:
        """Alias to rebuild the index from the database."""
        self.build_from_db(workspace_id)

    def build_from_db(self, workspace_id: str) -> None:
        """
        Query SQLite for all active graph components and construct
        CSR arrays in memory.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            # 1. Load entities
            entities = conn.execute(
                "SELECT id FROM entity WHERE workspace_id = ? ORDER BY id", 
                (workspace_id,)
            ).fetchall()
            
            self.entity_ids = [row["id"] for row in entities]
            self.entity_id_map = {eid: i for i, eid in enumerate(self.entity_ids)}
            self.num_entities = len(self.entity_ids)
            
            # 2. Load edges (Entity -> Entity)
            edges_query = conn.execute(
                """
                SELECT src_entity_id, dst_entity_id, weight 
                FROM edge 
                WHERE workspace_id = ? AND archived = 0
                """,
                (workspace_id,)
            ).fetchall()
            
            edges = []
            edge_weights = []
            for r in edges_query:
                u = self.entity_id_map.get(r["src_entity_id"])
                v = self.entity_id_map.get(r["dst_entity_id"])
                if u is not None and v is not None:
                    edges.append((u, v))
                    edge_weights.append(float(r["weight"]))
                    
            self.e2e_indptr, self.e2e_indices, self.e2e_weights = self._build_csr(
                self.num_entities, edges, edge_weights
            )
            
            # 3. Load mentions (Memory <-> Entity)
            mentions_query = conn.execute(
                """
                SELECT m.memory_id, m.entity_id 
                FROM mention m
                INNER JOIN entity e ON m.entity_id = e.id
                WHERE e.workspace_id = ?
                """,
                (workspace_id,)
            ).fetchall()
            
            memories_set = set(row["memory_id"] for row in mentions_query)
            self.memory_ids = list(memories_set)
            self.memory_ids.sort() # Guarantee deterministic order
            self.memory_id_map = {mid: i for i, mid in enumerate(self.memory_ids)}
            self.num_memories = len(self.memory_ids)
            
            m2e_edges = []
            e2m_edges = []
            
            for r in mentions_query:
                m_int = self.memory_id_map.get(r["memory_id"])
                e_int = self.entity_id_map.get(r["entity_id"])
                if m_int is not None and e_int is not None:
                    m2e_edges.append((m_int, e_int))
                    e2m_edges.append((e_int, m_int))
                    
            self.m2e_indptr, self.m2e_indices, _ = self._build_csr(
                self.num_memories, m2e_edges, None
            )
            self.e2m_indptr, self.e2m_indices, _ = self._build_csr(
                self.num_entities, e2m_edges, None
            )
            
            logger.debug(f"[{workspace_id}] GraphIndex rebuilt: {self.get_stats()}")
            
        finally:
            conn.close()

    def _build_csr(
        self, 
        num_rows: int, 
        edge_list: List[Tuple[int, int]], 
        weights: Optional[List[float]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper to construct a Compressed Sparse Row (CSR) abstraction 
        using pure NumPy ops (without SciPy dependency).
        """
        # 1. Count out-degrees
        out_degree = np.zeros(num_rows, dtype=np.int32)
        for src, _ in edge_list:
            out_degree[src] += 1
            
        # 2. Cumulative sum into indptr
        indptr = np.zeros(num_rows + 1, dtype=np.int32)
        np.cumsum(out_degree, out=indptr[1:])
        
        # 3. Fill indices & data
        indices = np.zeros(len(edge_list), dtype=np.int32)
        if weights is not None:
            data = np.zeros(len(edge_list), dtype=np.float32)
        else:
            data = np.array([], dtype=np.float32)
            
        pos = indptr[:-1].copy()
        
        for i, (src, dst) in enumerate(edge_list):
            p = pos[src]
            indices[p] = dst
            if weights is not None:
                data[p] = weights[i]
            pos[src] += 1
            
        return indptr, indices, data

    def get_stats(self) -> Dict[str, int]:
        """Return dimensions and edge counts loaded in RAM."""
        return {
            "num_entities": self.num_entities,
            "num_memories": self.num_memories,
            "e2e_edges": len(self.e2e_indices),
            "m2e_mentions": len(self.m2e_indices),
            "e2m_mentions": len(self.e2m_indices),
        }
