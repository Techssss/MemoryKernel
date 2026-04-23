"""HNSW vector index for SQLite brains.

Sidecar file approach: each brain gets a `{brain_id}.hnsw` index file
and a `{brain_id}.hnsw.map` JSON file mapping int slots to neuron IDs.

Gracefully degrades when hnswlib is not installed — all operations
return empty results instead of raising.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import hnswlib
    import numpy as np

    _HNSWLIB_AVAILABLE = True
except ImportError:
    _HNSWLIB_AVAILABLE = False

# HNSW tuning (same defaults as Pro InfinityDB)
DEFAULT_M = 16
DEFAULT_EF_CONSTRUCTION = 200
DEFAULT_EF_SEARCH = 64  # Lower than Pro (100) for faster CPU search
DEFAULT_DIMENSIONS = 384  # all-MiniLM-L6-v2
DEFAULT_SPACE = "cosine"
INITIAL_CAPACITY = 1024
RESIZE_THRESHOLD = 0.8  # Resize when 80% full


def is_available() -> bool:
    """Check if hnswlib is installed."""
    return _HNSWLIB_AVAILABLE


class SQLiteVectorIndex:
    """HNSW vector index stored as sidecar files next to a SQLite DB.

    Files:
        {base_path}.hnsw     — hnswlib binary index
        {base_path}.hnsw.map — JSON mapping: slot_id (int) ↔ neuron_id (str)
    """

    def __init__(
        self,
        base_path: Path,
        dimensions: int = DEFAULT_DIMENSIONS,
        *,
        m: int = DEFAULT_M,
        ef_construction: int = DEFAULT_EF_CONSTRUCTION,
        ef_search: int = DEFAULT_EF_SEARCH,
        space: str = DEFAULT_SPACE,
    ) -> None:
        self._index_path = base_path.with_suffix(".hnsw")
        self._map_path = base_path.with_suffix(".hnsw.map")
        self._dimensions = dimensions
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._space = space

        # State
        self._index: Any = None  # hnswlib.Index when loaded
        self._max_elements = 0
        self._current_count = 0

        # ID mapping: slot (int) ↔ neuron_id (str)
        self._slot_to_id: dict[int, str] = {}
        self._id_to_slot: dict[str, int] = {}
        self._next_slot: int = 0
        self._is_open = False

    @property
    def count(self) -> int:
        """Number of indexed vectors."""
        return self._current_count

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        """Load or create the index and ID mapping."""
        if not _HNSWLIB_AVAILABLE:
            logger.debug("hnswlib not installed — vector index disabled")
            return

        # Load ID mapping first (determines capacity)
        self._load_mapping()

        self._index = hnswlib.Index(space=self._space, dim=self._dimensions)

        if self._index_path.exists() and self._index_path.stat().st_size > 0:
            max_elements = max(INITIAL_CAPACITY, len(self._slot_to_id) * 2)
            self._index.load_index(str(self._index_path), max_elements=max_elements)
            self._max_elements = max_elements
            self._current_count = self._index.get_current_count()
            self._index.set_ef(self._ef_search)
            logger.debug(
                "Loaded vector index: %d elements from %s",
                self._current_count,
                self._index_path,
            )
        else:
            self._max_elements = INITIAL_CAPACITY
            self._index.init_index(
                max_elements=INITIAL_CAPACITY,
                M=self._m,
                ef_construction=self._ef_construction,
            )
            self._index.set_ef(self._ef_search)
            logger.debug("Created new vector index: max=%d", INITIAL_CAPACITY)

        self._is_open = True

    def add(self, neuron_id: str, vector: list[float]) -> None:
        """Add or update a neuron's vector in the index."""
        if self._index is None:
            return

        vec = np.array(vector, dtype=np.float32)
        if vec.shape[0] != self._dimensions:
            logger.warning(
                "Vector dim mismatch: got %d, expected %d — skipping %s",
                vec.shape[0],
                self._dimensions,
                neuron_id,
            )
            return

        # If neuron already indexed, remove old entry first
        if neuron_id in self._id_to_slot:
            self.remove(neuron_id)

        slot = self._next_slot
        self._next_slot += 1

        self._ensure_capacity(self._current_count + 1)
        self._index.add_items(
            vec.reshape(1, -1),
            np.array([slot], dtype=np.int64),
        )
        self._slot_to_id[slot] = neuron_id
        self._id_to_slot[neuron_id] = slot
        self._current_count += 1

    def add_batch(self, neuron_ids: list[str], vectors: list[list[float]]) -> None:
        """Add multiple vectors at once."""
        if self._index is None or not neuron_ids:
            return

        vecs = np.array(vectors, dtype=np.float32)
        if vecs.shape[1] != self._dimensions:
            logger.warning("Batch vector dim mismatch — skipping")
            return

        # Remove any existing entries
        for nid in neuron_ids:
            if nid in self._id_to_slot:
                self.remove(nid)

        n = len(neuron_ids)
        slots = list(range(self._next_slot, self._next_slot + n))
        self._next_slot += n

        self._ensure_capacity(self._current_count + n)
        self._index.add_items(vecs, np.array(slots, dtype=np.int64))

        for slot, nid in zip(slots, neuron_ids, strict=True):
            self._slot_to_id[slot] = nid
            self._id_to_slot[nid] = slot
        self._current_count += n

    def search(
        self,
        query_vector: list[float],
        k: int = 20,
    ) -> list[tuple[str, float]]:
        """Search for k nearest neurons by vector similarity.

        Returns:
            [(neuron_id, similarity)] sorted by similarity descending.
            Cosine space: similarity = 1 - distance.
        """
        if self._index is None or self._current_count == 0:
            return []

        vec = np.array(query_vector, dtype=np.float32)
        k = min(k, self._current_count)
        labels, distances = self._index.knn_query(vec.reshape(1, -1), k=k)

        results: list[tuple[str, float]] = []
        for slot, dist in zip(labels[0].tolist(), distances[0].tolist(), strict=False):
            nid = self._slot_to_id.get(slot)
            if nid is not None:
                similarity = max(0.0, 1.0 - dist)
                results.append((nid, similarity))
        return results

    def remove(self, neuron_id: str) -> None:
        """Remove a neuron's vector from the index."""
        if self._index is None:
            return

        slot = self._id_to_slot.pop(neuron_id, None)
        if slot is None:
            return

        try:
            self._index.mark_deleted(slot)
            self._current_count = max(0, self._current_count - 1)
        except RuntimeError as e:
            if "not found" not in str(e).lower():
                raise
        self._slot_to_id.pop(slot, None)

    def save(self) -> None:
        """Persist index and mapping to disk."""
        if self._index is not None:
            self._index_path.parent.mkdir(parents=True, exist_ok=True)
            self._index.save_index(str(self._index_path))
        self._save_mapping()
        logger.debug("Saved vector index: %d elements", self._current_count)

    def close(self) -> None:
        """Save and release resources."""
        if self._is_open:
            self.save()
        self._index = None
        self._is_open = False

    def rebuild(self, neuron_vectors: dict[str, list[float]]) -> None:
        """Rebuild entire index from a neuron_id → vector mapping.

        Used for recovery when sidecar is lost but embeddings exist in metadata.
        """
        if not _HNSWLIB_AVAILABLE or not neuron_vectors:
            return

        neuron_ids = list(neuron_vectors.keys())
        vectors = np.array(list(neuron_vectors.values()), dtype=np.float32)
        n = len(neuron_ids)

        self._index = hnswlib.Index(space=self._space, dim=self._dimensions)
        max_elements = max(INITIAL_CAPACITY, n * 2)
        self._index.init_index(
            max_elements=max_elements,
            M=self._m,
            ef_construction=self._ef_construction,
        )
        self._index.set_ef(self._ef_search)
        self._max_elements = max_elements

        slots = list(range(n))
        self._index.add_items(vectors, np.array(slots, dtype=np.int64))

        self._slot_to_id = dict(enumerate(neuron_ids))
        self._id_to_slot = {nid: i for i, nid in enumerate(neuron_ids)}
        self._next_slot = n
        self._current_count = n
        self._is_open = True

        self.save()
        logger.info("Rebuilt vector index from %d neuron embeddings", n)

    # -- Private helpers --

    def _ensure_capacity(self, needed: int) -> None:
        if self._index is None:
            return
        if needed > self._max_elements:
            new_max = max(needed, self._max_elements * 2)
            self._index.resize_index(new_max)
            self._max_elements = new_max

    def _load_mapping(self) -> None:
        if self._map_path.exists():
            try:
                data = json.loads(self._map_path.read_text(encoding="utf-8"))
                self._slot_to_id = {int(k): v for k, v in data.get("slot_to_id", {}).items()}
                self._id_to_slot = {v: int(k) for k, v in self._slot_to_id.items()}
                self._next_slot = data.get("next_slot", 0)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Corrupt vector index mapping — starting fresh")
                self._slot_to_id = {}
                self._id_to_slot = {}
                self._next_slot = 0

    def _save_mapping(self) -> None:
        self._map_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "slot_to_id": {str(k): v for k, v in self._slot_to_id.items()},
            "next_slot": self._next_slot,
        }
        self._map_path.write_text(json.dumps(data), encoding="utf-8")
