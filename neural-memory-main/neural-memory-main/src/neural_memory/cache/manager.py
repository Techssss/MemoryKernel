"""Activation cache manager for session-level caching.

Handles save/load/invalidation of cached activation states.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from neural_memory.cache.models import ActivationCache, CachedState
from neural_memory.cache.serializer import (
    cache_exists,
    delete_cache,
    load_cache,
    save_cache,
)
from neural_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# Default limits
DEFAULT_MAX_ENTRIES = 200
DEFAULT_TTL_HOURS = 24
DEFAULT_MIN_ACTIVATION = 0.1


class ActivationCacheManager:
    """Manages activation state caching for warm-start recall.

    Usage:
        manager = ActivationCacheManager(storage)

        # Save at session end
        await manager.save_snapshot()

        # Load at session start
        states = await manager.load_snapshot()

        # Check validity
        if manager.is_cache_valid():
            # Use cached states
            ...
    """

    def __init__(
        self,
        storage: NeuralStorage,
        data_dir: Path | None = None,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_hours: int = DEFAULT_TTL_HOURS,
        min_activation: float = DEFAULT_MIN_ACTIVATION,
    ) -> None:
        """Initialize cache manager.

        Args:
            storage: Neural storage backend
            data_dir: Data directory for cache files
            max_entries: Max neurons to cache
            ttl_hours: Cache time-to-live
            min_activation: Minimum activation level to cache
        """
        self._storage = storage
        self._data_dir = data_dir
        self._max_entries = max_entries
        self._ttl_hours = max(1, ttl_hours)
        if ttl_hours < 1:
            logger.warning("ActivationCacheManager ttl_hours=%d invalid, clamping to 1", ttl_hours)
        self._min_activation = min_activation
        self._loaded_cache: ActivationCache | None = None
        self._hit_count = 0
        self._miss_count = 0

    @property
    def brain_id(self) -> str:
        """Get current brain ID from storage."""
        return getattr(self._storage, "brain_id", "") or ""

    @property
    def brain_name(self) -> str:
        """Get last-known brain name from cache (sync).

        Prefer `_resolve_brain_name()` in async contexts — it fetches
        authoritative name from storage and is consistent with save/load.
        """
        if self._loaded_cache is not None:
            return self._loaded_cache.brain_name
        brain_id = self.brain_id
        if not brain_id:
            return "default"
        return getattr(self._storage, "_brain_name", None) or "default"

    async def _resolve_brain_name(self) -> str:
        """Resolve brain name via storage.get_brain (authoritative, async)."""
        brain_id = self.brain_id
        if not brain_id:
            return "default"
        brain = await self._storage.get_brain(brain_id)
        return brain.name if brain else "default"

    async def save_snapshot(self) -> ActivationCache | None:
        """Save current activation states to cache.

        Captures top-N neurons by activation level and persists to disk.

        Returns:
            ActivationCache if saved, None on error
        """
        brain_id = self.brain_id
        if not brain_id:
            logger.debug("No brain_id, skipping activation cache save")
            return None

        try:
            brain_name = await self._resolve_brain_name()

            # Get all neuron states
            all_states = await self._get_top_states()

            if not all_states:
                logger.debug("No active neurons to cache")
                return None

            # Compute brain hash for staleness detection
            brain_hash = await self._compute_brain_hash()

            now = utcnow()
            cache = ActivationCache(
                brain_id=brain_id,
                brain_name=brain_name,
                cached_at=now,
                ttl_hours=self._ttl_hours,
                entries=tuple(all_states),
                brain_hash=brain_hash,
            )

            save_cache(cache, self._data_dir)
            self._loaded_cache = cache

            logger.info(
                "Saved activation cache: %d entries for brain '%s'",
                len(all_states),
                brain_name,
            )
            return cache

        except Exception as e:
            logger.error("Failed to save activation cache: %s", e, exc_info=True)
            return None

    async def load_snapshot(self) -> list[CachedState]:
        """Load cached activation states if valid.

        Returns:
            List of cached states, empty if cache invalid/missing
        """
        # Fetch brain name consistently with save_snapshot
        brain_id = self.brain_id
        if not brain_id:
            self._miss_count += 1
            return []

        brain_name = await self._resolve_brain_name()

        cache = load_cache(brain_name, self._data_dir)
        if cache is None:
            self._miss_count += 1
            return []

        # Check expiry
        if cache.is_expired():
            logger.debug("Activation cache expired for '%s'", brain_name)
            self._miss_count += 1
            return []

        # Check brain hash (staleness)
        current_hash = await self._compute_brain_hash()
        if cache.brain_hash and cache.brain_hash != current_hash:
            logger.debug(
                "Activation cache stale for '%s' (hash mismatch)",
                brain_name,
            )
            self._miss_count += 1
            return []

        self._loaded_cache = cache
        self._hit_count += 1

        logger.info(
            "Loaded activation cache: %d entries for brain '%s'",
            cache.entry_count,
            brain_name,
        )
        return list(cache.entries)

    def is_cache_valid(self) -> bool:
        """Check if current loaded cache is valid."""
        if self._loaded_cache is None:
            return False
        return not self._loaded_cache.is_expired()

    def get_warm_activations(self) -> dict[str, float]:
        """Return loaded cache as {neuron_id: activation_level} for warm-start.

        Empty dict if cache is missing or expired.
        """
        if self._loaded_cache is None or self._loaded_cache.is_expired():
            return {}
        return {e.neuron_id: e.activation_level for e in self._loaded_cache.entries}

    async def get_warm_activations_selective(
        self,
        query: str,
        embedding_provider: Any | None,
        top_k: int = 20,
        min_similarity: float = 0.3,
    ) -> dict[str, float]:
        """Return top-K warm activations ranked by query relevance.

        Uses SSC-lite: embeds the query, cosine-ranks cached neurons, keeps
        the top-K. Falls back to activation-ranked top-K when ranking is
        unavailable.

        Args:
            query: User recall query
            embedding_provider: Active embedding provider (may be None)
            top_k: Max entries to return (clamped to [1, max_entries])
            min_similarity: Cosine threshold (clamped to [-1.0, 1.0])

        Returns:
            Dict of {neuron_id: activation_level}, empty when cache is invalid.
        """
        if self._loaded_cache is None or self._loaded_cache.is_expired():
            return {}

        # Bounds check at the manager boundary (project convention).
        top_k = max(1, min(int(top_k), self._max_entries))
        min_similarity = max(-1.0, min(float(min_similarity), 1.0))

        from neural_memory.cache.selector import select_relevant, warm_activations_from_states

        selected = await select_relevant(
            cached_states=list(self._loaded_cache.entries),
            query=query,
            embedding_provider=embedding_provider,
            storage=self._storage,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        return warm_activations_from_states(selected)

    def _replace_cache(self, cache: ActivationCache | None) -> None:
        """Swap loaded cache reference (internal — used by invalidation module)."""
        self._loaded_cache = cache

    def get_cached_state(self, neuron_id: str) -> CachedState | None:
        """Get cached state for a neuron.

        Args:
            neuron_id: Neuron UUID

        Returns:
            CachedState if found in cache, None otherwise
        """
        if self._loaded_cache is None:
            self._miss_count += 1
            return None

        state = self._loaded_cache.get_state(neuron_id)
        if state:
            self._hit_count += 1
        else:
            self._miss_count += 1
        return state

    async def invalidate(self) -> bool:
        """Invalidate and delete current cache.

        Call this when brain state changes (new neurons, synapses, etc.)

        Returns:
            True if cache was deleted
        """
        brain_name = await self._resolve_brain_name()
        self._loaded_cache = None
        return delete_cache(brain_name, self._data_dir)

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics (async — resolves brain_name authoritatively).

        Returns:
            Dict with hit_count, miss_count, hit_rate, entries, age_seconds
        """
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0.0
        brain_name = await self._resolve_brain_name()

        stats: dict[str, Any] = {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(hit_rate, 3),
            "entries": 0,
            "age_seconds": 0,
            "brain_name": brain_name,
            "cache_exists": cache_exists(brain_name, self._data_dir),
        }

        if self._loaded_cache:
            stats["entries"] = self._loaded_cache.entry_count
            age = (utcnow() - self._loaded_cache.cached_at).total_seconds()
            stats["age_seconds"] = int(age)
            stats["ttl_hours"] = self._loaded_cache.ttl_hours

        return stats

    async def _get_top_states(self) -> list[CachedState]:
        """Get top neuron states by activation level."""
        # Get all neuron IDs
        all_neurons = await self._storage.find_neurons(limit=1000)
        if not all_neurons:
            return []

        neuron_ids = [n.id for n in all_neurons]
        states_map = await self._storage.get_neuron_states_batch(neuron_ids)

        # Filter and sort by activation
        cached: list[CachedState] = []
        for nid, state in states_map.items():
            if state.activation_level >= self._min_activation:
                cached.append(
                    CachedState(
                        neuron_id=nid,
                        activation_level=state.activation_level,
                        access_frequency=state.access_frequency,
                        last_activated=state.last_activated,
                    )
                )

        # Sort by activation descending, take top N
        cached.sort(key=lambda x: x.activation_level, reverse=True)
        return cached[: self._max_entries]

    async def _compute_brain_hash(self) -> str:
        """Compute hash of brain state for staleness detection.

        Primary signal: counts + last_modified. When last_modified is
        unavailable, samples recent neuron IDs as entropy to detect
        swap-equal mutations (add+remove of same count).
        """
        brain_id = self.brain_id
        if not brain_id:
            return ""

        try:
            stats = await self._storage.get_stats(brain_id)
            neuron_count = stats.get("neuron_count", 0)
            synapse_count = stats.get("synapse_count", 0)
            fiber_count = stats.get("fiber_count", 0)
            last_modified = stats.get("last_modified", "")

            parts = [brain_id, str(neuron_count), str(synapse_count), str(fiber_count)]
            if last_modified:
                parts.append(str(last_modified))
            else:
                # Fallback entropy: sample neuron IDs so swap-equal mutations
                # (add + delete of equal count) still change the hash. Sort to
                # keep the hash deterministic regardless of storage row order.
                try:
                    sample = await self._storage.find_neurons(limit=20)
                    parts.extend(sorted(n.id for n in sample))
                except Exception:
                    pass

            data = ":".join(parts)
            return hashlib.md5(data.encode()).hexdigest()[:16]
        except Exception:
            return ""
