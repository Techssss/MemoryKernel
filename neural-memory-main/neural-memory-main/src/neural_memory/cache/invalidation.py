"""Activation cache invalidation.

Provides three invalidation strategies:

1. **Full**: drop the whole cache (brain rebuild, schema migration).
2. **Partial**: drop specific neuron IDs from cached entries, keep the rest.
3. **Hash-based (staleness)**: compare brain state hash — if drift exceeds
   threshold, invalidate; otherwise keep and let TTL age the cache out.

The module is event-driven: storage/engine code calls
`InvalidationTracker.record_change(...)` when data mutates; callers then
decide when to apply (usually at snapshot-save or on-demand).

We never mutate the loaded ActivationCache in place — every transform
returns a new instance (immutability rule).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from neural_memory.cache.manager import ActivationCacheManager
    from neural_memory.cache.models import ActivationCache

logger = logging.getLogger(__name__)


ChangeKind = Literal["neuron_add", "neuron_delete", "synapse_change", "consolidation"]

# When the fraction of changed neurons exceeds this, prefer full invalidation.
FULL_INVALIDATE_RATIO = 0.25


@dataclass
class InvalidationTracker:
    """Accumulates neuron/synapse changes for batched invalidation.

    Intended lifetime: a single MCP session. Storage/engine hooks push
    change events here; the cache manager drains on snapshot-save.
    """

    dirty_neurons: set[str] = field(default_factory=set)
    consolidation_triggered: bool = False
    change_counts: dict[str, int] = field(default_factory=dict)

    def record_change(self, kind: ChangeKind, neuron_id: str | None = None) -> None:
        """Record a single change event.

        Args:
            kind: Change classification
            neuron_id: Affected neuron UUID (optional for some kinds)
        """
        self.change_counts[kind] = self.change_counts.get(kind, 0) + 1

        if kind == "consolidation":
            self.consolidation_triggered = True
            return

        if neuron_id:
            self.dirty_neurons.add(neuron_id)

    def record_bulk(self, kind: ChangeKind, neuron_ids: list[str]) -> None:
        """Record multiple changes at once (e.g. batch delete)."""
        if not neuron_ids:
            return
        self.change_counts[kind] = self.change_counts.get(kind, 0) + len(neuron_ids)
        if kind == "consolidation":
            self.consolidation_triggered = True
            return
        self.dirty_neurons.update(neuron_ids)

    def is_empty(self) -> bool:
        """True when no changes have been recorded."""
        return not self.dirty_neurons and not self.consolidation_triggered

    def reset(self) -> None:
        """Clear all recorded changes — call after applying invalidation."""
        self.dirty_neurons.clear()
        self.consolidation_triggered = False
        self.change_counts.clear()

    def summary(self) -> dict[str, int | bool]:
        """Serializable snapshot for logging/status tools."""
        return {
            "dirty_neurons": len(self.dirty_neurons),
            "consolidation_triggered": self.consolidation_triggered,
            **{f"count_{k}": v for k, v in self.change_counts.items()},
        }


def remove_neurons(cache: ActivationCache, neuron_ids: set[str]) -> ActivationCache:
    """Return a new cache with the given neurons removed.

    Used for partial invalidation: deleted/mutated neurons drop out but
    the rest of the warm state is preserved.
    """
    if not neuron_ids or not cache.entries:
        return cache
    remaining = tuple(e for e in cache.entries if e.neuron_id not in neuron_ids)
    if len(remaining) == len(cache.entries):
        return cache
    return replace(cache, entries=remaining)


def should_fully_invalidate(
    tracker: InvalidationTracker,
    cache: ActivationCache,
    full_ratio: float = FULL_INVALIDATE_RATIO,
) -> bool:
    """Decide whether cache should be dropped entirely vs. trimmed.

    Full invalidation triggers when:
      - Consolidation ran (graph topology likely shifted)
      - Change ratio ≥ `full_ratio` of cached entries
    """
    if tracker.consolidation_triggered:
        return True
    if not cache.entries:
        return False
    dirty_in_cache = sum(1 for e in cache.entries if e.neuron_id in tracker.dirty_neurons)
    if dirty_in_cache == 0:
        return False
    ratio = dirty_in_cache / len(cache.entries)
    return ratio >= full_ratio


async def apply_invalidation(
    manager: ActivationCacheManager,
    tracker: InvalidationTracker,
) -> dict[str, int | bool | str]:
    """Apply a tracker's changes against the manager's loaded cache.

    Returns a report describing the action taken. Always resets the
    tracker after a decision is made.
    """
    action: str
    removed = 0

    loaded = manager._loaded_cache
    if loaded is None:
        action = "no_cache"
    elif tracker.is_empty():
        action = "no_changes"
    elif should_fully_invalidate(tracker, loaded):
        await manager.invalidate()
        action = "full"
        removed = loaded.entry_count
    else:
        pruned = remove_neurons(loaded, tracker.dirty_neurons)
        removed = loaded.entry_count - pruned.entry_count
        if removed > 0:
            manager._replace_cache(pruned)
            action = "partial"
        else:
            action = "noop"

    report: dict[str, int | bool | str] = {
        "action": action,
        "removed": removed,
        "dirty_neurons": len(tracker.dirty_neurons),
        "consolidation": tracker.consolidation_triggered,
    }
    tracker.reset()

    if action in ("full", "partial"):
        logger.info("Cache invalidation: %s (%d entries affected)", action, removed)

    return report


async def detect_staleness(manager: ActivationCacheManager) -> bool:
    """Hash-based staleness check against live brain state.

    Returns True when the loaded cache's brain_hash no longer matches the
    current brain — caller should discard the warm cache.
    """
    loaded = manager._loaded_cache
    if loaded is None or not loaded.brain_hash:
        return False
    try:
        current_hash = await manager._compute_brain_hash()
    except Exception:
        logger.debug("detect_staleness: brain hash compute failed", exc_info=True)
        return False
    return bool(current_hash) and current_hash != loaded.brain_hash
