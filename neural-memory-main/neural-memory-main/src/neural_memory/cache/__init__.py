"""Activation cache module for computational state persistence.

Caches neuron activation states at session boundaries to enable
warm-start recall, reducing latency by skipping activation recomputation.

Inspired by "Memory Caching: RNNs with Growing Memory" (ICLR 2026).
"""

from neural_memory.cache.invalidation import (
    InvalidationTracker,
    apply_invalidation,
    detect_staleness,
    remove_neurons,
    should_fully_invalidate,
)
from neural_memory.cache.manager import ActivationCacheManager
from neural_memory.cache.models import ActivationCache, CachedState
from neural_memory.cache.selector import select_relevant, warm_activations_from_states

__all__ = [
    "ActivationCache",
    "ActivationCacheManager",
    "CachedState",
    "InvalidationTracker",
    "apply_invalidation",
    "detect_staleness",
    "remove_neurons",
    "select_relevant",
    "should_fully_invalidate",
    "warm_activations_from_states",
]
