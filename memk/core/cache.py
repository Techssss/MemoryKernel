import time
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple
from collections import OrderedDict

logger = logging.getLogger("memk.cache")

class LRUCache:
    """Simple thread-safe LRU cache with TTL."""
    def __init__(self, maxsize: int = 100, ttl_seconds: int = 3600):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Optional[Any]:
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            val, expiry = self.cache[key]
            if time.time() > expiry:
                del self.cache[key]
                self.misses += 1
                return None
            
            self.cache.move_to_end(key)
            self.hits += 1
            return val

    def set(self, key: Any, value: Any):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            
            expiry = time.time() + self.ttl
            self.cache[key] = (value, expiry)
            
            if len(self.cache) > self.maxsize:
                self.cache.popitem(last=False)

    def clear(self):
        with self.lock:
            self.cache.clear()

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "size": len(self.cache),
            "max_size": self.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{(self.hits / (self.hits + self.misses) * 100):.1f}%" if (self.hits + self.misses) > 0 else "0%"
        }

class MemoryCacheManager:
    """Orchestrates multi-layer caching for MemoryKernel with generation-based invalidation."""
    def __init__(self):
        # Layer 1: Raw string -> Embedding vector
        self.embeddings = LRUCache(maxsize=500, ttl_seconds=86400) # 24h
        
        # Layer 2: Search params -> List of items
        self.search_results = LRUCache(maxsize=100, ttl_seconds=300) # 5m
        
        # Layer 3: Context params -> Final string
        self.contexts = LRUCache(maxsize=100, ttl_seconds=300) # 5m
        
        # Track the generation this cache was built for
        self._cache_generation: Optional[int] = None
        self._lock = threading.Lock()

    def set_generation(self, generation: int):
        """
        Update the cache generation marker.
        If generation has changed, invalidate structural caches.
        """
        with self._lock:
            if self._cache_generation is not None and self._cache_generation != generation:
                logger.info(f"Generation changed ({self._cache_generation} -> {generation}). Invalidating caches.")
                self.search_results.clear()
                self.contexts.clear()
            self._cache_generation = generation

    def get_generation(self) -> Optional[int]:
        """Return the generation this cache is currently valid for."""
        with self._lock:
            return self._cache_generation

    def invalidate_structural(self):
        """Called when data changes (write/delete). Invalidate layers 2 & 3."""
        logger.info("New write detected. Invalidating search and context caches.")
        with self._lock:
            self.search_results.clear()
            self.contexts.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "embeddings": self.embeddings.stats,
            "search": self.search_results.stats,
            "contexts": self.contexts.stats,
            "cache_generation": self._cache_generation,
        }
