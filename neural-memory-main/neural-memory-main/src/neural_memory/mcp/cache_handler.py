"""Activation cache handler for MCP server."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)


class CacheHandler:
    """Mixin: activation cache tool handlers.

    Provides MCP tool for activation cache management:
    - status: Show cache age, hit rate, entries
    - clear: Invalidate cache
    - save: Force snapshot
    - load: Force restore (typically automatic on startup)
    """

    _cache_manager: Any = None  # ActivationCacheManager

    async def get_storage(self) -> NeuralStorage:
        """Protocol stub for storage access."""
        raise NotImplementedError

    async def _get_cache_manager(self) -> Any:
        """Get or create ActivationCacheManager instance."""
        if self._cache_manager is not None:
            return self._cache_manager

        from neural_memory.cache.manager import ActivationCacheManager

        storage = await self.get_storage()
        data_dir = Path.home() / ".neuralmemory"
        self._cache_manager = ActivationCacheManager(storage, data_dir)
        return self._cache_manager

    async def _cache(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle nmem_cache tool calls.

        Args:
            args: Tool arguments with 'action' key

        Actions:
            status: Show cache statistics
            clear: Invalidate and delete cache
            save: Force save current activation states
            load: Force load cached states (returns loaded count)
        """
        action = args.get("action", "status")
        mgr = await self._get_cache_manager()

        if action == "status":
            return await self._cache_status(mgr)
        elif action == "clear":
            return await self._cache_clear(mgr)
        elif action == "save":
            return await self._cache_save(mgr)
        elif action == "load":
            return await self._cache_load(mgr)
        else:
            return {"error": f"Unknown cache action: {action}"}

    async def _cache_status(self, mgr: Any) -> dict[str, Any]:
        """Get cache statistics."""
        stats = await mgr.get_stats()
        return {
            "status": "ok",
            "brain_name": stats.get("brain_name", ""),
            "cache_exists": stats.get("cache_exists", False),
            "entries": stats.get("entries", 0),
            "hit_count": stats.get("hit_count", 0),
            "miss_count": stats.get("miss_count", 0),
            "hit_rate": stats.get("hit_rate", 0.0),
            "age_seconds": stats.get("age_seconds", 0),
            "ttl_hours": stats.get("ttl_hours", 24),
        }

    async def _cache_clear(self, mgr: Any) -> dict[str, Any]:
        """Invalidate and delete cache."""
        deleted = await mgr.invalidate()
        return {
            "status": "ok",
            "deleted": deleted,
            "message": "Activation cache invalidated" if deleted else "No cache to delete",
        }

    async def _cache_save(self, mgr: Any) -> dict[str, Any]:
        """Force save current activation states."""
        cache = await mgr.save_snapshot()
        if cache is None:
            return {
                "status": "ok",
                "saved": False,
                "message": "No active neurons to cache",
            }
        return {
            "status": "ok",
            "saved": True,
            "entries": cache.entry_count,
            "brain_name": cache.brain_name,
            "message": f"Saved {cache.entry_count} activation states",
        }

    async def _cache_load(self, mgr: Any) -> dict[str, Any]:
        """Force load cached activation states."""
        states = await mgr.load_snapshot()
        if not states:
            return {
                "status": "ok",
                "loaded": False,
                "entries": 0,
                "message": "No valid cache found or cache expired",
            }
        return {
            "status": "ok",
            "loaded": True,
            "entries": len(states),
            "message": f"Loaded {len(states)} cached activation states",
        }

    async def save_activation_cache(self) -> None:
        """Save activation cache on session end.

        Called by scheduler's session_end event.
        """
        try:
            mgr = await self._get_cache_manager()
            cache = await mgr.save_snapshot()
            if cache:
                logger.info(
                    "Saved activation cache: %d entries for brain '%s'",
                    cache.entry_count,
                    cache.brain_name,
                )
        except Exception as e:
            logger.warning("Failed to save activation cache: %s", e)

    async def load_activation_cache(self) -> int:
        """Load activation cache on startup.

        Returns:
            Number of states loaded (0 if none)
        """
        try:
            mgr = await self._get_cache_manager()
            states = await mgr.load_snapshot()
            if states:
                logger.info("Loaded %d cached activation states", len(states))
            return len(states)
        except Exception as e:
            logger.warning("Failed to load activation cache: %s", e)
            return 0
