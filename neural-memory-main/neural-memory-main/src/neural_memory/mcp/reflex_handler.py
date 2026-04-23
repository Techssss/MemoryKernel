"""Reflex arc handler for MCP server — pin/unpin/list always-on neurons."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from neural_memory.mcp.tool_handler_utils import _require_brain_id

if TYPE_CHECKING:
    from neural_memory.storage.base import NeuralStorage
    from neural_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class ReflexHandler:
    """Mixin: reflex neuron pin/unpin/list tool handler."""

    config: UnifiedConfig

    async def get_storage(self) -> NeuralStorage:
        raise NotImplementedError

    async def _reflex(self, args: dict[str, Any]) -> dict[str, Any]:
        """Handle reflex arc actions: pin, unpin, list."""
        action = args.get("action", "list")

        if action == "pin":
            return await self._reflex_pin(args)
        elif action == "unpin":
            return await self._reflex_unpin(args)
        elif action == "list":
            return await self._reflex_list(args)
        return {"error": f"Unknown reflex action: {action}"}

    async def _reflex_pin(self, args: dict[str, Any]) -> dict[str, Any]:
        """Pin a neuron as a reflex (always-on in recall)."""
        from neural_memory.engine.reflex_conflict import pin_as_reflex

        storage = await self.get_storage()
        _require_brain_id(storage)

        neuron_id = (args.get("neuron_id") or "").strip()
        if not neuron_id:
            return {"error": "neuron_id is required for pin action"}

        brain = await storage.get_brain(storage.brain_id)  # type: ignore[arg-type]
        if brain is None:
            return {"error": "No brain configured"}

        result = await pin_as_reflex(neuron_id, storage, brain.config)

        if not result.pinned:
            return {"error": result.error or "Failed to pin reflex"}

        response: dict[str, Any] = {
            "status": "pinned",
            "neuron_id": neuron_id,
        }
        if result.conflicts_resolved:
            response["conflicts_resolved"] = [
                {
                    "superseded_id": c.existing_id,
                    "superseded_content": c.existing_content[:100],
                    "hamming_distance": c.hamming_distance,
                }
                for c in result.conflicts_resolved
            ]
        return response

    async def _reflex_unpin(self, args: dict[str, Any]) -> dict[str, Any]:
        """Unpin a neuron from reflex status."""
        from neural_memory.engine.reflex_conflict import unpin_reflex

        storage = await self.get_storage()
        _require_brain_id(storage)

        neuron_id = (args.get("neuron_id") or "").strip()
        if not neuron_id:
            return {"error": "neuron_id is required for unpin action"}

        success = await unpin_reflex(neuron_id, storage)
        if not success:
            return {"error": "Neuron not found or not a reflex"}
        return {"status": "unpinned", "neuron_id": neuron_id}

    async def _reflex_list(self, args: dict[str, Any]) -> dict[str, Any]:
        """List all reflex neurons for current brain."""
        storage = await self.get_storage()
        _require_brain_id(storage)

        limit = min(args.get("limit", 20), 50)
        reflexes = await storage.find_reflex_neurons(limit=limit)

        return {
            "reflex_count": len(reflexes),
            "reflexes": [
                {
                    "neuron_id": n.id,
                    "type": str(n.type),
                    "content": n.content[:200],
                    "created_at": n.created_at.isoformat() if n.created_at else None,
                }
                for n in reflexes
            ],
        }
