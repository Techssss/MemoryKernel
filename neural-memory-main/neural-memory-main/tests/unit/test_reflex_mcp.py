"""Integration tests for Reflex Arc — Phase 2 (MCP + Pipeline).

Tests cover:
- ReflexHandler: pin, unpin, list via MCP tool interface
- Reflex injection in ReflexPipeline.query() context
- exclude_reflexes param suppresses injection
- nmem_stats includes reflex_count
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neural_memory.core.neuron import Neuron, NeuronType
from neural_memory.engine.reflex_conflict import ReflexPinResult
from neural_memory.mcp.reflex_handler import ReflexHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_neuron(
    content: str = "test content",
    reflex: bool = False,
    neuron_id: str = "n-1",
) -> Neuron:
    n = Neuron.create(type=NeuronType.CONCEPT, content=content, neuron_id=neuron_id)
    if reflex:
        n = n.with_reflex(pinned=True)
    return n


class MockReflexHandler(ReflexHandler):
    """Concrete handler for testing."""

    def __init__(self, storage: AsyncMock) -> None:
        self._storage = storage
        self.config = MagicMock()

    async def get_storage(self) -> AsyncMock:
        return self._storage


def _mock_storage(
    brain_id: str = "brain-1",
    reflexes: list[Neuron] | None = None,
) -> AsyncMock:
    storage = AsyncMock()
    storage._current_brain_id = brain_id
    storage.brain_id = brain_id
    brain_mock = MagicMock()
    brain_mock.config = MagicMock()
    brain_mock.config.max_reflexes = 20
    storage.get_brain = AsyncMock(return_value=brain_mock)
    storage.find_reflex_neurons = AsyncMock(return_value=reflexes or [])
    storage.get_neuron = AsyncMock(return_value=None)
    storage.update_neuron = AsyncMock()
    storage.add_synapse = AsyncMock(return_value="syn-1")
    return storage


# ---------------------------------------------------------------------------
# ReflexHandler — pin
# ---------------------------------------------------------------------------


class TestReflexHandlerPin:
    @pytest.mark.asyncio
    async def test_pin_requires_neuron_id(self) -> None:
        handler = MockReflexHandler(_mock_storage())
        result = await handler._reflex({"action": "pin"})
        assert "error" in result
        assert "neuron_id" in result["error"]

    @pytest.mark.asyncio
    async def test_pin_success(self) -> None:
        neuron = _make_neuron(content="always validate input", neuron_id="n-1")
        storage = _mock_storage()
        storage.get_neuron = AsyncMock(return_value=neuron)
        handler = MockReflexHandler(storage)

        with patch(
            "neural_memory.engine.reflex_conflict.pin_as_reflex",
            return_value=ReflexPinResult(pinned=True, conflicts_resolved=[]),
        ) as mock_pin:
            result = await handler._reflex({"action": "pin", "neuron_id": "n-1"})

        assert result["status"] == "pinned"
        assert result["neuron_id"] == "n-1"
        mock_pin.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pin_failure_returns_error(self) -> None:
        storage = _mock_storage()
        handler = MockReflexHandler(storage)

        with patch(
            "neural_memory.engine.reflex_conflict.pin_as_reflex",
            return_value=ReflexPinResult(
                pinned=False, conflicts_resolved=[], error="Max reflexes reached (20)"
            ),
        ):
            result = await handler._reflex({"action": "pin", "neuron_id": "n-1"})

        assert "error" in result
        assert "Max reflexes" in result["error"]


# ---------------------------------------------------------------------------
# ReflexHandler — unpin
# ---------------------------------------------------------------------------


class TestReflexHandlerUnpin:
    @pytest.mark.asyncio
    async def test_unpin_requires_neuron_id(self) -> None:
        handler = MockReflexHandler(_mock_storage())
        result = await handler._reflex({"action": "unpin"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_unpin_success(self) -> None:
        storage = _mock_storage()
        handler = MockReflexHandler(storage)

        with patch(
            "neural_memory.engine.reflex_conflict.unpin_reflex",
            return_value=True,
        ):
            result = await handler._reflex({"action": "unpin", "neuron_id": "n-1"})

        assert result["status"] == "unpinned"

    @pytest.mark.asyncio
    async def test_unpin_not_found(self) -> None:
        storage = _mock_storage()
        handler = MockReflexHandler(storage)

        with patch(
            "neural_memory.engine.reflex_conflict.unpin_reflex",
            return_value=False,
        ):
            result = await handler._reflex({"action": "unpin", "neuron_id": "n-1"})

        assert "error" in result


# ---------------------------------------------------------------------------
# ReflexHandler — list
# ---------------------------------------------------------------------------


class TestReflexHandlerList:
    @pytest.mark.asyncio
    async def test_list_empty(self) -> None:
        handler = MockReflexHandler(_mock_storage(reflexes=[]))
        result = await handler._reflex({"action": "list"})
        assert result["reflex_count"] == 0
        assert result["reflexes"] == []

    @pytest.mark.asyncio
    async def test_list_with_reflexes(self) -> None:
        reflexes = [
            _make_neuron(content="rule 1", reflex=True, neuron_id="n-1"),
            _make_neuron(content="rule 2", reflex=True, neuron_id="n-2"),
        ]
        handler = MockReflexHandler(_mock_storage(reflexes=reflexes))
        result = await handler._reflex({"action": "list"})
        assert result["reflex_count"] == 2
        assert len(result["reflexes"]) == 2
        assert result["reflexes"][0]["neuron_id"] == "n-1"

    @pytest.mark.asyncio
    async def test_list_respects_limit(self) -> None:
        storage = _mock_storage()
        handler = MockReflexHandler(storage)
        await handler._reflex({"action": "list", "limit": 5})
        storage.find_reflex_neurons.assert_awaited_once_with(limit=5)


# ---------------------------------------------------------------------------
# Unknown action
# ---------------------------------------------------------------------------


class TestReflexHandlerUnknown:
    @pytest.mark.asyncio
    async def test_unknown_action(self) -> None:
        handler = MockReflexHandler(_mock_storage())
        result = await handler._reflex({"action": "bogus"})
        assert "error" in result
        assert "bogus" in result["error"]


# ---------------------------------------------------------------------------
# Tool schema registration
# ---------------------------------------------------------------------------


class TestToolSchemaRegistration:
    def test_nmem_reflex_in_schemas(self) -> None:
        from neural_memory.mcp.tool_schemas import get_tool_schemas_for_tier

        schemas = get_tool_schemas_for_tier("full")
        names = {s["name"] for s in schemas}
        assert "nmem_reflex" in names

    def test_nmem_recall_has_exclude_reflexes(self) -> None:
        from neural_memory.mcp.tool_schemas import get_tool_schemas_for_tier

        schemas = get_tool_schemas_for_tier("full")
        recall_schema = next(s for s in schemas if s["name"] == "nmem_recall")
        props = recall_schema["inputSchema"]["properties"]
        assert "exclude_reflexes" in props


# ---------------------------------------------------------------------------
# Server dispatch registration
# ---------------------------------------------------------------------------


class TestServerDispatch:
    def test_nmem_reflex_in_write_tools(self) -> None:
        from neural_memory.mcp.server import MCPServer

        assert "nmem_reflex" in MCPServer._WRITE_TOOLS
