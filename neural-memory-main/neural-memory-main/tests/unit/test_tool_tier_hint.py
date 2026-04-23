"""Tests for tool tier recommendation hint in stats."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from neural_memory.mcp.tool_schemas import TOOL_TIERS


def _make_handler(tier: str = "full") -> MagicMock:
    """Create a minimal StatsHandler-like object with _tool_tier_hint."""
    from neural_memory.mcp.stats_handler import StatsHandler

    handler = StatsHandler.__new__(StatsHandler)
    handler.config = MagicMock()
    handler.config.tool_tier.tier = tier
    return handler


def _make_storage(top_tools: list[dict]) -> AsyncMock:
    """Create mock storage returning given tool stats."""
    storage = AsyncMock()
    storage.get_tool_stats = AsyncMock(
        return_value={
            "total_events": sum(t.get("count", 0) for t in top_tools),
            "success_rate": 0.95,
            "top_tools": top_tools,
        }
    )
    return storage


class TestToolTierHint:
    """Tool tier recommendations based on usage data."""

    @pytest.mark.asyncio
    async def test_minimal_recommendation(self) -> None:
        """User on 'full' using only minimal tools → suggest minimal."""
        handler = _make_handler("full")
        storage = _make_storage(
            [
                {"tool_name": "nmem_remember", "server_name": "neural-memory", "count": 50},
                {"tool_name": "nmem_recall", "server_name": "neural-memory", "count": 30},
            ]
        )

        hint = await handler._tool_tier_hint(storage, "brain-1")
        assert hint is not None
        assert "minimal" in hint
        assert "80%" in hint

    @pytest.mark.asyncio
    async def test_standard_recommendation(self) -> None:
        """User on 'full' using standard-tier tools → suggest standard."""
        handler = _make_handler("full")
        storage = _make_storage(
            [
                {"tool_name": "nmem_remember", "server_name": "neural-memory", "count": 50},
                {"tool_name": "nmem_recall", "server_name": "neural-memory", "count": 30},
                {"tool_name": "nmem_auto", "server_name": "neural-memory", "count": 10},
                {"tool_name": "nmem_session", "server_name": "neural-memory", "count": 5},
            ]
        )

        hint = await handler._tool_tier_hint(storage, "brain-1")
        assert hint is not None
        assert "standard" in hint
        assert "60%" in hint

    @pytest.mark.asyncio
    async def test_no_hint_on_full_usage(self) -> None:
        """User using tools outside standard tier → no recommendation."""
        handler = _make_handler("full")
        storage = _make_storage(
            [
                {"tool_name": "nmem_remember", "server_name": "neural-memory", "count": 50},
                {"tool_name": "nmem_recall", "server_name": "neural-memory", "count": 30},
                {"tool_name": "nmem_consolidate", "server_name": "neural-memory", "count": 10},
                {"tool_name": "nmem_health", "server_name": "neural-memory", "count": 5},
            ]
        )

        hint = await handler._tool_tier_hint(storage, "brain-1")
        assert hint is None

    @pytest.mark.asyncio
    async def test_no_hint_when_already_minimal(self) -> None:
        """User already on 'minimal' → no recommendation."""
        handler = _make_handler("minimal")
        storage = _make_storage(
            [
                {"tool_name": "nmem_remember", "server_name": "neural-memory", "count": 50},
            ]
        )

        hint = await handler._tool_tier_hint(storage, "brain-1")
        assert hint is None

    @pytest.mark.asyncio
    async def test_no_hint_when_already_standard(self) -> None:
        """User already on 'standard' → no recommendation."""
        handler = _make_handler("standard")
        storage = _make_storage(
            [
                {"tool_name": "nmem_remember", "server_name": "neural-memory", "count": 50},
            ]
        )

        hint = await handler._tool_tier_hint(storage, "brain-1")
        assert hint is None

    @pytest.mark.asyncio
    async def test_no_hint_without_usage_data(self) -> None:
        """No tool events recorded → no recommendation."""
        handler = _make_handler("full")
        storage = _make_storage([])

        hint = await handler._tool_tier_hint(storage, "brain-1")
        assert hint is None

    @pytest.mark.asyncio
    async def test_ignores_non_nm_tools(self) -> None:
        """Non-NM tools should not affect the recommendation."""
        handler = _make_handler("full")
        storage = _make_storage(
            [
                {"tool_name": "nmem_remember", "server_name": "neural-memory", "count": 50},
                {"tool_name": "nmem_recall", "server_name": "neural-memory", "count": 30},
                # Non-NM tool — should be ignored
                {"tool_name": "web_search", "server_name": "brave", "count": 100},
            ]
        )

        hint = await handler._tool_tier_hint(storage, "brain-1")
        assert hint is not None
        assert "minimal" in hint

    @pytest.mark.asyncio
    async def test_nmem_prefix_detection(self) -> None:
        """Tools with nmem_ prefix detected even if server_name differs."""
        handler = _make_handler("full")
        storage = _make_storage(
            [
                {"tool_name": "nmem_remember", "server_name": "custom-server", "count": 50},
                {"tool_name": "nmem_recall", "server_name": "custom-server", "count": 30},
            ]
        )

        hint = await handler._tool_tier_hint(storage, "brain-1")
        assert hint is not None
        assert "minimal" in hint

    @pytest.mark.asyncio
    async def test_minimal_tier_is_subset_of_standard(self) -> None:
        """Sanity: minimal tools should be a subset of standard tools."""
        assert TOOL_TIERS["minimal"] < TOOL_TIERS["standard"]
