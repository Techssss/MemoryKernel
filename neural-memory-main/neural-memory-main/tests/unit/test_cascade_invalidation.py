"""Tests for engine/cascade_invalidation.py — cascade staleness propagation."""

from __future__ import annotations

import pytest

from neural_memory.engine.cascade_invalidation import CascadeReport


class TestCascadeReport:
    """Tests for CascadeReport frozen dataclass."""

    def test_creation(self) -> None:
        report = CascadeReport(
            neurons_marked=3,
            fibers_marked=2,
            depth_reached=2,
        )
        assert report.neurons_marked == 3
        assert report.fibers_marked == 2
        assert report.depth_reached == 2

    def test_immutable(self) -> None:
        report = CascadeReport(neurons_marked=0, fibers_marked=0, depth_reached=0)
        with pytest.raises(AttributeError):
            report.neurons_marked = 5  # type: ignore[misc]

    def test_zero_report(self) -> None:
        """Empty cascade when no downstream neurons exist."""
        report = CascadeReport(neurons_marked=0, fibers_marked=0, depth_reached=0)
        assert report.neurons_marked == 0
        assert report.depth_reached == 0
