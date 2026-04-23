"""Tests for anti-redundancy: attention set on SessionState."""

from __future__ import annotations

from neural_memory.engine.session_state import SessionState


class TestAttentionSet:
    """Tests for surfaced_fiber_ids tracking and FIFO eviction."""

    def test_initial_empty(self) -> None:
        state = SessionState(session_id="s1")
        assert state.surfaced_fiber_ids == []
        assert not state.is_surfaced("f1")

    def test_record_and_check(self) -> None:
        state = SessionState(session_id="s1")
        state.record_surfaced(["f1", "f2", "f3"])
        assert state.is_surfaced("f1")
        assert state.is_surfaced("f2")
        assert state.is_surfaced("f3")
        assert not state.is_surfaced("f4")

    def test_no_duplicates(self) -> None:
        state = SessionState(session_id="s1")
        state.record_surfaced(["f1", "f2"])
        state.record_surfaced(["f2", "f3"])
        assert len(state.surfaced_fiber_ids) == 3
        assert state.surfaced_fiber_ids == ["f1", "f2", "f3"]

    def test_fifo_eviction(self) -> None:
        state = SessionState(session_id="s1")
        # Fill to exactly MAX_SURFACED
        ids = [f"f{i}" for i in range(500)]
        state.record_surfaced(ids)
        assert len(state.surfaced_fiber_ids) == 500
        assert state.is_surfaced("f0")
        assert state.is_surfaced("f499")

        # Add one more — should evict f0
        state.record_surfaced(["f500"])
        assert len(state.surfaced_fiber_ids) == 500
        assert not state.is_surfaced("f0")
        assert state.is_surfaced("f1")
        assert state.is_surfaced("f500")

    def test_fifo_eviction_bulk(self) -> None:
        state = SessionState(session_id="s1")
        ids = [f"f{i}" for i in range(510)]
        state.record_surfaced(ids)
        assert len(state.surfaced_fiber_ids) == 500
        # First 10 should be evicted
        assert not state.is_surfaced("f0")
        assert not state.is_surfaced("f9")
        assert state.is_surfaced("f10")
        assert state.is_surfaced("f509")

    def test_record_empty_list(self) -> None:
        state = SessionState(session_id="s1")
        state.record_surfaced([])
        assert state.surfaced_fiber_ids == []

    def test_multiple_batches(self) -> None:
        state = SessionState(session_id="s1")
        state.record_surfaced(["f1", "f2"])
        state.record_surfaced(["f3"])
        state.record_surfaced(["f4", "f5"])
        assert len(state.surfaced_fiber_ids) == 5
        assert state.is_surfaced("f1")
        assert state.is_surfaced("f5")
