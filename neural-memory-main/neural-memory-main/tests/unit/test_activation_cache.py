"""Unit tests for activation cache module."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from neural_memory.cache.models import ActivationCache, CachedState
from neural_memory.cache.serializer import (
    cache_exists,
    delete_cache,
    load_cache,
    save_cache,
)


class TestCachedState:
    """Tests for CachedState dataclass."""

    def test_create_minimal(self) -> None:
        """Test creating CachedState with minimal args."""
        state = CachedState(neuron_id="n1", activation_level=0.75)
        assert state.neuron_id == "n1"
        assert state.activation_level == 0.75
        assert state.access_frequency == 0
        assert state.last_activated is None

    def test_create_full(self) -> None:
        """Test creating CachedState with all args."""
        now = datetime(2026, 1, 1, 12, 0, 0)
        state = CachedState(
            neuron_id="n2",
            activation_level=0.95,
            access_frequency=10,
            last_activated=now,
        )
        assert state.neuron_id == "n2"
        assert state.activation_level == 0.95
        assert state.access_frequency == 10
        assert state.last_activated == now

    def test_frozen(self) -> None:
        """Test that CachedState is immutable."""
        state = CachedState(neuron_id="n1", activation_level=0.5)
        with pytest.raises(AttributeError):
            state.activation_level = 0.9  # type: ignore[misc]

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        now = datetime(2026, 1, 1, 12, 0, 0)
        state = CachedState(
            neuron_id="n1",
            activation_level=0.8,
            access_frequency=5,
            last_activated=now,
        )
        d = state.to_dict()
        assert d["neuron_id"] == "n1"
        assert d["activation_level"] == 0.8
        assert d["access_frequency"] == 5
        assert d["last_activated"] == "2026-01-01T12:00:00"

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        d = {
            "neuron_id": "n2",
            "activation_level": 0.65,
            "access_frequency": 3,
            "last_activated": "2026-02-15T09:30:00",
        }
        state = CachedState.from_dict(d)
        assert state.neuron_id == "n2"
        assert state.activation_level == 0.65
        assert state.access_frequency == 3
        assert state.last_activated == datetime(2026, 2, 15, 9, 30, 0)

    def test_from_dict_minimal(self) -> None:
        """Test deserialization with minimal fields."""
        d = {"neuron_id": "n3"}
        state = CachedState.from_dict(d)
        assert state.neuron_id == "n3"
        assert state.activation_level == 0.0
        assert state.access_frequency == 0
        assert state.last_activated is None

    def test_roundtrip(self) -> None:
        """Test to_dict -> from_dict roundtrip."""
        original = CachedState(
            neuron_id="test",
            activation_level=0.123,
            access_frequency=42,
            last_activated=datetime(2026, 3, 10, 15, 45, 30),
        )
        roundtrip = CachedState.from_dict(original.to_dict())
        assert roundtrip == original


class TestActivationCache:
    """Tests for ActivationCache dataclass."""

    def test_create_minimal(self) -> None:
        """Test creating ActivationCache with minimal args."""
        now = datetime(2026, 1, 1, 12, 0, 0)
        cache = ActivationCache(
            brain_id="brain1",
            brain_name="default",
            cached_at=now,
        )
        assert cache.brain_id == "brain1"
        assert cache.brain_name == "default"
        assert cache.cached_at == now
        assert cache.ttl_hours == 24
        assert cache.entries == ()
        assert cache.brain_hash == ""

    def test_entry_count(self) -> None:
        """Test entry_count property."""
        states = (
            CachedState(neuron_id="n1", activation_level=0.9),
            CachedState(neuron_id="n2", activation_level=0.8),
            CachedState(neuron_id="n3", activation_level=0.7),
        )
        cache = ActivationCache(
            brain_id="b1",
            brain_name="test",
            cached_at=datetime.now(),
            entries=states,
        )
        assert cache.entry_count == 3

    def test_is_expired_not_expired(self) -> None:
        """Test is_expired returns False when within TTL."""
        now = datetime(2026, 1, 1, 12, 0, 0)
        cache = ActivationCache(
            brain_id="b1",
            brain_name="test",
            cached_at=now,
            ttl_hours=24,
        )
        # Check 12 hours later
        check_time = now + timedelta(hours=12)
        assert not cache.is_expired(check_time)

    def test_is_expired_expired(self) -> None:
        """Test is_expired returns True when past TTL."""
        now = datetime(2026, 1, 1, 12, 0, 0)
        cache = ActivationCache(
            brain_id="b1",
            brain_name="test",
            cached_at=now,
            ttl_hours=24,
        )
        # Check 25 hours later
        check_time = now + timedelta(hours=25)
        assert cache.is_expired(check_time)

    def test_get_state_found(self) -> None:
        """Test get_state returns state when found."""
        state = CachedState(neuron_id="target", activation_level=0.95)
        cache = ActivationCache(
            brain_id="b1",
            brain_name="test",
            cached_at=datetime.now(),
            entries=(state,),
        )
        result = cache.get_state("target")
        assert result == state

    def test_get_state_not_found(self) -> None:
        """Test get_state returns None when not found."""
        cache = ActivationCache(
            brain_id="b1",
            brain_name="test",
            cached_at=datetime.now(),
            entries=(),
        )
        assert cache.get_state("missing") is None

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        now = datetime(2026, 1, 1, 12, 0, 0)
        states = (CachedState(neuron_id="n1", activation_level=0.8),)
        cache = ActivationCache(
            brain_id="brain1",
            brain_name="test",
            cached_at=now,
            ttl_hours=12,
            entries=states,
            brain_hash="abc123",
        )
        d = cache.to_dict()
        assert d["brain_id"] == "brain1"
        assert d["brain_name"] == "test"
        assert d["cached_at"] == "2026-01-01T12:00:00"
        assert d["ttl_hours"] == 12
        assert len(d["entries"]) == 1
        assert d["brain_hash"] == "abc123"

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        d = {
            "brain_id": "b2",
            "brain_name": "default",
            "cached_at": "2026-03-15T10:00:00",
            "ttl_hours": 48,
            "entries": [{"neuron_id": "n1", "activation_level": 0.7}],
            "brain_hash": "xyz789",
        }
        cache = ActivationCache.from_dict(d)
        assert cache.brain_id == "b2"
        assert cache.brain_name == "default"
        assert cache.cached_at == datetime(2026, 3, 15, 10, 0, 0)
        assert cache.ttl_hours == 48
        assert cache.entry_count == 1
        assert cache.brain_hash == "xyz789"

    def test_roundtrip(self) -> None:
        """Test to_dict -> from_dict roundtrip."""
        states = (
            CachedState(
                neuron_id="n1",
                activation_level=0.9,
                access_frequency=5,
                last_activated=datetime(2026, 1, 1, 12, 0, 0),
            ),
        )
        original = ActivationCache(
            brain_id="b1",
            brain_name="test",
            cached_at=datetime(2026, 1, 1, 12, 0, 0),
            ttl_hours=24,
            entries=states,
            brain_hash="hash123",
        )
        roundtrip = ActivationCache.from_dict(original.to_dict())
        assert roundtrip.brain_id == original.brain_id
        assert roundtrip.brain_name == original.brain_name
        assert roundtrip.cached_at == original.cached_at
        assert roundtrip.ttl_hours == original.ttl_hours
        assert roundtrip.entry_count == original.entry_count
        assert roundtrip.brain_hash == original.brain_hash


class TestSerializer:
    """Tests for cache serialization functions."""

    def test_save_and_load_json(self) -> None:
        """Test save/load cycle with JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            states = (CachedState(neuron_id="n1", activation_level=0.85),)
            cache = ActivationCache(
                brain_id="b1",
                brain_name="test_brain",
                cached_at=datetime(2026, 1, 1, 12, 0, 0),
                entries=states,
            )

            # Save (force JSON by disabling msgpack)
            path = save_cache(cache, data_dir, use_msgpack=False)
            assert path.exists()
            assert path.suffix == ".cache"

            # Load
            loaded = load_cache("test_brain", data_dir)
            assert loaded is not None
            assert loaded.brain_id == "b1"
            assert loaded.entry_count == 1

    def test_cache_exists(self) -> None:
        """Test cache_exists function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            assert not cache_exists("missing", data_dir)

            # Create a cache
            cache = ActivationCache(
                brain_id="b1",
                brain_name="exists",
                cached_at=datetime.now(),
            )
            save_cache(cache, data_dir, use_msgpack=False)
            assert cache_exists("exists", data_dir)

    def test_delete_cache(self) -> None:
        """Test delete_cache function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create and delete
            cache = ActivationCache(
                brain_id="b1",
                brain_name="deleteme",
                cached_at=datetime.now(),
            )
            save_cache(cache, data_dir, use_msgpack=False)
            assert cache_exists("deleteme", data_dir)

            deleted = delete_cache("deleteme", data_dir)
            assert deleted
            assert not cache_exists("deleteme", data_dir)

    def test_delete_nonexistent(self) -> None:
        """Test delete_cache returns False for nonexistent cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            deleted = delete_cache("nonexistent", data_dir)
            assert not deleted

    def test_load_nonexistent(self) -> None:
        """Test load_cache returns None for nonexistent cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            loaded = load_cache("nonexistent", data_dir)
            assert loaded is None

    def test_load_corrupted(self) -> None:
        """Test load_cache returns None for corrupted cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            cache_path = data_dir / "corrupted.activation.cache"
            cache_path.write_text("not valid json")

            loaded = load_cache("corrupted", data_dir)
            assert loaded is None

    def test_path_traversal_sanitized(self) -> None:
        """Test that path traversal attempts are sanitized."""
        from neural_memory.cache.serializer import _get_cache_path, _sanitize_brain_name

        # Test sanitization function
        assert _sanitize_brain_name("../../etc/passwd") == "____etc_passwd"
        assert _sanitize_brain_name("..\\..\\windows\\system32") == "____windows_system32"
        assert _sanitize_brain_name("normal_brain") == "normal_brain"
        assert _sanitize_brain_name("") == "default"

        # BUG 3: null bytes must be stripped
        assert "\x00" not in _sanitize_brain_name("brain\x00evil")

        # Test that _get_cache_path stays within data_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            # Should not raise — sanitized path stays within data_dir
            path = _get_cache_path("../../etc/passwd", data_dir)
            assert path.is_relative_to(data_dir)

    def test_load_with_corrupt_entries(self) -> None:
        """BUG 2: from_dict must skip non-dict entries, not crash."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            cache_path = data_dir / "corrupt_entries.activation.cache"
            # Mix of valid dict, None, string, int
            payload = {
                "brain_id": "b1",
                "brain_name": "corrupt_entries",
                "cached_at": "2026-01-01T12:00:00",
                "ttl_hours": 24,
                "entries": [
                    {"neuron_id": "n1", "activation_level": 0.9},
                    None,
                    "not a dict",
                    42,
                    {"neuron_id": "n2", "activation_level": 0.5},
                ],
                "brain_hash": "",
            }
            cache_path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_cache("corrupt_entries", data_dir)
            assert loaded is not None
            # Only valid dict entries are kept
            assert loaded.entry_count == 2


class TestBugRegressions:
    """Regression tests for bugs found during review."""

    def test_ttl_zero_is_expired(self) -> None:
        """BUG 5: ttl_hours=0 should mark cache as expired immediately."""
        now = datetime(2026, 1, 1, 12, 0, 0)
        cache = ActivationCache(brain_id="b1", brain_name="t", cached_at=now, ttl_hours=0)
        assert cache.is_expired(now)

    def test_ttl_negative_is_expired(self) -> None:
        """BUG 5: negative ttl_hours should mark cache as expired."""
        now = datetime(2026, 1, 1, 12, 0, 0)
        cache = ActivationCache(brain_id="b1", brain_name="t", cached_at=now, ttl_hours=-5)
        assert cache.is_expired(now)

    def test_backward_clock_skew_expires(self) -> None:
        """BUG 6: cached_at in the future (clock skew) must not be immortal."""
        now = datetime(2026, 1, 1, 12, 0, 0)
        future = datetime(2026, 1, 2, 12, 0, 0)  # cached_at > now
        cache = ActivationCache(brain_id="b1", brain_name="t", cached_at=future, ttl_hours=24)
        assert cache.is_expired(now)

    def test_manager_clamps_invalid_ttl(self) -> None:
        """BUG 5: manager must clamp ttl_hours < 1 to 1."""
        from unittest.mock import MagicMock

        from neural_memory.cache.manager import ActivationCacheManager

        storage = MagicMock()
        mgr = ActivationCacheManager(storage, ttl_hours=0)
        assert mgr._ttl_hours == 1

        mgr = ActivationCacheManager(storage, ttl_hours=-10)
        assert mgr._ttl_hours == 1
