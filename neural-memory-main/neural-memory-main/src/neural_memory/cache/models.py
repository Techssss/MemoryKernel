"""Frozen dataclasses for activation state caching.

Stores computational state (activation levels) separate from semantic
structure (surface.nm) to enable warm-start recall.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CachedState:
    """A single cached neuron activation state.

    Attributes:
        neuron_id: UUID of the neuron
        activation_level: Activation at cache time (0.0-1.0)
        access_frequency: How often accessed in session
        last_activated: When last activated
    """

    neuron_id: str
    activation_level: float
    access_frequency: int = 0
    last_activated: datetime | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize to dict for JSON/msgpack."""
        return {
            "neuron_id": self.neuron_id,
            "activation_level": self.activation_level,
            "access_frequency": self.access_frequency,
            "last_activated": (self.last_activated.isoformat() if self.last_activated else None),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CachedState:
        """Deserialize from dict."""
        last_activated = data.get("last_activated")
        activation_raw = data.get("activation_level")
        freq_raw = data.get("access_frequency")

        # Parse activation_level with type safety
        activation_level = 0.0
        if isinstance(activation_raw, (int, float)):
            activation_level = float(activation_raw)
        elif activation_raw is not None:
            activation_level = float(str(activation_raw))

        # Parse access_frequency with type safety
        access_frequency = 0
        if isinstance(freq_raw, int):
            access_frequency = freq_raw
        elif freq_raw is not None:
            access_frequency = int(str(freq_raw))

        return cls(
            neuron_id=str(data["neuron_id"]),
            activation_level=activation_level,
            access_frequency=access_frequency,
            last_activated=(
                datetime.fromisoformat(str(last_activated)) if last_activated else None
            ),
        )


@dataclass(frozen=True)
class ActivationCache:
    """Complete activation cache for a brain.

    Attributes:
        brain_id: UUID of the brain
        brain_name: Human-readable brain name
        cached_at: When cache was created
        ttl_hours: Time-to-live in hours (default 24)
        entries: Cached activation states
        brain_hash: Hash of brain state for staleness detection
    """

    brain_id: str
    brain_name: str
    cached_at: datetime
    ttl_hours: int = 24
    entries: tuple[CachedState, ...] = ()
    brain_hash: str = ""

    @property
    def entry_count(self) -> int:
        """Number of cached entries."""
        return len(self.entries)

    def is_expired(self, now: datetime | None = None) -> bool:
        """Check if cache has exceeded TTL.

        Treats ttl_hours <= 0 as immediately expired (invalid config).
        Treats negative age (backward clock skew) as expired for safety.
        """
        from neural_memory.utils.timeutils import utcnow

        if self.ttl_hours <= 0:
            return True

        now = now or utcnow()
        age_hours = (now - self.cached_at).total_seconds() / 3600
        if age_hours < 0:
            return True
        return age_hours > self.ttl_hours

    def get_state(self, neuron_id: str) -> CachedState | None:
        """Look up cached state by neuron ID."""
        for entry in self.entries:
            if entry.neuron_id == neuron_id:
                return entry
        return None

    def to_dict(self) -> dict[str, object]:
        """Serialize to dict for JSON/msgpack."""
        return {
            "brain_id": self.brain_id,
            "brain_name": self.brain_name,
            "cached_at": self.cached_at.isoformat(),
            "ttl_hours": self.ttl_hours,
            "entries": [e.to_dict() for e in self.entries],
            "brain_hash": self.brain_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ActivationCache:
        """Deserialize from dict. Skips corrupt entries rather than crashing."""
        entries_raw = data.get("entries", [])
        entries_list: list[object] = list(entries_raw) if isinstance(entries_raw, list) else []
        parsed: list[CachedState] = []
        for entry in entries_list:
            if not isinstance(entry, dict):
                continue
            try:
                parsed.append(CachedState.from_dict(entry))
            except (KeyError, ValueError, TypeError):
                continue
        entries = tuple(parsed)

        # Parse ttl_hours with type safety
        ttl_raw = data.get("ttl_hours")
        ttl_hours = 24
        if isinstance(ttl_raw, int):
            ttl_hours = ttl_raw
        elif ttl_raw is not None:
            ttl_hours = int(str(ttl_raw))

        return cls(
            brain_id=str(data["brain_id"]),
            brain_name=str(data.get("brain_name", "default")),
            cached_at=datetime.fromisoformat(str(data["cached_at"])),
            ttl_hours=ttl_hours,
            entries=entries,
            brain_hash=str(data.get("brain_hash", "")),
        )
