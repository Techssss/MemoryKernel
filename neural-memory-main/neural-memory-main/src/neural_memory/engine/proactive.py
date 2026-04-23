"""Proactive memory engine — surface primed memories as hints.

Bridges the gap between the priming engine (which computes activation
boosts) and the MCP response layer (which presents results to the agent).
The priming engine already knows which neurons are "warming up" — this
module selects the most relevant ones and formats them as hints that
piggyback on tool responses.

Design: agent never calls a "proactive" tool. Hints appear automatically
in recall/recap/context responses when priming has relevant data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.engine.priming import PrimingResult
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

DEFAULT_MAX_HINTS = 3
DEFAULT_MAX_HINT_CHARS = 500
DEFAULT_MIN_ACTIVATION = 0.3
DEFAULT_SKIP_HIGH_CONFIDENCE = 0.9
MIN_USEFUL_HINT_CHARS = 20  # Skip hints truncated below this length


# ── Data Models ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProactiveHint:
    """A single proactive memory hint surfaced from priming."""

    neuron_id: str
    content: str
    activation_level: float
    source: str  # "cache", "topic", "habit", "co_activation"
    neuron_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "content": self.content,
            "source": self.source,
            "activation": round(self.activation_level, 3),
        }
        if self.neuron_type:
            result["type"] = self.neuron_type
        return result


# ── Core Logic ────────────────────────────────────────────────────────


async def select_proactive_hints(
    priming_result: PrimingResult,
    storage: NeuralStorage,
    result_neuron_ids: set[str] | None = None,
    max_hints: int = DEFAULT_MAX_HINTS,
    max_chars: int = DEFAULT_MAX_HINT_CHARS,
    min_activation: float = DEFAULT_MIN_ACTIVATION,
) -> list[ProactiveHint]:
    """Select top primed neurons as proactive hints.

    Filters out neurons already in the recall result, enforces budget
    limits, and returns formatted hints sorted by activation level.

    Args:
        priming_result: Combined priming output from compute_priming().
        storage: Storage backend for neuron content lookup.
        result_neuron_ids: Neuron IDs already returned in the main result
            (these are excluded to avoid redundancy).
        max_hints: Maximum number of hints to return.
        max_chars: Maximum total characters across all hint contents.
        min_activation: Minimum activation level to qualify as a hint.

    Returns:
        List of ProactiveHint, sorted by activation (highest first).
    """
    if not priming_result.activation_boosts:
        return []

    already_returned = result_neuron_ids or set()

    # Filter and sort candidates by activation level (descending)
    candidates: list[tuple[str, float]] = [
        (nid, level)
        for nid, level in priming_result.activation_boosts.items()
        if level >= min_activation and nid not in already_returned
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    if not candidates:
        return []

    # Determine which priming source each neuron came from
    source_map = _build_source_map(priming_result)

    # Select top candidates within budget
    hints: list[ProactiveHint] = []
    total_chars = 0

    for nid, level in candidates:
        if len(hints) >= max_hints:
            break
        if total_chars >= max_chars:
            break

        try:
            neuron = await storage.get_neuron(nid)
            if neuron is None:
                continue

            content = neuron.content or ""
            if not content.strip():
                continue

            # Truncate individual hint if needed
            remaining_budget = max_chars - total_chars
            if len(content) > remaining_budget:
                if remaining_budget < MIN_USEFUL_HINT_CHARS:
                    break  # Not enough room for a useful hint
                content = content[:remaining_budget].rsplit(" ", 1)[0] + "..."
                if len(content) < MIN_USEFUL_HINT_CHARS:
                    continue  # Truncation left too little

            hint = ProactiveHint(
                neuron_id=nid,
                content=content,
                activation_level=level,
                source=source_map.get(nid, "unknown"),
                neuron_type=neuron.type.value if neuron.type else "",
            )
            hints.append(hint)
            total_chars += len(content)

        except Exception:
            logger.debug("Failed to load neuron %s for proactive hint", nid, exc_info=True)

    return hints


def _build_source_map(priming_result: PrimingResult) -> dict[str, str]:
    """Determine the primary priming source for each neuron.

    Since PrimingResult.source_counts only has per-source neuron counts
    (not per-neuron attribution), we assign the most-represented source
    as the default, with priority tie-breaking for equal counts.

    The source label is informational (shown to agent), not functional.
    """
    source_priority = {"habit": 4, "co_activation": 3, "topic": 2, "cache": 1}
    source_map: dict[str, str] = {}

    if not priming_result.source_counts:
        return source_map

    # Pick source with most neurons; tie-break by priority
    best_source = max(
        priming_result.source_counts.keys(),
        key=lambda s: (priming_result.source_counts[s], source_priority.get(s, 0)),
    )

    for nid in priming_result.activation_boosts:
        source_map[nid] = best_source

    return source_map


def format_hints_for_response(hints: list[ProactiveHint]) -> list[dict[str, Any]]:
    """Format hints for inclusion in MCP tool response."""
    return [h.to_dict() for h in hints]
