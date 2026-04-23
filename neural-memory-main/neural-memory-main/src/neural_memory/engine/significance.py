"""Significance weighting engine — amygdala boost for auto-capture.

Scores detected memories for emotional/cognitive significance before saving.
High-significance content (contradictions, corrections, novel topics) gets
boosted priority; near-duplicate content gets deprioritized.

Uses the existing prediction_error engine for novelty/contradiction detection
and adds correction-pattern classification on top.

All scoring is deterministic and rule-based (zero LLM calls).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_memory.core.brain import BrainConfig
    from neural_memory.storage.base import NeuralStorage

logger = logging.getLogger(__name__)

# ── Correction Detection ─────────────────────────────────────────────

# Patterns that indicate user corrections (subset of preference signals)
# These represent explicit "you're wrong, do X instead" statements
_CORRECTION_PATTERNS: list[re.Pattern[str]] = [
    # English corrections
    re.compile(r"(?:that's|it's|this is)\s+(?:wrong|incorrect|not right)", re.I),
    re.compile(r"(?:actually|no)[,:\s]+(?:it|that)?\s*should\s+(?:be|have)", re.I),
    re.compile(r"(?:change|update|fix|correct)\s+(?:it|that|this)?\s*(?:to|from)", re.I),
    re.compile(r"instead of\s+.+?[,:\s]+(?:use|do|try)", re.I),
    # Vietnamese corrections
    re.compile(r"(?:sai rồi|không đúng|chưa đúng)", re.I),
    re.compile(r"(?:phải là|nên là|đúng ra là)", re.I),
    re.compile(r"(?:sửa|đổi|chuyển)\s+(?:lại|thành)", re.I),
]


def is_correction(content: str) -> bool:
    """Check if content represents a user correction."""
    return any(p.search(content) for p in _CORRECTION_PATTERNS)


# ── Significance Result ──────────────────────────────────────────────


@dataclass(frozen=True)
class SignificanceResult:
    """Result of significance scoring for a detected memory."""

    surprise_bonus: float  # 0.0-3.0 from prediction_error
    is_correction: bool
    is_contradiction: bool  # surprise_bonus >= 2.5 indicates reversal
    is_novel: bool  # surprise_bonus >= 1.0 indicates novelty
    adjusted_priority: int  # final priority after all boosts (capped at 10)
    boost_applied: float  # total boost applied to base priority

    def to_metadata(self) -> dict[str, Any]:
        """Format for inclusion in saved memory metadata."""
        return {
            "surprise": round(self.surprise_bonus, 2),
            "correction": self.is_correction,
            "contradiction": self.is_contradiction,
            "novel": self.is_novel,
            "boost": round(self.boost_applied, 2),
        }


# ── Core Scoring ─────────────────────────────────────────────────────


async def score_significance(
    content: str,
    detected_type: str,
    base_priority: int,
    storage: NeuralStorage,
    brain_config: BrainConfig,
    *,
    correction_boost: float = 2.0,
    contradiction_boost: float = 2.5,
    novelty_boost: float = 1.5,
) -> SignificanceResult:
    """Score a detected memory for cognitive significance.

    Combines multiple signals to determine priority boost:
    1. Prediction error (novelty/contradiction via existing engine)
    2. Correction detection (user explicitly correcting agent)

    The highest applicable boost wins (not additive) to avoid
    over-inflation. Final priority is capped at 10.

    Args:
        content: The detected memory content.
        detected_type: Memory type from auto-capture ("decision", "error", etc.)
        base_priority: Original priority from pattern detection.
        storage: Storage backend for prediction error lookups.
        brain_config: Brain configuration for prediction error.
        correction_boost: Priority boost for user corrections.
        contradiction_boost: Priority boost for contradictions.
        novelty_boost: Priority boost for novel topics.

    Returns:
        SignificanceResult with adjusted priority and scoring breakdown.
    """
    from neural_memory.utils.simhash import simhash

    content_hash = simhash(content)
    tags = _extract_tags(content)

    # Get surprise bonus from prediction error engine
    from neural_memory.engine.prediction_error import compute_surprise_bonus

    try:
        surprise = await compute_surprise_bonus(
            content=content,
            tags=tags,
            content_hash=content_hash,
            storage=storage,
            config=brain_config,
        )
    except Exception:
        logger.debug("Surprise bonus computation failed, using default", exc_info=True)
        surprise = 1.0  # default moderate novelty on failure

    # Classify signals
    correction = is_correction(content)
    contradiction = surprise >= 2.5
    novel = surprise >= 1.0

    # Determine boost — highest signal wins (not additive)
    boost = 0.0
    if contradiction:
        boost = max(boost, contradiction_boost)
    if correction:
        boost = max(boost, correction_boost)
    if novel and not contradiction:
        boost = max(boost, novelty_boost)

    # Near-duplicate penalty: surprise < 0.5 means very similar content exists
    if surprise < 0.5:
        boost = min(boost, 0.0)
        # Apply penalty for near-duplicates
        if surprise == 0.0:
            boost = -2.0

    adjusted = min(max(round(base_priority + boost), 1), 10)

    logger.debug(
        "Significance: type=%s surprise=%.2f correction=%s boost=%.1f priority=%d->%d",
        detected_type,
        surprise,
        correction,
        boost,
        base_priority,
        adjusted,
    )

    return SignificanceResult(
        surprise_bonus=surprise,
        is_correction=correction,
        is_contradiction=contradiction,
        is_novel=novel,
        adjusted_priority=adjusted,
        boost_applied=boost,
    )


def _extract_tags(content: str) -> set[str]:
    """Extract simple tags from content for prediction error lookup.

    Uses capitalized words and tech terms as search tags.
    """
    # CamelCase words
    camel = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", content)
    # Capitalized words (3+ chars)
    caps = re.findall(r"\b[A-Z][a-z]{2,}\b", content)
    # Tech terms
    tech = re.findall(r"\b[A-Z][a-zA-Z]*(?:SQL|DB|JS|API|CSS|HTML)\b", content)

    tags = set(camel + caps + tech)
    # Lowercase for consistency
    return {t.lower() for t in tags if len(t) >= 3}
