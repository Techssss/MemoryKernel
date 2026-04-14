"""
memk.core.scorer
================
Memory Scoring and Ranking Engine.

Scoring formula
---------------
    final_score = w1 * vector_similarity
                + w2 * keyword_score
                + w3 * importance
                + w4 * recency
                + w5 * confidence

All component scores are normalized to [0, 1] before weighting so that no
single dimension can dominate by accident.

Recency uses an exponential decay curve:
    recency(age_days) = exp(-decay_rate * age_days)

  half_life_days=30 → recency=0.50 at 30 days, 0.25 at 60 days, ~0 at 200 days
  half_life_days=7  → fast decay (chat logs / short-term memory)
  half_life_days=90 → slow decay (long-term project knowledge)

The engine is stateless and deterministic — the same inputs always produce
the same score. Weights are validated to be non-negative at construction time.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score Component Breakdown
# ---------------------------------------------------------------------------

@dataclass
class ScoreBreakdown:
    """
    Full transparency object for a single scored item.
    Allows the CLI, tests, and downstream code to inspect why something ranked where it did.
    """
    vector_similarity: float   # Cosine sim normalized to [0, 1]
    keyword_score: float       # 0.0 or 1.0 (binary exact match)
    importance: float          # [0, 1] normalized from stored value
    recency: float             # Exponential decay score in [0, 1]
    confidence: float          # [0, 1] from stored value

    # Computed final
    final_score: float = 0.0

    # Diagnostics
    age_days: float = 0.0      # How old the item is (for debugging)
    is_fact: bool = False       # True = fact, False = raw memory

    def as_dict(self) -> dict:
        return {
            "vector_similarity": round(self.vector_similarity, 4),
            "keyword_score": round(self.keyword_score, 4),
            "importance": round(self.importance, 4),
            "recency": round(self.recency, 4),
            "confidence": round(self.confidence, 4),
            "final_score": round(self.final_score, 4),
            "age_days": round(self.age_days, 1),
            "is_fact": self.is_fact,
        }

    def short_repr(self) -> str:
        return (
            f"score={self.final_score:.3f} "
            f"[vec={self.vector_similarity:.2f} kw={self.keyword_score:.1f} "
            f"imp={self.importance:.2f} rec={self.recency:.2f} conf={self.confidence:.2f}]"
        )


# ---------------------------------------------------------------------------
# Scoring Weights
# ---------------------------------------------------------------------------

@dataclass
class ScoringWeights:
    """
    Five-dimensional weight vector.
    Weights do NOT need to sum to 1 — they are relative priorities.
    Component scores are each in [0, 1] so the max possible final score
    equals the sum of all weights.

    Defaults tuned for an AI agent assistant context:
      - Vector similarity carries the most weight (semantic recall).
      - Keyword exact match is a strong booster but not required.
      - Importance and recency are secondary but meaningful.
      - Confidence is a light tie-breaker.
    """
    w1: float = 0.35   # vector similarity  (semantic relevance)
    w2: float = 0.20   # keyword score      (exact lexical signal)
    w3: float = 0.20   # importance         (domain priority)
    w4: float = 0.15   # recency            (freshness / forgetting curve)
    w5: float = 0.10   # confidence         (epistemic certainty)

    # Multiplier applied to facts over memories (denser, more reliable knowledge)
    fact_multiplier: float = 1.3

    def __post_init__(self):
        for name, val in [("w1", self.w1), ("w2", self.w2), ("w3", self.w3),
                          ("w4", self.w4), ("w5", self.w5),
                          ("fact_multiplier", self.fact_multiplier)]:
            if val < 0:
                raise ValueError(f"ScoringWeights.{name} must be >= 0, got {val}")

    @property
    def total(self) -> float:
        return self.w1 + self.w2 + self.w3 + self.w4 + self.w5

    def normalize(self) -> "ScoringWeights":
        """Return a copy with weights scaled so they sum to 1.0."""
        t = self.total
        if t == 0:
            raise ValueError("Cannot normalize: all weights are zero.")
        return ScoringWeights(
            w1=self.w1 / t,
            w2=self.w2 / t,
            w3=self.w3 / t,
            w4=self.w4 / t,
            w5=self.w5 / t,
            fact_multiplier=self.fact_multiplier,
        )


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class MemoryScorer:
    """
    Stateless scoring engine.

    Usage
    -----
    scorer = MemoryScorer()                           # default weights
    scorer = MemoryScorer(ScoringWeights(w3=0.5))    # boost importance

    breakdown = scorer.score(
        vector_similarity=0.82,
        keyword_score=1.0,
        importance=0.9,
        created_at="2025-01-01T00:00:00",
        confidence=0.95,
        is_fact=True,
    )
    print(breakdown.final_score)   # → weighted sum × fact_multiplier
    """

    def __init__(
        self,
        weights: Optional[ScoringWeights] = None,
        half_life_days: float = 30.0,
    ):
        """
        Parameters
        ----------
        weights : ScoringWeights | None
            Custom weight vector. Defaults to ScoringWeights().
        half_life_days : float
            Recency half-life. Items this many days old receive recency=0.5.
            Lower = faster forgetting, higher = longer memory retention.
        """
        self.weights = weights or ScoringWeights()
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive.")
        # Pre-compute decay rate from half-life: exp(-λ * t_half) = 0.5
        self._decay_rate: float = math.log(2) / half_life_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        *,
        vector_similarity: float,
        keyword_score: float,
        importance: float,
        created_at: str,
        confidence: float,
        is_fact: bool = False,
    ) -> ScoreBreakdown:
        """
        Compute the full scoring breakdown for a single candidate item.

        All float inputs are clamped to [0, 1] defensively.

        Parameters
        ----------
        vector_similarity : float
            Cosine similarity already normalized to [0, 1].
        keyword_score : float
            1.0 if the item matched the keyword filter, else 0.0.
        importance : float
            Stored importance value normalized to [0, 1].
        created_at : str
            ISO 8601 timestamp string (stored as UTC in the DB).
        confidence : float
            Epistemic confidence of the item in [0, 1].
        is_fact : bool
            If True, applies the fact_multiplier boost.
        """
        # Clamp all inputs defensively
        vec  = _clamp(vector_similarity)
        kw   = _clamp(keyword_score)
        imp  = _clamp(importance)
        conf = _clamp(confidence)

        age_days = _age_days(created_at)
        rec = self.recency_score(age_days)

        w = self.weights
        raw = (
            w.w1 * vec
            + w.w2 * kw
            + w.w3 * imp
            + w.w4 * rec
            + w.w5 * conf
        )

        multiplier = w.fact_multiplier if is_fact else 1.0
        final = raw * multiplier

        return ScoreBreakdown(
            vector_similarity=vec,
            keyword_score=kw,
            importance=imp,
            recency=rec,
            confidence=conf,
            final_score=final,
            age_days=age_days,
            is_fact=is_fact,
        )

    def recency_score(self, age_days: float) -> float:
        """
        Exponential decay: recency = exp(-λ * age_days).

        Returns 1.0 for brand-new items, 0.5 at half_life_days,
        approaching 0 for very old items. Always in [0, 1].
        """
        return math.exp(-self._decay_rate * max(0.0, age_days))

    def score_metadata_only(
        self,
        *,
        importance: float,
        created_at: str,
        confidence: float,
        is_fact: bool = False,
    ) -> ScoreBreakdown:
        """
        Score using only metadata (no query context).
        Useful for 'cold' rankings in `memk doctor` and idle analytics,
        where there is no query vector or keyword to compare against.
        """
        return self.score(
            vector_similarity=0.0,
            keyword_score=0.0,
            importance=importance,
            created_at=created_at,
            confidence=confidence,
            is_fact=is_fact,
        )


# ---------------------------------------------------------------------------
# Utility helpers (module-level, importable)
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, float(value)))


def _age_days(created_at: str) -> float:
    """
    Parse a stored ISO 8601 timestamp and return how many days ago it was.
    Handles both 'T' and space separators. Always returns a non-negative float.
    Uses UTC consistently to avoid timezone drift.
    """
    try:
        # Normalize separator
        ts = created_at.replace(" ", "T")
        # Strip microseconds if present beyond 6 digits
        if "." in ts:
            base, frac = ts.split(".", 1)
            ts = base + "." + frac[:6]
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(tz=timezone.utc)
        delta = now - dt
        return max(0.0, delta.total_seconds() / 86400.0)
    except (ValueError, TypeError) as exc:
        logger.warning(f"Could not parse timestamp '{created_at}': {exc}. Assuming age=0.")
        return 0.0
