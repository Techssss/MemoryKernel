"""Spaced repetition review schedule — Leitner box system."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from neural_memory.utils.timeutils import utcnow

# Leitner box intervals (in days): box 1 = 1d, box 2 = 3d, box 3 = 7d, box 4 = 14d, box 5 = 30d
LEITNER_INTERVALS: dict[int, int] = {
    1: 1,
    2: 3,
    3: 7,
    4: 14,
    5: 30,
}

MAX_BOX = 5
MIN_BOX = 1


@dataclass(frozen=True)
class ReviewSchedule:
    """A spaced repetition schedule for a fiber.

    Uses SM-2 algorithm with Leitner boxes:
    - ease_factor adjusts per-item difficulty (SM-2 formula)
    - After box 2, intervals scale by ease_factor: I(n) = I(n-1) * EF
    - On failure: graduated drop (1-2 boxes) instead of hard reset to box 1

    Attributes:
        fiber_id: The fiber being reviewed
        brain_id: Brain owning the fiber
        box: Current Leitner box (1-5)
        next_review: When this fiber is next due for review
        last_reviewed: When last reviewed (None if never)
        review_count: Total number of reviews
        streak: Consecutive successful reviews
        ease_factor: SM-2 ease factor (1.3 to 3.0, default 2.5)
        created_at: When this schedule was created
    """

    fiber_id: str
    brain_id: str
    box: int = MIN_BOX
    next_review: datetime | None = None
    last_reviewed: datetime | None = None
    review_count: int = 0
    streak: int = 0
    ease_factor: float = 2.5
    created_at: datetime | None = None

    @classmethod
    def create(cls, fiber_id: str, brain_id: str) -> ReviewSchedule:
        """Create a new schedule starting at box 1, due immediately."""
        now = utcnow()
        return cls(
            fiber_id=fiber_id,
            brain_id=brain_id,
            box=MIN_BOX,
            next_review=now,
            last_reviewed=None,
            review_count=0,
            streak=0,
            ease_factor=2.5,
            created_at=now,
        )

    def advance(self, success: bool, quality: int = 4) -> ReviewSchedule:
        """Return a new schedule after a review.

        Args:
            success: True if recall was successful, False otherwise
            quality: Review quality 0-5 (SM-2 scale). Default 4 for success,
                     auto-set to 1 for failure if not explicitly provided.

        Returns:
            New ReviewSchedule with updated box, streak, ease_factor, next_review
        """
        now = utcnow()

        # Clamp quality to valid range
        q = max(0, min(quality if success else min(quality, 2), 5))

        # SM-2 ease factor update: EF' = EF + (0.1 - (5-q) * (0.08 + (5-q) * 0.02))
        ef_delta = 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)
        new_ef = max(1.3, min(self.ease_factor + ef_delta, 3.0))

        if success:
            new_box = min(self.box + 1, MAX_BOX)
            new_streak = self.streak + 1
        else:
            # Graduated failure: drop 1-2 boxes instead of hard reset
            drop = 1 if self.box <= 3 else 2
            new_box = max(MIN_BOX, self.box - drop)
            new_streak = 0

        # Interval: Leitner base for boxes 1-2, then scale by ease_factor
        base_interval = LEITNER_INTERVALS[new_box]
        if new_box > 2 and success:
            interval_days = base_interval * new_ef
        else:
            interval_days = float(base_interval)

        new_next_review = now + timedelta(days=interval_days)

        return replace(
            self,
            box=new_box,
            next_review=new_next_review,
            last_reviewed=now,
            review_count=self.review_count + 1,
            streak=new_streak,
            ease_factor=round(new_ef, 4),
        )
