import math
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ForgettingEngine:
    """
    Implements usage-based and time-based memory decay.
    Prevents knowledge explosion by identifying and removing 'cold' memories.
    """

    def __init__(self, base_lambda: float = 0.05, cold_threshold: float = 0.2, warm_threshold: float = 0.5):
        """
        Parameters
        ----------
        base_lambda    : Speed of decay. Higher = faster forgetting.
        cold_threshold : Scores below this are 'cold' (prunable).
        warm_threshold : Scores between cold and warm are 'warm'.
        """
        self.base_lambda = base_lambda
        self.cold_threshold = cold_threshold
        self.warm_threshold = warm_threshold

    def calculate_decay_score(self, importance: float, access_count: int, age_days: float) -> float:
        """
        Calculates the new decay score based on the formula:
        score = e^(-lambda * age)
        where lambda is scaled by importance and usage (access_count).
        
        Usage/Importance slows down the decay rate.
        """
        # Usage-adjusted decay rate
        # More access and higher importance reduce the decay rate lambda
        scale = (1.0 + math.log1p(access_count)) * (1.0 + importance)
        adjusted_lambda = self.base_lambda / scale
        
        # Exponential decay
        score = math.exp(-adjusted_lambda * max(0, age_days))
        
        # Multiply by initial importance weight to maintain priority
        return score * importance

    def get_state(self, score: float) -> str:
        """Categorize a memory based on its decay score."""
        if score >= self.warm_threshold:
            return "hot"
        elif score >= self.cold_threshold:
            return "warm"
        else:
            return "cold"
