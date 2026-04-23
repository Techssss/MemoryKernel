"""Tests for stratum-aware MMR diversity config."""

from __future__ import annotations

from neural_memory.core.brain import BrainConfig


class TestStratumDiversityConfig:
    """Verify BrainConfig fields for stratum-aware MMR."""

    def test_defaults(self) -> None:
        config = BrainConfig()
        assert config.cascade_staleness_enabled is True
        assert config.stratum_diversity_cap == 0.4

    def test_override(self) -> None:
        config = BrainConfig(
            stratum_diversity_cap=0.5,
            cascade_staleness_enabled=False,
        )
        assert config.stratum_diversity_cap == 0.5
        assert config.cascade_staleness_enabled is False

    def test_with_updates(self) -> None:
        config = BrainConfig()
        updated = config.with_updates(stratum_diversity_cap=0.3)
        assert updated.stratum_diversity_cap == 0.3
        # Original unchanged
        assert config.stratum_diversity_cap == 0.4

    def test_cap_range(self) -> None:
        """Cap should be between 0 and 1 for meaningful behavior."""
        config = BrainConfig()
        assert 0 < config.stratum_diversity_cap <= 1.0

    def test_max_per_stratum_computation(self) -> None:
        """At 40% cap with 10 target results, max 4 per stratum."""
        config = BrainConfig()
        target = 10
        max_per = max(1, int(target * config.stratum_diversity_cap))
        assert max_per == 4
