"""Tests for SimHash pre-filter — exclude distant neurons before activation."""

from __future__ import annotations

from neural_memory.engine.simhash_filter import compute_exclude_set
from neural_memory.utils.simhash import simhash


class TestComputeExcludeSet:
    """Tests for compute_exclude_set()."""

    def test_disabled_when_threshold_zero(self) -> None:
        """Threshold=0 should return empty set (disabled)."""
        hashes = [("n1", simhash("hello world")), ("n2", simhash("goodbye"))]
        result = compute_exclude_set("hello world", hashes, threshold=0)
        assert result == set()

    def test_empty_neuron_list(self) -> None:
        """Empty neuron list should return empty set."""
        result = compute_exclude_set("hello world", [], threshold=10)
        assert result == set()

    def test_similar_texts_not_excluded(self) -> None:
        """Similar texts should NOT be excluded."""
        text = "the quick brown fox jumps over the lazy dog"
        similar = "the quick brown fox jumped over the lazy dog"
        hashes = [("n1", simhash(similar))]
        result = compute_exclude_set(text, hashes, threshold=10)
        assert "n1" not in result

    def test_dissimilar_texts_excluded(self) -> None:
        """Very different texts should be excluded."""
        hashes = [
            ("n1", simhash("quantum physics dark matter neutron star")),
            ("n2", simhash("cooking recipe pasta tomato basil")),
        ]
        result = compute_exclude_set(
            "machine learning neural networks deep learning",
            hashes,
            threshold=5,  # Very tight threshold
        )
        # At least one should be excluded with such a tight threshold
        assert len(result) > 0

    def test_hash_zero_never_excluded(self) -> None:
        """Neurons with content_hash=0 should NEVER be excluded."""
        hashes = [
            ("n1", 0),  # Legacy neuron — no hash
            ("n2", simhash("completely unrelated xyz abc 123")),
        ]
        result = compute_exclude_set("hello world", hashes, threshold=5)
        assert "n1" not in result  # hash=0 → always kept

    def test_exact_match_not_excluded(self) -> None:
        """Exact same text should never be excluded (distance=0)."""
        text = "neural memory spreading activation"
        hashes = [("n1", simhash(text))]
        result = compute_exclude_set(text, hashes, threshold=1)
        assert "n1" not in result

    def test_empty_query_returns_empty(self) -> None:
        """Empty query produces hash=0, should return empty set."""
        hashes = [("n1", simhash("hello"))]
        result = compute_exclude_set("", hashes, threshold=10)
        assert result == set()

    def test_whitespace_query_returns_empty(self) -> None:
        """Whitespace-only query produces hash=0, should return empty set."""
        hashes = [("n1", simhash("hello"))]
        result = compute_exclude_set("   ", hashes, threshold=10)
        assert result == set()

    def test_negative_threshold_returns_empty(self) -> None:
        """Negative threshold should behave like disabled."""
        hashes = [("n1", simhash("hello"))]
        result = compute_exclude_set("world", hashes, threshold=-1)
        assert result == set()

    def test_mixed_hashes_partial_exclusion(self) -> None:
        """Mix of similar and dissimilar should only exclude dissimilar."""
        query = "python programming language"
        hashes = [
            ("n1", simhash("python programming tutorial")),  # similar (distance ~13)
            (
                "n2",
                simhash("ocean biology marine life coral reef ecosystem"),
            ),  # very different (~31)
            ("n3", 0),  # no hash
        ]
        result = compute_exclude_set(query, hashes, threshold=15)
        assert "n1" not in result  # similar enough — kept
        assert "n3" not in result  # hash=0 — always kept
        assert "n2" in result  # very different — excluded

    def test_large_threshold_excludes_nothing(self) -> None:
        """Threshold=64 (full 64-bit hash) should exclude nothing."""
        hashes = [
            ("n1", simhash("abc")),
            ("n2", simhash("xyz")),
            ("n3", simhash("123")),
        ]
        result = compute_exclude_set("hello", hashes, threshold=64)
        assert len(result) == 0


class TestSimhashPrefilterConfig:
    """Tests for BrainConfig simhash_prefilter_threshold field."""

    def test_default_disabled(self) -> None:
        """Default threshold should be 0 (disabled)."""
        from neural_memory.core.brain import BrainConfig

        config = BrainConfig()
        assert config.simhash_prefilter_threshold == 0

    def test_custom_threshold(self) -> None:
        """Should accept custom threshold."""
        from neural_memory.core.brain import BrainConfig

        config = BrainConfig(simhash_prefilter_threshold=15)
        assert config.simhash_prefilter_threshold == 15
