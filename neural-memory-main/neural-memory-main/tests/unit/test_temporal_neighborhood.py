"""Tests for temporal neighborhood query in causal_traversal.py."""

from __future__ import annotations

from neural_memory.engine.causal_traversal import (
    CausalChain,
    CausalStep,
    EventSequence,
    EventStep,
    query_temporal_neighborhood,
)


class TestTemporalNeighborhoodImport:
    """Verify the function exists and is importable."""

    def test_importable(self) -> None:
        assert callable(query_temporal_neighborhood)

    def test_event_step_creation(self) -> None:
        """EventStep dataclass should work correctly."""
        step = EventStep(
            neuron_id="n1",
            content="test event",
            fiber_id="f1",
            timestamp=None,
            position=0,
        )
        assert step.neuron_id == "n1"
        assert step.position == 0

    def test_event_sequence_creation(self) -> None:
        """EventSequence should be constructible."""
        seq = EventSequence(
            seed_neuron_id="n1",
            direction="forward",
            events=(),
        )
        assert seq.seed_neuron_id == "n1"
        assert len(seq.events) == 0


class TestCausalTraversalDataStructures:
    """Verify existing data structures still work after modifications."""

    def test_causal_chain_empty(self) -> None:
        chain = CausalChain(
            seed_neuron_id="n1",
            direction="causes",
            steps=(),
            total_weight=0.0,
        )
        assert len(chain.steps) == 0
        assert chain.total_weight == 0.0

    def test_causal_step_fields(self) -> None:
        from neural_memory.core.synapse import SynapseType

        step = CausalStep(
            neuron_id="n2",
            content="something happened",
            synapse_type=SynapseType.CAUSED_BY,
            weight=0.75,
            depth=1,
        )
        assert step.depth == 1
        assert step.weight == 0.75
