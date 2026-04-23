"""SimHash pre-filter for recall — exclude distant neurons before spreading activation.

Computes Hamming distance between query SimHash and stored neuron hashes.
Neurons whose content is too dissimilar (distance > threshold) are excluded
from the candidate pool, reducing the graph traversal search space.
"""

from __future__ import annotations

import logging

from neural_memory.utils.simhash import hamming_distance, simhash

logger = logging.getLogger(__name__)


def compute_exclude_set(
    query: str,
    neuron_hashes: list[tuple[str, int]],
    threshold: int,
) -> set[str]:
    """Compute set of neuron IDs to exclude based on SimHash distance.

    Args:
        query: The recall query text.
        neuron_hashes: List of (neuron_id, content_hash) pairs from storage.
        threshold: Maximum Hamming distance to keep (1-64).
            Neurons with distance > threshold are excluded.

    Returns:
        Set of neuron IDs to exclude from anchor search.
    """
    if threshold <= 0 or not neuron_hashes:
        return set()

    query_hash = simhash(query)
    if query_hash == 0:
        return set()

    exclude: set[str] = set()
    for neuron_id, content_hash in neuron_hashes:
        # content_hash=0 means no hash computed — never exclude
        if content_hash == 0:
            continue
        if hamming_distance(query_hash, content_hash) > threshold:
            exclude.add(neuron_id)

    logger.debug(
        "SimHash pre-filter: %d/%d neurons excluded (threshold=%d)",
        len(exclude),
        len(neuron_hashes),
        threshold,
    )
    return exclude
