"""
tests.test_graph_propagation
============================
Verify the Personalized PageRank / PPNP implementation.
"""
import numpy as np
from memk.core.graph_propagation import propagate_ppnp

def test_propagation_chain():
    """
    Test a linear chain: 
    A (0) --CEO_OF--> B (1) --LOCATED_IN--> C (2)
    """
    num_entities = 3
    
    # CSR mapping
    indptr = np.array([0, 1, 2, 2], dtype=np.int32)
    indices = np.array([1, 2], dtype=np.int32)
    weights = np.array([1.0, 1.0], dtype=np.float32)
    
    # Seed A with score 1.0
    seed = {0: 1.0}
    
    # Run PPNP: 2 steps, alpha=0.5
    res = propagate_ppnp(
        seed_scores=seed,
        indptr=indptr, 
        indices=indices, 
        weights=weights,
        num_entities=num_entities, 
        alpha=0.5, 
        steps=2, 
        max_active_entities=10
    )
    
    # Mathematical expectation:
    # step 0: [1.0, 0.0, 0.0]
    # step 1: (0.5 * scatter) + 0.5 * h0
    #         = 0.5 * [0, 1.0, 0] + [0.5, 0, 0] = [0.5, 0.5, 0]
    # step 2: 
    #         scatter from [0.5, 0.5, 0] -> [0, 0.5, 0.5]
    #         = 0.5 * [0, 0.5, 0.5] + [0.5, 0, 0]
    #         = [0.5, 0.25, 0.25]
    
    assert len(res) == 3
    assert np.isclose(res[0], 0.5)
    assert np.isclose(res[1], 0.25)
    assert np.isclose(res[2], 0.25)

def test_hard_pruning():
    """Test that max_active_entities prunes low scored entities correctly."""
    num_entities = 5
    
    # 0 -> 1, 2, 3, 4 (out degree 4)
    indptr = np.array([0, 4, 4, 4, 4, 4], dtype=np.int32)
    indices = np.array([1, 2, 3, 4], dtype=np.int32)
    weights = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
    
    # Note: our code auto-normalizes array locally, meaning weights sum=1 
    #       which holds true here anyway (0.4+0.3+0.2+0.1 = 1.0).
    
    seed = {0: 1.0}
    
    # 1 step, max_active = 2
    res = propagate_ppnp(
        seed_scores=seed, 
        indptr=indptr, 
        indices=indices, 
        weights=weights, 
        num_entities=num_entities, 
        alpha=0.5, 
        steps=1, 
        max_active_entities=2
    )
    
    # Math:
    # step 1: scatter from 1.0
    #         h_next = [0, 0.4, 0.3, 0.2, 0.1]
    #         after alpha: 0.5 * h_next + [0.5, 0...]
    #         = [0.5, 0.2, 0.15, 0.1, 0.05]
    # pruning = top 2 nodes -> 0 and 1
    
    assert len(res) == 2
    assert 0 in res
    assert 1 in res
    assert 2 not in res 
    assert 3 not in res
    assert 4 not in res
    
    assert np.isclose(res[0], 0.5)
    assert np.isclose(res[1], 0.2)

def test_missing_weights():
    """Test unweighted graphs automatically normalize evenly."""
    num_entities = 3
    # 0 -> 1, 2
    indptr = np.array([0, 2, 2, 2], dtype=np.int32)
    indices = np.array([1, 2], dtype=np.int32)
    
    seed = {0: 1.0}
    
    res = propagate_ppnp(
        seed_scores=seed, 
        indptr=indptr, 
        indices=indices, 
        weights=None, 
        num_entities=num_entities, 
        alpha=0.0, # Pure random walk
        steps=1,
        max_active_entities=10
    )
    
    # 1 step uniform random walk should split 1.0 to 1 and 2 = 0.5 each
    assert np.isclose(res[1], 0.5)
    assert np.isclose(res[2], 0.5)
    assert 0 not in res 

if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v"]))
