"""
memk.core.graph_propagation
===========================
Lightweight Personalized PageRank / PPNP (Predictive Power of Network Propagation)
for the in-memory GraphIndex sidecar.

Implements efficient scatter-based sparse vector propagation 
to avoid dense N x N matrix multiplications. Hard pruning is 
enforced after every step to cap computational cost.
"""

import numpy as np


def propagate_ppnp(
    seed_scores: dict[int, float],
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    num_entities: int,
    alpha: float = 0.5,
    steps: int = 3,
    max_active_entities: int = 100,
) -> dict[int, float]:
    """
    Perform Personalized PageRank propagation constrained by hard pruning.
    
    Formula: h_{t+1} = \alpha h_0 + (1 - \alpha) A^T_{hat} h_t
    
    Algorithm is implemented using scatter strategy from sparse active nodes.
    
    Parameters
    ----------
    seed_scores : dict[int, float]
        Initial active nodes and their starting relevance scores (h0).
    indptr : np.ndarray
        CSR row offsets for entity edges.
    indices : np.ndarray
        CSR column indices (target entities).
    weights : np.ndarray
        CSR data (edge weights).
    num_entities : int
        Total number of entities in the adjacency arrays.
    alpha : float, optional
        Teleport probability. Controls how much score anchors back to h0.
        (0.0 = pure random walk, 1.0 = score stays entirely at h0).
    steps : int, optional
        Number of propagation iterations.
    max_active_entities : int, optional
        Hard pruning threshold. Max top K active nodes retained per step.
        
    Returns
    -------
    dict[int, float]
        Entity IDs and their converged propagation scores.
    """
    if not seed_scores or num_entities == 0:
        return {}

    # h0 vector anchoring the random walk
    h0 = np.zeros(num_entities, dtype=np.float32)
    for k, v in seed_scores.items():
        if 0 <= k < num_entities:
            h0[k] = v
            
    h_current = h0.copy()
    active_nodes = np.array([k for k in seed_scores.keys() if 0 <= k < num_entities], dtype=np.int32)
    
    has_weights = (weights is not None and len(weights) > 0)
    
    for step in range(steps):
        # 1. Next state vector
        h_next = np.zeros(num_entities, dtype=np.float32)
        
        # 2. Scatter values from active nodes avoiding dense graph matvec
        for u in active_nodes:
            score = h_current[u]
            if score <= 0.0:
                continue
                
            start, end = indptr[u], indptr[u+1]
            if start == end:
                continue  # No outbound edges
                
            out_indices = indices[start:end]
            
            # Row-normalize out-edges dynamically
            if has_weights:
                out_weights = weights[start:end]
                w_sum = out_weights.sum()
                if w_sum > 0:
                    norm_weights = out_weights / w_sum
                else: 
                    norm_weights = np.ones(end - start, dtype=np.float32) / (end - start)
            else:
                norm_weights = np.ones(end - start, dtype=np.float32) / (end - start)
                
            # Scatter using numpy bulk index assignment
            # equivalent to A^T_{hat} h_t mapping
            # using np.add.at is slightly safer if duplicate targets exist, 
            # though standard graph implies unique out-edges.
            np.add.at(h_next, out_indices, score * norm_weights)
            
        # 3. Apply formula: h_{t+1} = alpha * h_0 + (1 - alpha) * h_next
        h_current = (1.0 - alpha) * h_next
        for u in seed_scores.keys():
            if 0 <= u < num_entities:
                h_current[u] += alpha * h0[u]
        
        # 4. Hard pruning to bounds
        alive = np.nonzero(h_current > 0)[0]
        if len(alive) > max_active_entities:
            alive_scores = h_current[alive]
            # O(M) top-K partitioning where M = len(alive)
            top_k_idx = np.argpartition(alive_scores, -max_active_entities)[-max_active_entities:]
            top_k_nodes = alive[top_k_idx]
            
            # Reconstruct array explicitly discarding tails
            h_pruned = np.zeros(num_entities, dtype=np.float32)
            h_pruned[top_k_nodes] = h_current[top_k_nodes]
            h_current = h_pruned
            active_nodes = top_k_nodes
        else:
            active_nodes = alive
            
    # Filter back to dictionary return
    final_nodes = np.nonzero(h_current > 0)[0]
    
    # Sort descending by value for convenience
    sorted_node_idx = np.argsort(-h_current[final_nodes])
    sorted_nodes = final_nodes[sorted_node_idx]
    
    return {int(node): float(h_current[node]) for node in sorted_nodes}
