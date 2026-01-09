import numpy as np
from ca.rules import rule_id_to_table


def evolve_ca_r2(rule_id, init_state, steps):
    """
    Evolve a 1D radius-2 binary CA using vectorized operations.
    
    This function uses NumPy vectorization to compute all cell updates
    simultaneously, avoiding Python loops for significant speedup.

    Parameters
    ----------
    rule_id : int
        32-bit integer defining the rule.
    init_state : np.ndarray, shape (W,)
        Initial state, values in {0,1}.
    steps : int
        Number of time steps.

    Returns
    -------
    X : np.ndarray, shape (steps, W)
        Spacetime evolution.
    """
    init_state = init_state.astype(np.uint8)
    W = init_state.shape[0]

    # Get lookup table for this rule
    table = rule_id_to_table(rule_id)

    # Initialize spacetime array
    X = np.zeros((steps, W), dtype=np.uint8)
    X[0] = init_state

    # Vectorized evolution: compute all cells in parallel for each time step
    for t in range(1, steps):
        prev = X[t - 1]
        
        # Extract neighborhoods using circular indexing
        # For radius-2, we need 5 neighbors: [i-2, i-1, i, i+1, i+2]
        # Use np.roll for circular shifts
        neigh_minus2 = np.roll(prev, 2)   # i-2
        neigh_minus1 = np.roll(prev, 1)   # i-1
        neigh_center = prev                # i
        neigh_plus1 = np.roll(prev, -1)   # i+1
        neigh_plus2 = np.roll(prev, -2)   # i+2
        
        # Convert 5-bit neighborhoods to indices (0-31)
        # Each neighbor contributes one bit: bit0=neigh_minus2, bit1=neigh_minus1, etc.
        indices = (
            neigh_minus2.astype(np.uint32) * 16 +
            neigh_minus1.astype(np.uint32) * 8 +
            neigh_center.astype(np.uint32) * 4 +
            neigh_plus1.astype(np.uint32) * 2 +
            neigh_plus2.astype(np.uint32)
        )
        
        # Lookup new states using the rule table
        X[t] = table[indices]

    return X
