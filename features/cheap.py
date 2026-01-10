'''
    Each feature captures one intuitive aspect of the dynamics:
    - rho_mean: average “fill level” of 1s (how full the system is on average)
    - rho_std: how much that fill level wobbles over time (stable vs oscillatory)
    - flip_rate: how many cells change between steps (overall activity / noisiness)
    - autocorr_t1: similarity between consecutive time steps (persistence in time)
    - space_autocorr: similarity between neighboring cells in space (stripe/block strength)
    - h3_global: overall spatial pattern complexity from 3-cell blocks
    - space_entropy_mean: average within-row disorder of 3-cell patterns
    - activity_tail: recent activity in the last ~100 steps (whether it has “settled down”)
    - is_fixedpoint: 1 if the system is effectively frozen at the end, else 0
'''

import numpy as np

def cheap_features(X):
    """
    Compute enhanced features from CA spacetime evolution.
    
    Parameters
    ----------
    X : np.ndarray, shape (T, W)
        Spacetime evolution, values in {0,1}
    
    Returns
    -------
    dict
        Dictionary of feature values including spatial metrics and convergence.
    """
    T = X.shape[0]

    # 1. Density statistics
    density_t = X.mean(axis=1)
    rho_mean = density_t.mean()
    rho_std = density_t.std()

    # 2. Activity (flip rate)
    flips = np.abs(X[1:] - X[:-1])
    flip_rate = flips.mean()

    # 3. Temporal autocorrelation (lag 1)
    x_t = density_t[:-1]
    y_t = density_t[1:]
    if np.std(x_t) < 1e-8 or np.std(y_t) < 1e-8:
        autocorr_t1 = 0.0
    else:
        autocorr_t1 = np.corrcoef(x_t, y_t)[0, 1]

    # 4. Spatial Autocorrelation (lag 1)
    # Compute correlation between X[t, i] and X[t, i+1] for each t, then average
    spatial_corrs = []
    for t in range(T):
        row = X[t]
        row_next = np.roll(row, -1)
        if np.std(row) < 1e-8:
            spatial_corrs.append(1.0 if np.mean(row) > 0.5 else 0.0)
        else:
            c = np.corrcoef(row, row_next)[0, 1]
            spatial_corrs.append(c)
    space_autocorr = np.mean(spatial_corrs)

    # 5. Spatial Block Entropy (H3)
    # Global H3 (as originally implemented)
    blocks = np.lib.stride_tricks.sliding_window_view(X, window_shape=3, axis=1)
    blocks_flat = blocks.reshape(-1, 3)
    block_indices = blocks_flat[:, 0] * 4 + blocks_flat[:, 1] * 2 + blocks_flat[:, 2]
    _, counts = np.unique(block_indices, return_counts=True)
    p = counts / counts.sum()
    h3_global = -np.sum(p * np.log(p + 1e-12))

    # Mean Spatial Entropy (per time step)
    # This measures the average 'disorder' within rows
    row_entropies = []
    for t in range(T):
        row_blocks = blocks[t].reshape(-1, 3)
        row_indices = row_blocks[:, 0] * 4 + row_blocks[:, 1] * 2 + row_blocks[:, 2]
        _, row_counts = np.unique(row_indices, return_counts=True)
        rp = row_counts / row_counts.sum()
        row_entropies.append(-np.sum(rp * np.log(rp + 1e-12)))
    space_entropy_mean = np.mean(row_entropies)

    # 6. Convergence / Fixed Point Flag
    # Check activity in the final 100 steps
    tail_window = min(100, T - 1)
    activity_tail = flips[-tail_window:].mean() if T > 1 else 0.0
    is_fixedpoint = 1.0 if activity_tail < 1e-5 else 0.0

    return {
        "rho_mean": float(rho_mean),
        "rho_std": float(rho_std),
        "flip_rate": float(flip_rate),
        "autocorr_t1": float(autocorr_t1),
        "space_autocorr": float(space_autocorr),
        "h3_global": float(h3_global),
        "space_entropy_mean": float(space_entropy_mean),
        "activity_tail": float(activity_tail),
        "is_fixedpoint": is_fixedpoint
    }