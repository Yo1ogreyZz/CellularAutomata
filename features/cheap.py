import numpy as np


def cheap_features(X):
    """
    Compute features from CA spacetime evolution.
    
    Parameters
    ----------
    X : np.ndarray, shape (T, W)
        Spacetime evolution, values in {0,1}
    
    Returns
    -------
    dict
        Dictionary of feature values
    """
    T = X.shape[0]

    # Density statistics
    density_t = X.mean(axis=1)
    rho_mean = density_t.mean()
    rho_std = density_t.std()

    # Activity (flip rate)
    flips = np.abs(X[1:] - X[:-1])
    flip_rate = flips.mean()

    # Temporal autocorrelation (lag 1)
    # Compute mean density at each time step
    X0 = X[:-1].reshape(T-1, -1)
    X1 = X[1:].reshape(T-1, -1)
    x = X0.mean(axis=1)
    y = X1.mean(axis=1)
    
    # Handle case where variance is near zero (constant density)
    # Set autocorrelation to 0.0 instead of NaN for compatibility with downstream analysis
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        autocorr_t1 = 0.0
    else:
        autocorr_t1 = np.corrcoef(x, y)[0, 1]

    # Spatial block entropy (size 3) - vectorized version
    # Use sliding window view to extract all 3-bit blocks efficiently
    # For each row, extract overlapping 3-bit blocks
    # Result shape: (T, W-2, 3) - all overlapping 3-bit blocks across all time steps
    blocks = np.lib.stride_tricks.sliding_window_view(X, window_shape=3, axis=1)
    
    # Flatten to (T*(W-2), 3) for unique counting
    blocks_flat = blocks.reshape(-1, 3)
    
    # Convert blocks to integer indices for faster unique counting
    # Each 3-bit block can be represented as an integer 0-7
    block_indices = blocks_flat[:, 0] * 4 + blocks_flat[:, 1] * 2 + blocks_flat[:, 2]
    
    # Count unique blocks
    _, counts = np.unique(block_indices, return_counts=True)
    p = counts / counts.sum()
    H3 = -np.sum(p * np.log(p + 1e-12))

    return {
        "rho_mean": rho_mean,
        "rho_std": rho_std,
        "flip_rate": flip_rate,
        "autocorr_t1": autocorr_t1,
        "H3": H3,
    }
