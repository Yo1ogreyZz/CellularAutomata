import numpy as np
from sklearn.feature_selection import mutual_info_classif

def normalized_mutual_info_classif(X, y, **kwargs):
    """
    Compute normalized mutual information between each feature in X and a discrete target y.
    Normalization is done by dividing by the entropy of y (Theil's U).
    
    Parameters:
        X : array-like, shape (n_samples, n_features)
            The continuous feature(s).
        y : array-like, shape (n_samples,)
            The discrete target variable.
        **kwargs:
            Additional keyword arguments to pass to mutual_info_classif (e.g., 'n_neighbors').
    
    Returns:
        normalized_mi : array of floats, shape (n_features,)
            The normalized mutual information for each feature.
    """
    # Compute mutual information using sklearn's estimator.
    mi = mutual_info_classif(X, y, **kwargs)
    
    # Compute the entropy of y.
    y = np.asarray(y)
    # Count occurrences of each class (assuming non-negative integer labels)
    counts = np.bincount(y)
    probs = counts[counts > 0] / len(y)
    entropy_y = -np.sum(probs * np.log(probs))  # natural log; results in nats
    
    # Avoid division by zero
    if entropy_y == 0:
        # If y has zero entropy (e.g., a constant), return zeros or handle as desired.
        return np.zeros_like(mi)
    
    # Normalize mutual information by H(y)
    normalized_mi = mi / entropy_y
    return np.clip(normalized_mi, 0.0, 1.0)