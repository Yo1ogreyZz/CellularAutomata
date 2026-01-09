"""
Feature Descriptions for Cellular Automata Analysis
===================================================

This script uses the following features extracted from CA evolution:

rho_mean
    Average fraction of cells in state 1 (overall density of "active" cells).
    - Closer to 0: mostly 0s
    - Closer to 1: mostly 1s

rho_std
    Standard deviation of density over time (temporal fluctuation).
    - Small value: density remains roughly constant
    - Large value: density changes significantly over time

flip_rate
    Average fraction of cells that change state between two consecutive time steps.
    - Higher values: pattern is very "busy", many cells switching states
    - Lower values: pattern is more stable

autocorr_t1
    Lag-1 temporal autocorrelation (similarity between consecutive time steps).
    - High value: pattern at t+1 is very similar to t (persistent, slowly evolving)
    - Low/negative value: pattern changes strongly between steps

H3
    Spatial block entropy based on 3-cell blocks (pattern complexity measure).
    - Higher values: more complex and less regular patterns
    - Lower values: more repetitive and structured patterns

Summary
-------
Together, these features capture:
- Average system state (rho_mean)
- Temporal stability/oscillation (rho_std + autocorr_t1)
- Local activity (flip_rate)
- Structural complexity (H3)
"""

import pandas as pd
import umap
import matplotlib.pyplot as plt


df = pd.read_csv("data/features/cheap_features.csv")


feature_cols = [
    "rho_mean",
    "rho_std",
    "flip_rate",
    "autocorr_t1",
    "H3",
]


X = df[feature_cols].values

reducer = umap.UMAP(
    n_neighbors=20,
    min_dist=0.1,
    n_jobs=-1
)
Z = reducer.fit_transform(X)

plt.figure(figsize=(6, 5))
plt.scatter(Z[:, 0], Z[:, 1], s=5, alpha=0.7)
plt.title("UMAP of r=2 CA (cheap features)")
plt.tight_layout()
plt.savefig("data/figures/umap_cheap.png", dpi=200)
plt.show()
