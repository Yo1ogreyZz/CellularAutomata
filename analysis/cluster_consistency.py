'''
This evaluates the 'Consistency' between two classification schemes:
1. Classification based on the original 9-dimensional physical feature space.
2. Classification based on the 2D UMAP manifold projection.

In physical terms, we are measuring whether the 'Phase Clusters' identified 
in the full parameter space remain stable when the system's dimensionality 
is reduced. Low consistency (ARI/NMI) suggests that the UMAP projection 
is capturing non-linear structural relationships that are 'hidden' or 
treated as noise in the raw high-dimensional Euclidean space.

'''

import os
import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Paths to clustering results
HIGHDIM_LABELS = "data/embeddings/hdbscan_results_highdim.csv"
UMAP_LABELS = "data/embeddings/hdbscan_results_umap.csv"
REPORT_OUT = "data/specs/clustering_consistency_report.txt"

def main():
    if not os.path.exists(HIGHDIM_LABELS) or not os.path.exists(UMAP_LABELS):
        raise FileNotFoundError("Clustering label files missing. Run clustering.py for both modes first.")

    # Load and merge labels on rule_id
    df_hd = pd.read_csv(HIGHDIM_LABELS).rename(columns={"cluster": "cluster_hd", "membership_prob": "prob_hd"})
    df_umap = pd.read_csv(UMAP_LABELS).rename(columns={"cluster": "cluster_umap", "membership_prob": "prob_umap"})
    
    combined = pd.merge(df_hd, df_umap, on="rule_id")

    # Metrics calculation (ARI and NMI are robust to label permutations)
    ari = adjusted_rand_score(combined["cluster_hd"], combined["cluster_umap"])
    nmi = normalized_mutual_info_score(combined["cluster_hd"], combined["cluster_umap"])

    # Quantify noise (-1) stability
    both_noise = len(combined[(combined["cluster_hd"] == -1) & (combined["cluster_umap"] == -1)])
    hd_only_noise = len(combined[(combined["cluster_hd"] == -1) & (combined["cluster_umap"] != -1)])
    umap_only_noise = len(combined[(combined["cluster_hd"] != -1) & (combined["cluster_umap"] == -1)])

    # Output report
    os.makedirs(os.path.dirname(REPORT_OUT), exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        f.write("=== Clustering Consistency Analysis ===\n")
        f.write(f"Adjusted Rand Index (ARI): {ari:.4f}\n")
        f.write(f"Normalized Mutual Info (NMI): {nmi:.4f}\n\n")
        f.write("=== Noise/Outlier Stability ===\n")
        f.write(f"Shared Noise Points: {both_noise}\n")
        f.write(f"Noise only in High-Dim: {hd_only_noise}\n")
        f.write(f"Noise only in UMAP: {umap_only_noise}\n")
    
    print(f"Consistency analysis complete. ARI: {ari:.4f}, NMI: {nmi:.4f}")
    print(f"Full report saved to {REPORT_OUT}")

if __name__ == "__main__":
    main()