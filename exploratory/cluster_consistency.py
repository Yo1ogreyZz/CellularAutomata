import os
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

HIGHDIM_LABELS = "data/embeddings/hdbscan_results_highdim.csv"
UMAP_LABELS = "data/embeddings/hdbscan_results_umap.csv"
REPORT_OUT = "data/specs/clustering_consistency_report.txt"

def main():
    if not os.path.exists(HIGHDIM_LABELS) or not os.path.exists(UMAP_LABELS):
        raise FileNotFoundError("Clustering label files missing. Run clustering.py for both modes first.")

    df_hd = pd.read_csv(HIGHDIM_LABELS).rename(columns={"cluster": "cluster_hd", "membership_prob": "prob_hd"})
    df_umap = pd.read_csv(UMAP_LABELS).rename(columns={"cluster": "cluster_umap", "membership_prob": "prob_umap"})
    combined = pd.merge(df_hd, df_umap, on="rule_id")

    ari = adjusted_rand_score(combined["cluster_hd"], combined["cluster_umap"])
    nmi = normalized_mutual_info_score(combined["cluster_hd"], combined["cluster_umap"])

    both_noise = len(combined[(combined["cluster_hd"] == -1) & (combined["cluster_umap"] == -1)])
    hd_only_noise = len(combined[(combined["cluster_hd"] == -1) & (combined["cluster_umap"] != -1)])
    umap_only_noise = len(combined[(combined["cluster_hd"] != -1) & (combined["cluster_umap"] == -1)])

    os.makedirs(os.path.dirname(REPORT_OUT), exist_ok=True)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
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