'''
This acts as an automated 'Phase Classifier' for Cellular Automata.
It groups rules into clusters (Phases) based on their dynamical observables 
(Density, Entropy, Activity).

Two operation modes are supported:
- HIGH-DIM: Clusters rules using all 9 physical parameters simultaneously.
- UMAP: Clusters rules based on their coordinates in the learned 2D manifold.

The script also identifies 'Centroids' (typical rules of a phase) and 
'Boundary Rules' (rules at the transition between phases) for further 
deep analysis (FFT/GNN).
'''

import os
import argparse
import pandas as pd
import hdbscan
from sklearn.preprocessing import StandardScaler

# Configuration
# Default to raw features, fallback to UMAP output if --use-umap is specified
FEATURES_PATH = "data/features/cheap_features.csv"
UMAP_PATH = "data/embeddings/features_with_umap.csv"
BASE_EXPORT_PATH = "data/embeddings/hdbscan_results"
REP_EXPORT_PATH = "data/specs/representative_rules"

FEATURE_COLS = [
    "rho_mean", "rho_std", "flip_rate", "autocorr_t1",
    "space_autocorr", "h3_global", "space_entropy_mean",
    "activity_tail", "is_fixedpoint"
]

def run_clustering(data, min_size=50):
    """Execute HDBSCAN with standard density parameters."""
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_size,
        min_samples=15,
        metric='euclidean',
        prediction_data=True,
        core_dist_n_jobs=-1
    )
    labels = clusterer.fit_predict(data)
    probs = clusterer.probabilities_
    return labels, probs

def main():
    parser = argparse.ArgumentParser(description="CA Rule Clustering Pipeline")
    parser.add_argument(
        "--use-umap", 
        action="store_true", 
        default=False,
        help="Use 2D UMAP embeddings instead of high-dimensional features"
    )
    args = parser.parse_args()

    # Select input file based on mode
    if args.use_umap:
        input_path = UMAP_PATH
        mode = "umap"
    else:
        input_path = FEATURES_PATH
        mode = "highdim"

    df = pd.read_csv(input_path)
    
    # Select feature space
    if args.use_umap:
        target_data = df[["umap_1", "umap_2"]].values
    else:
        target_data = StandardScaler().fit_transform(df[FEATURE_COLS].values)

    print(f"Executing HDBSCAN in {mode} mode...")
    df["cluster"], df["membership_prob"] = run_clustering(target_data)

    # Export clustering labels
    os.makedirs(os.path.dirname(BASE_EXPORT_PATH), exist_ok=True)
    export_file = f"{BASE_EXPORT_PATH}_{mode}.csv"
    df[["rule_id", "cluster", "membership_prob"]].to_csv(export_file, index=False)

    # Extract representatives for downstream FFT/GNN analysis
    representatives = []
    valid_clusters = [c for c in df["cluster"].unique() if c != -1]

    for cluster_id in valid_clusters:
        subset = df[df["cluster"] == cluster_id]
        
        # Select centroids (high stability) and boundary rules (transition states)
        centers = subset.nlargest(30, "membership_prob")
        borders = subset.nsmallest(10, "membership_prob")
        
        combined = pd.concat([centers, borders]).drop_duplicates(subset="rule_id")
        for _, row in combined.iterrows():
            representatives.append({
                "rule_id": int(row["rule_id"]),
                "cluster": cluster_id,
                "type": "center" if row["membership_prob"] > 0.5 else "border"
            })

    rep_df = pd.DataFrame(representatives)
    os.makedirs(os.path.dirname(REP_EXPORT_PATH), exist_ok=True)
    rep_df.to_csv(f"{REP_EXPORT_PATH}_{mode}.csv", index=False)

    print(f"Clustering finished. Found {len(valid_clusters)} clusters.")
    print(f"Representative rules saved for GNN sampling: {len(rep_df)} items.")

if __name__ == "__main__":
    main()