import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import hdbscan

# Config
FEATURE_PATH = "data/features/cheap_features.csv"
LABEL_OUT = "data/embeddings/cheap_hdbscan_labels.csv"
REP_OUT = "data/specs/representative_rules.csv"

FEATURE_COLS = [
    "rho_mean",
    "rho_std",
    "flip_rate",
    "autocorr_t1",
    "H3",
]

MIN_CLUSTER_SIZE = 50      # Conservative but stable for 100k data points
MIN_SAMPLES = 10           # Controls boundary/noise sensitivity
N_REP_CENTER = 10          # Number of center representatives per cluster
N_REP_BORDER = 10          # Number of border representatives per cluster


# Load data
df = pd.read_csv(FEATURE_PATH)
X = df[FEATURE_COLS].values

# Remove degenerate rules (e.g., std=0)
valid_mask = np.std(X, axis=1) > 0
df = df.loc[valid_mask].reset_index(drop=True)
X = X[valid_mask]

# Standardize features (required)
scaler = StandardScaler()
Xz = scaler.fit_transform(X)


# HDBSCAN
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    metric="euclidean",
    core_dist_n_jobs=-1,
)

labels = clusterer.fit_predict(Xz)
probs = clusterer.probabilities_

df["cluster"] = labels
df["membership_prob"] = probs

# Save all labels
os.makedirs(os.path.dirname(LABEL_OUT), exist_ok=True)
df[["rule_id", "cluster", "membership_prob"]].to_csv(
    LABEL_OUT, index=False
)

# Select representative rules
rep_rows = []

for c in sorted(df["cluster"].unique()):
    if c == -1:
        continue  # Skip noise points

    sub = df[df["cluster"] == c]

    # Center rules (high membership probability)
    center = sub.sort_values(
        "membership_prob", ascending=False
    ).head(N_REP_CENTER)

    # Border rules (low membership probability)
    border = sub.sort_values(
        "membership_prob", ascending=True
    ).head(N_REP_BORDER)

    for _, r in pd.concat([center, border]).iterrows():
        rep_rows.append({
            "rule_id": r["rule_id"],
            "cluster": c,
            "membership_prob": r["membership_prob"],
            "type": "center" if r in center.values else "border"
        })

rep_df = pd.DataFrame(rep_rows)

os.makedirs(os.path.dirname(REP_OUT), exist_ok=True)
rep_df.to_csv(REP_OUT, index=False)


n_clusters = len([c for c in df["cluster"].unique() if c != -1])
noise_ratio = np.mean(df["cluster"] == -1)

print(f"[DONE] HDBSCAN clusters: {n_clusters}")
print(f"[DONE] Noise ratio: {noise_ratio:.3f}")
print(f"[DONE] Representative rules saved to {REP_OUT}")
