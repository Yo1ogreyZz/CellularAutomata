'''
This performs 'Sensitivity Analysis' or 'Feature Attribution'. 
It answers a fundamental physical question: Which of the 9 measured features 
is most responsible for the geometric structure of the rule space manifold?

It uses:
1. Spearman Correlation: Measures how individual physical quantities 
   monotonically track the manifold coordinates.
2. Random Forest Importance: A non-linear method to quantify the 'Weight' 
   of each feature in defining the system's global organization.

'''

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Configuration
DATA_PATH = "data/embeddings/features_with_umap.csv"
ATTRIBUTION_OUT = "data/specs/umap_feature_attribution.csv"

FEATURE_COLS = [
    "rho_mean", "rho_std", "flip_rate", "autocorr_t1",
    "space_autocorr", "h3_global", "space_entropy_mean",
    "activity_tail", "is_fixedpoint"
]

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Input file {DATA_PATH} not found.")

    df = pd.read_csv(DATA_PATH)
    
    # 1. Linear Correlation Analysis (Spearman for non-linear monotonic relations)
    corr_umap1 = df[FEATURE_COLS].corrwith(df["umap_1"], method='spearman')
    corr_umap2 = df[FEATURE_COLS].corrwith(df["umap_2"], method='spearman')

    # 2. Non-linear Importance Analysis using Random Forest
    # This measures how much each feature contributes to predicting the UMAP coordinates
    X = df[FEATURE_COLS].values
    y1 = df["umap_1"].values
    y2 = df["umap_2"].values

    print("Calculating non-linear feature attribution via Random Forest...")
    rf1 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    rf1.fit(X, y1)
    rf2.fit(X, y2)

    # Compile results
    attribution_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "spearman_umap1": corr_umap1.values,
        "spearman_umap2": corr_umap2.values,
        "importance_umap1": rf1.feature_importances_,
        "importance_umap2": rf2.feature_importances_,
        "total_importance": (rf1.feature_importances_ + rf2.feature_importances_) / 2
    }).sort_values(by="total_importance", ascending=False)

    # Save quantitative table
    os.makedirs(os.path.dirname(ATTRIBUTION_OUT), exist_ok=True)
    attribution_df.to_csv(ATTRIBUTION_OUT, index=False)
    
    print(f"Feature attribution table saved to {ATTRIBUTION_OUT}")
    print("\nTop 5 Contributing Features:")
    print(attribution_df.head(5)[["feature", "total_importance"]])

if __name__ == "__main__":
    main()