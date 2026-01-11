import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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
    
    corr_umap1 = df[FEATURE_COLS].corrwith(df["umap_1"], method='spearman')
    corr_umap2 = df[FEATURE_COLS].corrwith(df["umap_2"], method='spearman')

    X = df[FEATURE_COLS].values
    y1 = df["umap_1"].values
    y2 = df["umap_2"].values

    print("Computing feature importance...")
    rf1 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf1.fit(X, y1)
    rf2.fit(X, y2)

    attribution_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "spearman_umap1": corr_umap1.values,
        "spearman_umap2": corr_umap2.values,
        "importance_umap1": rf1.feature_importances_,
        "importance_umap2": rf2.feature_importances_,
        "total_importance": (rf1.feature_importances_ + rf2.feature_importances_) / 2
    }).sort_values(by="total_importance", ascending=False)

    os.makedirs(os.path.dirname(ATTRIBUTION_OUT), exist_ok=True)
    attribution_df.to_csv(ATTRIBUTION_OUT, index=False)
    
    print(f"Feature attribution table saved to {ATTRIBUTION_OUT}")
    print("\nTop 5 Contributing Features:")
    print(attribution_df.head(5)[["feature", "total_importance"]])

if __name__ == "__main__":
    main()