import os
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

INPUT_PATH = "data/features/cheap_features.csv"
OUTPUT_CSV = "data/embeddings/features_with_umap.csv"
FIGURE_OUT = "data/figures/umap.png"

FEATURE_COLS = [
    "rho_mean", "rho_std", "flip_rate", "autocorr_t1",
    "space_autocorr", "h3_global", "space_entropy_mean",
    "activity_tail", "is_fixedpoint"
]

def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Source feature file not found at {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    x_raw = df[FEATURE_COLS].values
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)

    reducer = umap.UMAP(
        n_neighbors=25,
        min_dist=0.15,
        metric='euclidean',
        random_state=42,
        n_jobs=-1
    )

    print(f"Running UMAP on {len(df)} rules...")
    embedding = reducer.fit_transform(x_scaled)

    df["umap_1"] = embedding[:, 0]
    df["umap_2"] = embedding[:, 1]

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    
    os.makedirs(os.path.dirname(FIGURE_OUT), exist_ok=True)
    plt.figure(figsize=(12, 10))
    plt.scatter(
        df["umap_1"], 
        df["umap_2"], 
        s=1, 
        c=df["h3_global"], 
        cmap='magma', 
        alpha=0.6
    )
    plt.colorbar(label='H3 Global Entropy')
    plt.title("UMAP Manifold Projection of r=2 CA Rules")
    plt.tight_layout()
    plt.savefig(FIGURE_OUT, dpi=300)
    print(f"Process complete. Embedding saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()