import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ca.evolve import evolve_ca_r2
from ca.init import random_init

INPUT_CSV = "data/specs/representative_rules_umap.csv"
OUTPUT_DIR = "data/spectral_atlas"

WIDTH = 256
STEPS = 1024
BURNIN = 500

def compute_psd(X):
    fft = np.fft.fft2(X - np.mean(X))
    fft_shift = np.fft.fftshift(fft)
    psd = np.abs(fft_shift)**2
    return np.log(psd + 1e-12)

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"{INPUT_CSV} not found.")
        
    df = pd.read_csv(INPUT_CSV)
    clusters = sorted([c for c in df['cluster'].unique() if c != -1])
    
    print(f"Processing {len(clusters)} clusters from {len(df)} rules...")
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
    
    stats = []

    for cid in tqdm(clusters, desc="Clusters"):
        sub_df = df[df['cluster'] == cid]
        psd_accum = np.zeros((STEPS-BURNIN, WIDTH))
        valid_count = 0
        example_spacetime = None
        
        for _, row in sub_df.iterrows():
            rid = int(row['rule_id'])
            init = random_init(WIDTH)
            X = evolve_ca_r2(rid, init, STEPS)
            X_stable = X[BURNIN:]
            
            if example_spacetime is None:
                example_spacetime = X_stable
            
            psd = compute_psd(X_stable)
            psd_accum += psd
            valid_count += 1
            
        mean_psd = psd_accum / valid_count
        peak_energy = np.max(mean_psd)
        
        stats.append({
            "cluster": cid,
            "n_rules": valid_count,
            "peak_spectral_energy": peak_energy
        })

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(example_spacetime[:300], cmap='binary', aspect='auto', interpolation='nearest')
        ax[0].set_title(f"Cluster {cid}")
        ax[0].set_xlabel("Space")
        ax[0].set_ylabel("Time")
        
        im = ax[1].imshow(mean_psd, cmap='inferno', aspect='auto', interpolation='nearest')
        ax[1].set_title(f"Cluster {cid}: Mean FFT")
        ax[1].set_xlabel("Wavenumber")
        ax[1].set_ylabel("Frequency")
        plt.colorbar(im, ax=ax[1])
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plots/cluster_{cid:03d}_atlas.png", dpi=150)
        plt.close()

    pd.DataFrame(stats).to_csv(f"{OUTPUT_DIR}/cluster_spectral_stats.csv", index=False)
    print(f"Atlas generation complete. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()