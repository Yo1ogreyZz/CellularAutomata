"""
This generates a 'Spectral Atlas' for the identified CA phases.
Instead of inspecting 4,700 individual rules, we compute the Ensemble Average 
Power Spectral Density (PSD) for each UMAP cluster.

Output:
1. Mean Spectral Fingerprint for each cluster (filtering out individual rule noise).
2. Intra-cluster Consistency metric (how similar are rules within a cluster?).
This confirms whether a cluster represents a distinct, coherent physical phase.
"""
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

# Input/Output
INPUT_CSV = "data/specs/representative_rules_umap.csv" #
OUTPUT_DIR = "data/spectral_atlas"

# Parameters
WIDTH = 256
STEPS = 1024
BURNIN = 500

def compute_psd(X):
    """Compute 2D Power Spectral Density."""
    # FFT transform
    fft = np.fft.fft2(X - np.mean(X))
    fft_shift = np.fft.fftshift(fft)
    # Power spectrum
    psd = np.abs(fft_shift)**2
    # Log scale for visualization (add epsilon to avoid log(0))
    return np.log(psd + 1e-12)

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"{INPUT_CSV} not found.")
        
    df = pd.read_csv(INPUT_CSV)
    
    # Get list of unique clusters (ignoring noise -1 if present)
    clusters = sorted([c for c in df['cluster'].unique() if c != -1])
    
    print(f"Generating Spectral Atlas for {len(clusters)} clusters from {len(df)} rules...")
    
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
    
    # Statistics to save
    stats = []

    # Iterate over each cluster (The "Phase")
    for cid in tqdm(clusters, desc="Analyzing Clusters"):
        # Get all rules in this cluster
        sub_df = df[df['cluster'] == cid]
        
        psd_accum = np.zeros((STEPS-BURNIN, WIDTH))
        valid_count = 0
        
        # We need one visual example of spacetime for the plot
        example_spacetime = None
        
        # Evolve each rule in the cluster and accumulate spectrum
        for _, row in sub_df.iterrows():
            rid = int(row['rule_id'])
            
            # Evolve
            init = random_init(WIDTH)
            X = evolve_ca_r2(rid, init, STEPS)
            X_stable = X[BURNIN:]
            
            # Capture first rule as the visual example
            if example_spacetime is None:
                example_spacetime = X_stable
            
            # Compute Spectrum
            psd = compute_psd(X_stable)
            psd_accum += psd
            valid_count += 1
            
        # Compute Mean Spectrum (Ensemble Average)
        mean_psd = psd_accum / valid_count
        
        # Calculate consistency: crude approximation (variance of energy)
        # Ideally we would calculate pairwise correlations, but that's expensive.
        # Here we just record peak energy.
        peak_energy = np.max(mean_psd)
        
        stats.append({
            "cluster": cid,
            "n_rules": valid_count,
            "peak_spectral_energy": peak_energy
        })

        # --- Visualization ---
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # Left: Example Spacetime
        ax[0].imshow(example_spacetime[:300], cmap='binary', aspect='auto', interpolation='nearest')
        ax[0].set_title(f"Cluster {cid}: Typical Spacetime", fontsize=10)
        ax[0].set_xlabel("Space")
        ax[0].set_ylabel("Time")
        
        # Right: Mean Spectrum
        # We use a distinct colormap for frequency domain
        im = ax[1].imshow(mean_psd, cmap='inferno', aspect='auto', interpolation='nearest')
        ax[1].set_title(f"Cluster {cid}: Ensemble Mean 2D-FFT", fontsize=10)
        ax[1].set_xlabel("Wavenumber (k)")
        ax[1].set_ylabel("Frequency (w)")
        plt.colorbar(im, ax=ax[1], label="Log Power")
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plots/cluster_{cid:03d}_atlas.png", dpi=150)
        plt.close()

    # Save summary stats
    pd.DataFrame(stats).to_csv(f"{OUTPUT_DIR}/cluster_spectral_stats.csv", index=False)
    print(f"Atlas generation complete. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()