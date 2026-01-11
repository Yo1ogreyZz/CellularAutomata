import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ca.evolve import evolve_ca_r2
from ca.init import random_init

INPUT_CSV = "data/specs/representative_rules_umap.csv"
OUTPUT_DIR = "data/spectral_analysis"

def compute_2d_fft(X):
    fft = np.fft.fft2(X - np.mean(X))
    fft_shift = np.fft.fftshift(fft)
    return np.abs(fft_shift)**2

def main():
    df = pd.read_csv(INPUT_CSV)
    WIDTH = 256
    STEPS = 1024
    BURNIN = 500
    
    os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)

    for _, row in df.head(10).iterrows():
        rid = int(row['rule_id'])
        cluster = row['cluster']
        
        init = random_init(WIDTH)
        X = evolve_ca_r2(rid, init, STEPS)
        psd = compute_2d_fft(X[BURNIN:])
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(X[BURNIN:BURNIN+300, :], cmap='binary', aspect='auto')
        ax[0].set_title(f"Rule {rid}")
        
        ax[1].imshow(np.log(psd + 1), cmap='magma', aspect='auto')
        ax[1].set_title(f"FFT (Cluster {cluster})")
        
        plt.savefig(f"{OUTPUT_DIR}/plots/rule_{rid}_cluster_{cluster}.png")
        plt.close()

    print(f"Spectral validation plots saved to {OUTPUT_DIR}/plots")

if __name__ == "__main__":
    main()