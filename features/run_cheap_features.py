import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os

from ca.evolve import evolve_ca_r2
from features.cheap import cheap_features

# Experiment configuration
N_RULES = 100000
WIDTH = 256           
STEPS = 2000          
BURNIN = 500          
SEED = 0              
N_WORKERS = None      

def sample_rules(n):
    return np.random.randint(0, 2**32, size=n, dtype=np.uint32)

def process_single_rule(args):
    rule_id, width, steps, burnin, seed_offset = args
    np.random.seed(SEED + seed_offset)
    
    init_state = random_init(width, p=0.5)
    
    X = evolve_ca_r2(
        rule_id=int(rule_id),
        init_state=init_state,
        steps=steps
    )
    
    # Extract features from the full evolution to allow convergence check
    # The burn-in windowing is handled inside specific feature calculations if needed
    # but for simplicity we pass the post-burnin slice for statistical features
    X_stable = X[burnin:]
    
    feats = cheap_features(X_stable)
    feats["rule_id"] = int(rule_id)
    
    return feats

def main():
    np.random.seed(SEED)
    rule_ids = sample_rules(N_RULES)
    
    n_workers = N_WORKERS if N_WORKERS is not None else cpu_count()
    print(f"Processing {N_RULES} rules with {n_workers} workers")
    
    args_list = [
        (rid, WIDTH, STEPS, BURNIN, i)
        for i, rid in enumerate(rule_ids)
    ]
    
    with Pool(processes=n_workers) as pool:
        # Use chunksize to reduce communication overhead
        # Optimal chunksize: total // (4 * n_workers) for better load balancing
        chunksize = max(1, N_RULES // (4 * n_workers))
        records = list(tqdm(
            pool.imap(process_single_rule, args_list, chunksize=chunksize),
            total=N_RULES,
            desc="Extracting features"
        ))
    
    df = pd.DataFrame(records)
    output_dir = "data/features"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cheap_features.csv")
    
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} records to {output_path}")

if __name__ == "__main__":
    main()