import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from ca.evolve import evolve_ca_r2
from ca.init import random_init
from features.cheap import cheap_features



# Experiment configuration

N_RULES = 100000        # Number of rules to process
WIDTH = 256           # Grid width
STEPS = 2000          # Total evolution steps
BURNIN = 500          # Burn-in steps to discard
SEED = 0              # Random seed for reproducibility
N_WORKERS = None      # Number of parallel workers


def sample_rules(n):
    """Sample random r=2 rule IDs."""
    return np.random.randint(0, 2**32, size=n, dtype=np.uint32)


def process_single_rule(args):
    """
    Process a single rule: evolve CA and extract features.
    
    This function is designed to be called in parallel by multiprocessing.
    Each worker processes one rule independently.
    
    Parameters
    ----------
    args : tuple
        (rule_id, width, steps, burnin, seed_offset)
        - rule_id: Rule ID to process
        - width: Grid width
        - steps: Total evolution steps
        - burnin: Burn-in steps to discard
        - seed_offset: Offset for random seed (to ensure different init states)
    
    Returns
    -------
    dict
        Feature dictionary with 'rule_id' key
    """
    rule_id, width, steps, burnin, seed_offset = args
    
    # Set random seed for this worker (different offset per worker)
    np.random.seed(SEED + seed_offset)
    
    # Initialize random state
    init_state = random_init(width, p=0.5)
    
    # Evolve CA
    X = evolve_ca_r2(
        rule_id=int(rule_id),
        init_state=init_state,
        steps=steps
    )
    
    # Drop burn-in period
    X = X[burnin:]
    
    # Extract features
    feats = cheap_features(X)
    feats["rule_id"] = int(rule_id)
    
    return feats


def main():
    """Main function to process multiple CA rules in parallel."""
    # Set global random seed
    np.random.seed(SEED)
    
    # Sample rule IDs
    rule_ids = sample_rules(N_RULES)
    
    # Determine number of workers
    n_workers = N_WORKERS if N_WORKERS is not None else cpu_count()
    print(f"Processing {N_RULES} rules with {n_workers} parallel workers")
    print(f"Configuration: WIDTH={WIDTH}, STEPS={STEPS}, BURNIN={BURNIN}")
    
    # Prepare arguments for parallel processing
    # Each worker gets a unique seed offset to ensure different initial states
    args_list = [
        (rid, WIDTH, STEPS, BURNIN, i)
        for i, rid in enumerate(rule_ids)
    ]
    
    # Process rules in parallel
    with Pool(processes=n_workers) as pool:
        # Use imap for progress tracking with tqdm
        records = list(tqdm(
            pool.imap(process_single_rule, args_list),
            total=N_RULES,
            desc="Processing rules"
        ))
    
    # Convert to DataFrame and save
    df = pd.DataFrame(records)
    output_path = "data/features/cheap_features.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} records to {output_path}")


if __name__ == "__main__":
    main()
