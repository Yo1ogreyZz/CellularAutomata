import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

from ca.debruijn import rule_id_to_debruijn_graph


class CADebruijnDataset(Dataset):
    def __init__(self, csv_path, split_indices=None):
        """
        Dataset for CA rules with De Bruijn graphs only.
        
        Args:
            csv_path: Path to cheap_features.csv
            split_indices: Optional list of row indices to use (for train/val/test split)
        """
        df = pd.read_csv(csv_path)
        
        if split_indices is not None:
            df = df.iloc[split_indices].reset_index(drop=True)
        
        df = df.dropna(subset=["rule_id"])
        self.rule_ids = df["rule_id"].values.astype(np.int64)
    
    def __len__(self):
        return len(self.rule_ids)
    
    def __getitem__(self, idx):
        rule_id = self.rule_ids[idx]
        # Use the function from module namespace (allows dynamic replacement in train.py)
        # This works correctly in multiprocessing because each worker imports the module
        graph = rule_id_to_debruijn_graph(rule_id)
        return graph


def create_train_val_test_splits(csv_path, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
    Create train/val/test splits from the full dataset.
    
    Args:
        csv_path: Path to cheap_features.csv
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (test = 1 - train - val)
        seed: Random seed
        
    Returns:
        train_indices, val_indices, test_indices
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["rule_id"])
    
    n = len(df)
    indices = np.arange(n)
    
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    return train_indices, val_indices, test_indices


if __name__ == "__main__":
    csv_path = "data/features/cheap_features.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        exit(1)
    
    train_idx, val_idx, test_idx = create_train_val_test_splits(csv_path)
    
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    train_dataset = CADebruijnDataset(csv_path, split_indices=train_idx)
    val_dataset = CADebruijnDataset(csv_path, split_indices=val_idx)
    test_dataset = CADebruijnDataset(csv_path, split_indices=test_idx)
    
    print(f"\nDataset loaded: {len(train_dataset)} training samples")
    
    graph = train_dataset[0]
    print("\nSample graph:")
    print(f"  Rule ID: {graph.y.item()}")
    print(f"  Graph nodes: {graph.x.shape}")
    print(f"  Graph edges: {graph.edge_index.shape}")