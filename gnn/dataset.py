"""
Multi-view dataset for CA rule classification.

Loads 4 graph views (symbol, lattice, debruijn, dependency) for each rule
and associates them with labels from Zhaoyun's dataset.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# Import graph generators - will be updated after moving graphs.py to ca/
try:
    from ca.graphs import CAGraphRepresentation
except ImportError:
    # Fallback if graphs.py is still in home directory
    import sys
    sys.path.insert(0, '/home/claude')
    from graphs import CAGraphRepresentation


class CAMultiViewDataset(Dataset):
    """
    Dataset that provides 4 graph views for each CA rule.
    
    Labels are derived from Zhaoyun's aggregated predictions.
    """
    
    # Class name mapping
    CLASS_NAMES = ['Homogeneous', 'Stable', 'Propagate', 'Chaotic', 'Complex']
    
    def __init__(self, csv_path, split_indices=None, 
                 lattice_N=20, dependency_T=4, dependency_W=7):
        """
        Args:
            csv_path: Path to generated_dataset_100rules_10seeds.csv
            split_indices: Optional indices for train/val/test split
            lattice_N: Lattice size for view 1
            dependency_T: Time steps for view 3
            dependency_W: Width for view 3
        """
        self.lattice_N = lattice_N
        self.dependency_T = dependency_T
        self.dependency_W = dependency_W
        
        # Load data
        df = pd.read_csv(csv_path)
        
        # Apply split if provided
        if split_indices is not None:
            df = df.iloc[split_indices].reset_index(drop=True)
        
        # Extract rule IDs
        self.rule_ids = df['ruleId'].values
        
        # Extract labels using majority voting
        self.labels = self._extract_labels(df)
        
        # Extract dynamical features (for future alignment tasks)
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        self.features = df[feature_cols].values if feature_cols else None
        
        # Extract prediction counts (for uncertainty quantification)
        count_cols = [col for col in df.columns if col.startswith('count_')]
        self.prediction_counts = df[count_cols].values if count_cols else None
        
        print(f"Loaded {len(self)} rules from {csv_path}")
        print(f"Label distribution: {np.bincount(self.labels)}")
    
    def _extract_labels(self, df):
        """
        Extract class labels from count columns using majority voting.
        
        Returns:
            np.array of integer labels (0-4)
        """
        count_cols = ['count_Homogeneous', 'count_Stable', 'count_Propagate', 
                      'count_Chaotic', 'count_Complex']
        
        # Check if all count columns exist
        if not all(col in df.columns for col in count_cols):
            raise ValueError(f"Missing count columns in CSV. Found: {df.columns.tolist()}")
        
        # Extract counts as numpy array
        counts = df[count_cols].values  # shape: (n_rules, 5)
        
        # Majority voting: argmax across classes
        labels = np.argmax(counts, axis=1)
        
        return labels
    
    def get_prediction_entropy(self, idx):
        """
        Compute prediction entropy for a rule (measure of uncertainty).
        
        High entropy = controversial rule (predictions are scattered)
        Low entropy = clear consensus
        
        Returns:
            float: entropy in [0, log(10)] for 10 seeds
        """
        if self.prediction_counts is None:
            return None
        
        counts = self.prediction_counts[idx]
        probs = counts / counts.sum()
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy
    
    def __len__(self):
        return len(self.rule_ids)
    
    def __getitem__(self, idx):
        """
        Returns a dictionary containing:
            - 4 graph views (PyG Data objects)
            - label (int)
            - rule_id (int)
            - features (optional, for alignment)
            - prediction_entropy (optional, for uncertainty)
        """
        rule_id = int(self.rule_ids[idx])
        label = int(self.labels[idx])
        
        # Generate all 4 graph views
        generator = CAGraphRepresentation(rule_id)
        views = generator.get_all_views(
            lattice_N=self.lattice_N,
            dependency_T=self.dependency_T,
            dependency_W=self.dependency_W
        )
        
        # Package data
        sample = {
            'symbol': views['symbol'],
            'lattice': views['lattice'],
            'debruijn': views['debruijn'],
            'dependency': views['dependency'],
            'label': torch.tensor(label, dtype=torch.long),
            'rule_id': torch.tensor(rule_id, dtype=torch.long)
        }
        
        # Add optional data
        if self.features is not None:
            sample['features'] = torch.tensor(self.features[idx], dtype=torch.float)
        
        if self.prediction_counts is not None:
            sample['prediction_entropy'] = torch.tensor(
                self.get_prediction_entropy(idx), 
                dtype=torch.float
            )
        
        return sample


def create_splits(csv_path, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Create train/val/test splits from the full dataset.
    
    Args:
        csv_path: Path to CSV file
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (test = 1 - train - val)
        seed: Random seed
    
    Returns:
        train_indices, val_indices, test_indices
    """
    df = pd.read_csv(csv_path)
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


def custom_collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Handles batching of 4 different graph structures.
    Each view needs separate batching.
    """
    from torch_geometric.data import Batch
    
    # Separate each view
    symbol_batch = Batch.from_data_list([item['symbol'] for item in batch])
    lattice_batch = Batch.from_data_list([item['lattice'] for item in batch])
    debruijn_batch = Batch.from_data_list([item['debruijn'] for item in batch])
    dependency_batch = Batch.from_data_list([item['dependency'] for item in batch])
    
    # Stack scalars
    labels = torch.stack([item['label'] for item in batch])
    rule_ids = torch.stack([item['rule_id'] for item in batch])
    
    collated = {
        'symbol': symbol_batch,
        'lattice': lattice_batch,
        'debruijn': debruijn_batch,
        'dependency': dependency_batch,
        'label': labels,
        'rule_id': rule_ids
    }
    
    # Add optional fields if present
    if 'features' in batch[0]:
        collated['features'] = torch.stack([item['features'] for item in batch])
    
    if 'prediction_entropy' in batch[0]:
        collated['prediction_entropy'] = torch.stack([item['prediction_entropy'] for item in batch])
    
    return collated


if __name__ == "__main__":
    # Test the dataset
    csv_path = "data/benchmark/generated_dataset_100rules_10seeds.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        print("Please ensure Zhaoyun's data is in the correct location")
        exit(1)
    
    # Create splits
    train_idx, val_idx, test_idx = create_splits(csv_path, seed=42)
    
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    # Create datasets
    train_dataset = CAMultiViewDataset(csv_path, split_indices=train_idx)
    val_dataset = CAMultiViewDataset(csv_path, split_indices=val_idx)
    test_dataset = CAMultiViewDataset(csv_path, split_indices=test_idx)
    
    # Test loading a sample
    print("\nTesting sample loading...")
    sample = train_dataset[0]
    
    print(f"\nSample structure:")
    print(f"  Rule ID: {sample['rule_id'].item()}")
    print(f"  Label: {sample['label'].item()} ({CAMultiViewDataset.CLASS_NAMES[sample['label'].item()]})")
    print(f"  Symbol graph: {sample['symbol'].num_nodes} nodes, {sample['symbol'].num_edges} edges")
    print(f"  Lattice graph: {sample['lattice'].num_nodes} nodes, {sample['lattice'].num_edges} edges")
    print(f"  De Bruijn graph: {sample['debruijn'].num_nodes} nodes, {sample['debruijn'].num_edges} edges")
    print(f"  Dependency graph: {sample['dependency'].num_nodes} nodes, {sample['dependency'].num_edges} edges")
    
    if 'features' in sample:
        print(f"  Dynamical features: {sample['features'].shape}")
    
    if 'prediction_entropy' in sample:
        print(f"  Prediction entropy: {sample['prediction_entropy'].item():.4f}")
    
    # Test DataLoader with custom collate
    from torch.utils.data import DataLoader
    
    loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=custom_collate_fn
    )
    
    print("\nTesting DataLoader...")
    batch = next(iter(loader))
    print(f"Batch size: {len(batch['label'])}")
    print(f"  Symbol batch: {batch['symbol'].num_graphs} graphs")
    print(f"  Lattice batch: {batch['lattice'].num_graphs} graphs")
    print(f"  Labels: {batch['label']}")
    
    print("\n✓ Dataset test passed!")