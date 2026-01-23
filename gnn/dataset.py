"""Multi-view dataset for CA rule classification (3 views: symbol, debruijn, dependency)."""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


CLASS_NAMES = ['Homogeneous', 'Stable', 'Propagate', 'Chaotic', 'Complex']
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class CAMultiViewDataset(Dataset):
    """Dataset providing 3 graph views for each CA rule (no lattice)."""
    
    CLASS_NAMES = CLASS_NAMES
    
    def __init__(self, data_path, split_indices=None, dependency_T=4, dependency_W=7):
        """
        Args:
            data_path: Path to CSV file or .pt cache file
            split_indices: Optional indices for train/val/test split
            dependency_T: Time steps for dependency graph
            dependency_W: Width for dependency graph
        """
        self.dependency_T = dependency_T
        self.dependency_W = dependency_W
        self.from_cache = data_path.endswith('.pt')
        
        if self.from_cache:
            self._load_from_cache(data_path, split_indices)
        else:
            self._load_from_csv(data_path, split_indices)
    
    def _load_from_cache(self, pt_path, split_indices):
        """Load pre-computed PyG graphs from .pt file."""
        samples = torch.load(pt_path, weights_only=False)
        
        if split_indices is not None:
            samples = [samples[i] for i in split_indices]
        
        self.samples = samples
        self.rule_ids = [s['rule_id'].item() for s in samples]
        self.labels = [s['label'].item() for s in samples]
        
        self._print_stats(pt_path)
    
    def _load_from_csv(self, csv_path, split_indices):
        """Load from CSV, generate graphs on-the-fly."""
        df = pd.read_csv(csv_path)
        
        if split_indices is not None:
            df = df.iloc[split_indices].reset_index(drop=True)
        
        self.samples = None  # generate on-the-fly
        self.rule_ids = df['ruleId'].values
        self.labels = self._extract_labels(df)
        
        self._print_stats(csv_path)
    
    def _extract_labels(self, df):
        count_cols = ['count_Homogeneous', 'count_Stable', 'count_Propagate', 
                      'count_Chaotic', 'count_Complex']
        
        if all(col in df.columns for col in count_cols):
            return np.argmax(df[count_cols].values, axis=1)
        
        if 'className' in df.columns:
            return np.array([CLASS_TO_IDX[c] for c in df['className'].values])
        
        raise ValueError(f"Need count_* columns or className. Found: {df.columns.tolist()}")
    
    def _print_stats(self, path):
        print(f"Loaded {len(self)} samples from {path}")
        unique, counts = np.unique(self.labels, return_counts=True)
        dist = {CLASS_NAMES[i]: c for i, c in zip(unique, counts)}
        print(f"Distribution: {dist}")
    
    def __len__(self):
        return len(self.rule_ids)
    
    def __getitem__(self, idx):
        if self.samples is not None:
            # from cache
            return self.samples[idx]
        
        # generate on-the-fly
        from ca.graphs import CAGraphRepresentation
        
        rule_id = int(self.rule_ids[idx])
        label = int(self.labels[idx])
        
        gen = CAGraphRepresentation(rule_id)
        
        return {
            'symbol': gen.get_symbol_graph(),
            'debruijn': gen.get_debruijn_graph(),
            'dependency': gen.get_dependency_graph(T=self.dependency_T, W=self.dependency_W),
            'label': torch.tensor(label, dtype=torch.long),
            'rule_id': torch.tensor(rule_id, dtype=torch.long)
        }


def create_splits(csv_path, train_ratio=0.7, val_ratio=0.15, seed=42):
    df = pd.read_csv(csv_path)
    n = len(df)
    
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


def create_stratified_splits(csv_path, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Stratified split for class imbalance."""
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(csv_path)
    
    if 'className' in df.columns:
        labels = df['className'].values
    else:
        count_cols = ['count_Homogeneous', 'count_Stable', 'count_Propagate', 
                      'count_Chaotic', 'count_Complex']
        labels = np.argmax(df[count_cols].values, axis=1)
    
    indices = np.arange(len(df))
    
    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio, stratify=labels, random_state=seed
    )
    
    temp_labels = labels[temp_idx]
    val_size = val_ratio / (1 - train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_size, stratify=temp_labels, random_state=seed
    )
    
    return train_idx, val_idx, test_idx


def custom_collate_fn(batch):
    from torch_geometric.data import Batch
    
    return {
        'symbol': Batch.from_data_list([item['symbol'] for item in batch]),
        'debruijn': Batch.from_data_list([item['debruijn'] for item in batch]),
        'dependency': Batch.from_data_list([item['dependency'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
        'rule_id': torch.stack([item['rule_id'] for item in batch])
    }


def get_class_weights(data_path):
    """Compute class weights for imbalanced data."""
    if data_path.endswith('.pt'):
        samples = torch.load(data_path, weights_only=False)
        labels = [s['label'].item() for s in samples]
    else:
        df = pd.read_csv(data_path)
        if 'className' in df.columns:
            labels = [CLASS_TO_IDX[c] for c in df['className'].values]
        else:
            count_cols = ['count_Homogeneous', 'count_Stable', 'count_Propagate', 
                          'count_Chaotic', 'count_Complex']
            labels = np.argmax(df[count_cols].values, axis=1).tolist()
    
    counts = np.bincount(labels, minlength=5)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(CLASS_NAMES)
    
    return torch.tensor(weights, dtype=torch.float)
