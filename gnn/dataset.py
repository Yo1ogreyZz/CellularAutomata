"""Multi-view dataset for CA rule classification (3 views: symbol, debruijn, dependency)."""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


CLASS_NAMES = ['Homogeneous', 'Stable', 'Propagate', 'Chaotic', 'Complex']
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


class CAMultiViewDataset(Dataset):
    CLASS_NAMES = CLASS_NAMES
    
    def __init__(self, data_path, split_indices=None, dependency_T=4, dependency_W=7,
                 filter_classes=None):
        self.dependency_T = dependency_T
        self.dependency_W = dependency_W
        self.filter_classes = filter_classes
        self.from_cache = data_path.endswith('.pt')
        
        # Update class names if filtering
        if filter_classes is not None:
            self.CLASS_NAMES = [CLASS_NAMES[i] for i in filter_classes]
            self.class_map = {old: new for new, old in enumerate(filter_classes)}
        else:
            self.class_map = None
        
        if self.from_cache:
            self._load_from_cache(data_path, split_indices)
        else:
            self._load_from_csv(data_path, split_indices)
    
    def _load_from_cache(self, pt_path, split_indices):
        """Load pre-computed PyG graphs from .pt file."""
        samples = torch.load(pt_path, weights_only=False)
        
        # Filter classes if specified
        if self.filter_classes is not None:
            samples = [s for s in samples if s['label'].item() in self.filter_classes]
            # Remap labels
            for s in samples:
                original_label = s['label'].item()
                s['label'] = torch.tensor(self.class_map[original_label], dtype=torch.long)
        
        if split_indices is not None:
            samples = [samples[i] for i in split_indices if i < len(samples)]
        
        self.samples = samples
        self.rule_ids = [s['rule_id'].item() for s in samples]
        self.labels = [s['label'].item() for s in samples]
        
        self._print_stats(pt_path)
    
    def _load_from_csv(self, csv_path, split_indices):
        df = pd.read_csv(csv_path)
        
        if split_indices is not None:
            df = df.iloc[split_indices].reset_index(drop=True)
        
        self.samples = None  # generate on-the-fly
        self.rule_ids = df['ruleId'].values
        self.labels = self._extract_labels(df)
        
        # Filter classes if specified
        if self.filter_classes is not None:
            mask = np.isin(self.labels, self.filter_classes)
            self.rule_ids = self.rule_ids[mask]
            self.labels = self.labels[mask]
            # Remap labels
            self.labels = np.array([self.class_map[l] for l in self.labels])
        
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
        dist = {self.CLASS_NAMES[i]: c for i, c in zip(unique, counts)}
        print(f"Distribution: {dist}")
    
    def __len__(self):
        return len(self.rule_ids)
    
    def __getitem__(self, idx):
        if self.samples is not None:
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


def create_splits(csv_path, train_ratio=0.7, val_ratio=0.15, seed=42, filter_classes=None):
    df = pd.read_csv(csv_path)
    
    if filter_classes is not None:
        if 'className' in df.columns:
            labels = np.array([CLASS_TO_IDX[c] for c in df['className'].values])
        else:
            count_cols = ['count_Homogeneous', 'count_Stable', 'count_Propagate', 
                          'count_Chaotic', 'count_Complex']
            labels = np.argmax(df[count_cols].values, axis=1)
        mask = np.isin(labels, filter_classes)
        n = mask.sum()
    else:
        n = len(df)
    
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]


def create_stratified_splits(csv_path, train_ratio=0.7, val_ratio=0.15, seed=42, filter_classes=None):
    """Stratified split for class imbalance."""
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(csv_path)
    
    if 'className' in df.columns:
        labels = df['className'].values
    else:
        count_cols = ['count_Homogeneous', 'count_Stable', 'count_Propagate', 
                      'count_Chaotic', 'count_Complex']
        labels = np.argmax(df[count_cols].values, axis=1)
    
    # Filter and remap if specified
    if filter_classes is not None:
        if isinstance(labels[0], str):
            mask = np.array([CLASS_TO_IDX[c] in filter_classes for c in labels])
            labels = labels[mask]
            class_map = {old: new for new, old in enumerate(filter_classes)}
            labels = np.array([class_map[CLASS_TO_IDX[c]] for c in labels])
        else:
            mask = np.isin(labels, filter_classes)
            labels = labels[mask]
            class_map = {old: new for new, old in enumerate(filter_classes)}
            labels = np.array([class_map[l] for l in labels])
        indices = np.arange(len(labels))
    else:
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


def get_class_weights(data_path, filter_classes=None):
    if data_path.endswith('.pt'):
        samples = torch.load(data_path, weights_only=False)
        if filter_classes is not None:
            samples = [s for s in samples if s['label'].item() in filter_classes]
            class_map = {old: new for new, old in enumerate(filter_classes)}
            labels = [class_map[s['label'].item()] for s in samples]
            n_classes = len(filter_classes)
        else:
            labels = [s['label'].item() for s in samples]
            n_classes = 5
    else:
        df = pd.read_csv(data_path)
        if 'className' in df.columns:
            all_labels = [CLASS_TO_IDX[c] for c in df['className'].values]
        else:
            count_cols = ['count_Homogeneous', 'count_Stable', 'count_Propagate', 
                          'count_Chaotic', 'count_Complex']
            all_labels = np.argmax(df[count_cols].values, axis=1).tolist()
        
        if filter_classes is not None:
            labels = [l for l in all_labels if l in filter_classes]
            class_map = {old: new for new, old in enumerate(filter_classes)}
            labels = [class_map[l] for l in labels]
            n_classes = len(filter_classes)
        else:
            labels = all_labels
            n_classes = 5
    
    counts = np.bincount(labels, minlength=n_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    
    return torch.tensor(weights, dtype=torch.float)