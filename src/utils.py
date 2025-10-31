"""
Utility Functions for ECA-GNN

Provides:
- Wolfram classification labels
- Data saving/loading
- PyTorch Geometric conversion
- Batch processing helpers
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

# Try to import PyTorch Geometric (optional)
try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Wolfram Classification
# =============================================================================

WOLFRAM_CLASSES = {
    # Class I: Uniform (8 rules)
    0: 'I', 8: 'I', 32: 'I', 40: 'I', 128: 'I', 136: 'I', 160: 'I', 168: 'I',

    # Class II: Periodic (65 rules)
    1: 'II', 2: 'II', 3: 'II', 4: 'II', 5: 'II', 6: 'II', 7: 'II', 9: 'II',
    10: 'II', 11: 'II', 12: 'II', 13: 'II', 14: 'II', 15: 'II', 19: 'II', 23: 'II',
    24: 'II', 25: 'II', 26: 'II', 27: 'II', 28: 'II', 29: 'II', 33: 'II', 34: 'II',
    35: 'II', 36: 'II', 37: 'II', 38: 'II', 42: 'II', 43: 'II', 44: 'II', 46: 'II',
    50: 'II', 51: 'II', 56: 'II', 57: 'II', 58: 'II', 62: 'II', 72: 'II', 73: 'II',
    74: 'II', 76: 'II', 77: 'II', 78: 'II', 94: 'II', 104: 'II', 108: 'II',
    130: 'II', 132: 'II', 134: 'II', 138: 'II', 140: 'II', 142: 'II', 152: 'II',
    154: 'II', 156: 'II', 162: 'II', 164: 'II', 170: 'II', 172: 'II', 178: 'II',
    184: 'II', 200: 'II', 204: 'II', 232: 'II',

    # Class III: Chaotic (11 rules)
    18: 'III', 22: 'III', 30: 'III', 45: 'III', 60: 'III', 90: 'III', 105: 'III',
    122: 'III', 126: 'III', 146: 'III', 150: 'III',

    # Class IV: Complex (4 rules)
    41: 'IV', 54: 'IV', 106: 'IV', 110: 'IV'
}

CLASS_TO_ID = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
ID_TO_CLASS = {0: 'I', 1: 'II', 2: 'III', 3: 'IV'}

CLASS_NAMES = {
    'I': 'Class I (Uniform)', 
    'II': 'Class II (Periodic)',
    'III': 'Class III (Chaotic)', 
    'IV': 'Class IV (Complex)'
}


def get_wolfram_class(rule_number: int) -> str:
    """Get Wolfram class label ('I', 'II', 'III', 'IV') for a rule."""
    return WOLFRAM_CLASSES.get(rule_number, 'II')  # Default to Class II


def get_wolfram_class_id(rule_number: int) -> int:
    """Get integer class ID (0-3) for a rule."""
    return CLASS_TO_ID[get_wolfram_class(rule_number)]


def get_wolfram_class_name(rule_number: int) -> str:
    """Get human-readable class name for a rule."""
    label = get_wolfram_class(rule_number)
    return CLASS_NAMES[label]


def get_rules_by_class() -> Dict[str, List[int]]:
    """Get all 256 rules grouped by class label."""
    result = {'I': [], 'II': [], 'III': [], 'IV': []}
    for rule in range(256):
        label = get_wolfram_class(rule)
        result[label].append(rule)
    return result


def get_class_distribution() -> Dict[str, int]:
    """Get the count of rules in each Wolfram class."""
    classes = get_rules_by_class()
    return {label: len(rules) for label, rules in classes.items()}


# =============================================================================
# Data Saving/Loading
# =============================================================================

def save_graph_data(graph_data: Dict, 
                   filepath: Union[str, Path], 
                   format: str = 'pickle') -> None:
    """
    Save graph data to file.
    
    Args:
        graph_data: Dictionary containing graph information
        filepath: Path to save file
        format: 'pickle' or 'json'
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(graph_data, f)
    elif format == 'json':
        json_data = _prepare_for_json(graph_data)
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'pickle' or 'json'")


def load_graph_data(filepath: Union[str, Path], 
                   format: str = 'pickle') -> Dict:
    """
    Load graph data from file.
    
    Args:
        filepath: Path to file
        format: 'pickle' or 'json'
        
    Returns:
        Dictionary containing graph information
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if format == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif format == 'json':
        with open(filepath, 'r') as f:
            json_data = json.load(f)
        return _restore_from_json(json_data)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'pickle' or 'json'")


def _prepare_for_json(data: Dict) -> Dict:
    """Convert numpy arrays to lists for JSON serialization."""
    result = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            result[key] = [
                item.tolist() if isinstance(item, np.ndarray) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def _restore_from_json(data: Dict) -> Dict:
    """Convert lists back to numpy arrays after JSON loading."""
    result = {}
    for key, value in data.items():
        if key in ['node_features', 'edge_features']:
            result[key] = np.array(value)
        else:
            result[key] = value
    return result


# =============================================================================
# PyTorch Geometric Conversion
# =============================================================================

def to_pyg_data(graph_data: Dict) -> 'Data':
    """
    Convert graph dictionary to PyTorch Geometric Data object.
    
    Args:
        graph_data: Dictionary with keys:
            - node_features: np.ndarray
            - edges: List[Tuple[int, int]]
            - edge_features: np.ndarray (optional)
            - rule_number: int
            - graph_type: str
            
    Returns:
        PyG Data object
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch Geometric is required. Install with:\n"
            "  pip install torch torch-geometric"
        )
    
    # Node features
    x = torch.tensor(graph_data['node_features'], dtype=torch.float)
    
    # Edge index
    edge_list = graph_data['edges']
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Edge features (optional)
    edge_attr = None
    if 'edge_features' in graph_data:
        edge_attr = torch.tensor(graph_data['edge_features'], dtype=torch.float)
    
    # Label
    rule_number = graph_data['rule_number']
    y = torch.tensor([get_wolfram_class_id(rule_number)], dtype=torch.long)
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        rule_number=rule_number,
        graph_type=graph_data['graph_type']
    )
    
    return data


def batch_to_pyg_dataset(graph_dict_list: List[Dict], 
                         save_dir: Optional[Union[str, Path]] = None) -> List['Data']:
    """
    Convert multiple graph dictionaries to PyG Data objects.
    
    Args:
        graph_dict_list: List of graph dictionaries
        save_dir: Optional directory to save individual .pt files
        
    Returns:
        List of PyG Data objects
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch Geometric is required. Install with:\n"
            "  pip install torch torch-geometric"
        )
    
    dataset = []
    for graph_data in graph_dict_list:
        pyg_data = to_pyg_data(graph_data)
        dataset.append(pyg_data)
        
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            rule_num = graph_data['rule_number']
            graph_type = graph_data['graph_type']
            filename = f"rule_{rule_num}_{graph_type}.pt"
            torch.save(pyg_data, save_dir / filename)
    
    return dataset


# =============================================================================
# Batch Processing & Analysis
# =============================================================================

def batch_build_graphs(rule_numbers: List[int], 
                      method: str = 'truth_table', 
                      **kwargs) -> Dict[int, Dict]:
    """
    Build graphs for multiple rules.
    
    Args:
        rule_numbers: List of rule numbers
        method: 'truth_table', 'dependency', or 'evolution'
        **kwargs: Additional arguments for EvolutionGraph
        
    Returns:
        Dictionary mapping rule numbers to graph data
    """
    from .rule2graph import ECARule, TruthTableGraph, DependencyGraph, EvolutionGraph
    
    results = {}
    for rule_num in rule_numbers:
        rule = ECARule(rule_num)
        
        if method == 'truth_table':
            graph = TruthTableGraph(rule)
        elif method == 'dependency':
            graph = DependencyGraph(rule)
        elif method == 'evolution':
            graph = EvolutionGraph(rule, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results[rule_num] = graph.build()
    
    return results


def compute_graph_statistics(graph_data: Dict) -> Dict:
    """Compute basic statistics for a graph."""
    num_nodes = len(graph_data['nodes'])
    num_edges = len(graph_data['edges'])
    avg_degree = num_edges / num_nodes if num_nodes > 0 else 0
    
    node_features = graph_data['node_features']
    feature_dim = node_features.shape[1]
    feature_mean = node_features.mean(axis=0)
    feature_std = node_features.std(axis=0)
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': avg_degree,
        'feature_dim': feature_dim,
        'feature_mean': feature_mean.tolist(),
        'feature_std': feature_std.tolist(),
        'graph_type': graph_data['graph_type'],
        'rule_number': graph_data['rule_number']
    }


def batch_compute_statistics(graph_dict_list: List[Dict]) -> List[Dict]:
    """Compute statistics for multiple graphs."""
    return [compute_graph_statistics(g) for g in graph_dict_list]


def get_representative_rules() -> Dict[str, List[int]]:
    """Get representative rules for each Wolfram class."""
    return {
        'I': [0, 8, 32, 136, 160],
        'II': [4, 37, 90, 108, 184, 232],
        'III': [18, 22, 30, 45, 126, 150],
        'IV': [41, 54, 106, 110]
    }


def get_test_rules() -> List[int]:
    """Get one test rule from each class."""
    return [0, 90, 30, 110]  # [I, II, III, IV]