"""
Utility Functions for ECA-GNN
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import torch
    from torch_geometric.data import Data
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Wolfram Classification
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

# Controversial Rules Database
CONTROVERSIAL_RULES = {
    18: {
        'wolfram_label': 'III',
        'disputed_as': ['II', 'IV'],
        'reason': 'Propagating kink structures with non-trivial interactions',
        'priority': 'medium'
    },
    54: {
        'wolfram_label': 'IV', 
        'disputed_as': ['III'],
        'reason': 'Gliders similar to Rule 110 but universal computation unclear',
        'priority': 'high' 
    },
    62: {
        'wolfram_label': 'II',
        'disputed_as': ['IV'],
        'reason': 'IC-dependent: periodic (Class II) or glider patterns (Class IV)',
        'priority': 'high'
    },
    73: {
        'wolfram_label': 'II',
        'disputed_as': ['III'],
        'reason': 'IC-sensitive: periodic with barriers or chaotic behavior',
        'priority': 'high'
    },
    110: {
        'wolfram_label': 'IV',
        'disputed_as': [],
        'reason': 'Turing-complete proven but extremely rich dynamics make classification difficult',
        'priority': 'medium'
    },
    126: {
        'wolfram_label': 'III',
        'disputed_as': ['II', 'IV'],
        'reason': 'Hybrid of Classes 2/3/4: complex but not strictly chaotic',
        'priority': 'medium'
    },
}


def get_wolfram_class(rule_number: int) -> str:
    """Get Wolfram class label ('I', 'II', 'III', 'IV') for a rule."""
    return WOLFRAM_CLASSES.get(rule_number, 'II')


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


def get_controversial_rules(priority: Optional[str] = None) -> List[int]:
    """Get list of controversial rules."""
    if priority:
        return [r for r, info in CONTROVERSIAL_RULES.items() 
                if info['priority'] == priority]
    return list(CONTROVERSIAL_RULES.keys())


def is_controversial(rule_number: int) -> bool:
    """Check if a rule is controversial."""
    return rule_number in CONTROVERSIAL_RULES


def get_controversy_info(rule_number: int) -> Optional[Dict]:
    """Get controversy information for a rule."""
    return CONTROVERSIAL_RULES.get(rule_number)


# Data I/O
def save_graph_data(graph_data: Dict, filepath: Union[str, Path], format: str = 'pickle') -> None:
    """Save graph data to file."""
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
        raise ValueError(f"Unknown format: {format}")


def load_graph_data(filepath: Union[str, Path], format: str = 'pickle') -> Dict:
    """Load graph data from file."""
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
        raise ValueError(f"Unknown format: {format}")


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


# PyTorch Geometric Conversion
def to_pyg_data(graph_data: Dict) -> 'Data':
    """Convert graph dictionary to PyTorch Geometric Data object."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch Geometric is required")
    
    x = torch.tensor(graph_data['node_features'], dtype=torch.float)
    
    edge_list = graph_data['edges']
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    edge_attr = None
    if 'edge_features' in graph_data:
        edge_attr = torch.tensor(graph_data['edge_features'], dtype=torch.float)
    
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