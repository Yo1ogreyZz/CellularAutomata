"""Utility functions for ECA-GNN"""

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
    Data = None



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
    return WOLFRAM_CLASSES.get(rule_number, 'II')


def get_wolfram_class_id(rule_number: int) -> int:
    class_label = get_wolfram_class(rule_number)
    return CLASS_TO_ID[class_label]


def get_wolfram_class_name(rule_number: int) -> str:
    class_label = get_wolfram_class(rule_number)
    return CLASS_NAMES[class_label]


def get_rules_by_class() -> Dict[str, List[int]]:
    result = {'I': [], 'II': [], 'III': [], 'IV': []}
    for rule in range(256):
        label = get_wolfram_class(rule)
        result[label].append(rule)
    return result


def get_class_distribution() -> Dict[str, int]:
    classes = get_rules_by_class()
    return {label: len(rules) for label, rules in classes.items()}



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
        'reason': 'Turing-complete proven but extremely rich dynamics',
        'priority': 'medium'
    },
    126: {
        'wolfram_label': 'III',
        'disputed_as': ['II', 'IV'],
        'reason': 'Hybrid of Classes 2/3/4: complex but not strictly chaotic',
        'priority': 'medium'
    },
    224: {
        'wolfram_label': 'INCONSISTENT',
        'disputed_as': ['multiple'],
        'reason': 'Literature shows contradictory classifications',
        'priority': 'high'
    }
}


def get_controversial_rules(priority: Optional[str] = None) -> List[int]:
    if priority:
        return [r for r, info in CONTROVERSIAL_RULES.items() 
                if info['priority'] == priority]
    return list(CONTROVERSIAL_RULES.keys())


def is_controversial(rule_number: int) -> bool:
    """Check if a rule is controversial"""
    return rule_number in CONTROVERSIAL_RULES


def get_controversy_info(rule_number: int) -> Optional[Dict]:
    """Get controversy information for a rule"""
    return CONTROVERSIAL_RULES.get(rule_number)



def save_graph_data(graph_data: Dict, 
                   filepath: Union[str, Path], 
                   format: str = 'pickle') -> None:
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
    """Convert numpy arrays to lists for JSON serialization"""
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
    """Convert lists back to numpy arrays after JSON loading"""
    result = {}
    for key, value in data.items():
        if key in ['node_features', 'edge_features', 'spacetime']:
            result[key] = np.array(value)
        else:
            result[key] = value
    return result



def to_pyg_data(graph_data: Dict) -> 'Data':
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch Geometric is required. Install with:\n"
            "  pip install torch torch-geometric"
        )
    
    x = torch.tensor(graph_data['node_features'], dtype=torch.float)
    edge_list = graph_data['edges']
    
    # 处理边索引：确保即使边列表为空，也创建正确形状的张量 [2, num_edges]
    if len(edge_list) == 0:
        # 如果没有边，创建空的 edge_index，形状为 [2, 0]
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # 处理 edge_attr：确保与边的数量一致
    edge_attr = None
    if 'edge_features' in graph_data and graph_data['edge_features'] is not None:
        if len(graph_data['edge_features']) > 0:
            edge_attr = torch.tensor(graph_data['edge_features'], dtype=torch.float)
        # 如果 edge_features 存在但是空数组，edge_attr 保持为 None
    # 注意：对于批处理，所有图必须要么都有 edge_attr，要么都没有
    # 如果需要统一处理，可以在这里添加逻辑
    
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


def normalize_dataset_edge_attr(dataset: List['Data'], graph_type: str) -> List['Data']:
    """
    根据图类型统一处理数据集的 edge_attr，确保批处理兼容性
    
    Args:
        dataset: PyTorch Geometric Data 对象列表
        graph_type: 图类型 ('truth_table', 'dependency', 'evolution', 'pattern_vocabulary')
    
    Returns:
        处理后的数据集
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch Geometric is required")
    
    # 根据图类型决定是否需要统一处理 edge_attr
    needs_normalization = {
        'truth_table': False,      # truth_table 没有 edge_attr
        'dependency': False,       # dependency 所有图都有 edge_attr
        'evolution': False,        # evolution 没有 edge_attr
        'pattern_vocabulary': True # pattern_vocabulary 可能有些图有 edge_attr，有些没有
    }
    
    if graph_type not in needs_normalization:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    
    if not needs_normalization[graph_type]:
        # 不需要统一处理，直接返回
        return dataset
    
    # 需要统一处理（pattern_vocabulary）
    # 检查是否有图包含 edge_attr
    has_edge_attr_list = [hasattr(data, 'edge_attr') and data.edge_attr is not None for data in dataset]
    has_any_edge_attr = any(has_edge_attr_list)
    
    if not has_any_edge_attr:
        # 所有图都没有 edge_attr，不需要处理
        return dataset
    
    # 有些图有 edge_attr，需要确保所有图都有
    # 获取第一个有 edge_attr 的图的特征维度
    sample_data = next((d for d in dataset if hasattr(d, 'edge_attr') and d.edge_attr is not None), None)
    if sample_data is not None:
        edge_dim = sample_data.edge_attr.size(1)
        for i, data in enumerate(dataset):
            if not has_edge_attr_list[i]:
                # 为没有 edge_attr 的图创建默认值
                # 安全地获取边的数量：检查 edge_index 是否存在及其维度
                if hasattr(data, 'edge_index') and data.edge_index is not None:
                    if data.edge_index.dim() >= 2:
                        num_edges = data.edge_index.size(1)
                    elif data.edge_index.dim() == 1:
                        # 如果是一维，可能是空的或格式不对
                        num_edges = 0
                    else:
                        num_edges = 0
                else:
                    # 没有 edge_index
                    num_edges = 0
                
                if num_edges > 0:
                    data.edge_attr = torch.zeros(num_edges, edge_dim, dtype=torch.float, device=data.x.device)
                else:
                    # 如果没有边，创建空的 edge_attr 以保持一致性
                    data.edge_attr = torch.zeros(0, edge_dim, dtype=torch.float, device=data.x.device)
    
    return dataset



def get_representative_rules() -> Dict[str, List[int]]:
    """Get representative rules for each Wolfram class"""
    return {
        'I': [0, 8, 32, 136, 160],
        'II': [4, 37, 51, 108, 184, 232],
        'III': [18, 22, 30, 45, 126, 150],
        'IV': [41, 54, 106, 110]
    }


def get_test_rules() -> List[int]:
    """Get one test rule from each class"""
    return [0, 4, 30, 110]  # [I, II, III, IV]


def compute_graph_statistics(graph_data: Dict) -> Dict:
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


# Quick test
if __name__ == "__main__":
    print("Testing utils module...")
    
    # Test classification functions
    print(f"\nRule 30: {get_wolfram_class_name(30)}")
    print(f"Rule 110: {get_wolfram_class_name(110)}")
    
    # Test class distribution
    dist = get_class_distribution()
    print(f"\nClass distribution: {dist}")
    print(f"Total rules: {sum(dist.values())}")
    
    # Test controversial rules
    print(f"\nHigh priority controversial rules: {get_controversial_rules('high')}")
    print(f"Rule 54 controversial: {is_controversial(54)}")
    if is_controversial(54):
        info = get_controversy_info(54)
        print(f"  Reason: {info['reason']}")



def reflect_rule(rule: int) -> int:
    """Mirror reflection of rule"""
    lookup = format(rule, '08b')[::-1]
    new_lookup = ['0'] * 8
    
    for i in range(8):
        left = (i >> 2) & 1
        center = (i >> 1) & 1
        right = i & 1
        
        mirrored_idx = (right << 2) | (center << 1) | left
        new_lookup[mirrored_idx] = lookup[i]
    
    return int(''.join(new_lookup[::-1]), 2)


def complement_rule(rule: int) -> int:
    """Bit complement of rule"""
    lookup = format(rule, '08b')[::-1]
    new_lookup = ['0'] * 8
    
    for i in range(8):
        left = (i >> 2) & 1
        center = (i >> 1) & 1
        right = i & 1
        
        comp_idx = ((1-left) << 2) | ((1-center) << 1) | (1-right)
        new_lookup[comp_idx] = '1' if lookup[i] == '0' else '0'
    
    return int(''.join(new_lookup[::-1]), 2)


def get_equivalent_rules(rule: int) -> set:
    """Get all symmetric equivalents of a rule"""
    equivalents = {rule}
    
    reflected = reflect_rule(rule)
    equivalents.add(reflected)
    
    complemented = complement_rule(rule)
    equivalents.add(complemented)
    
    reflected_comp = reflect_rule(complemented)
    equivalents.add(reflected_comp)
    
    return equivalents


def get_canonical_rule(rule: int) -> int:
    """Get canonical representative of equivalence class"""
    equivalents = get_equivalent_rules(rule)
    return min(equivalents)


def build_equivalence_classes() -> Dict[int, set]:
    """Build all equivalence classes for 256 rules"""
    classes = {}
    processed = set()
    
    for rule in range(256):
        if rule in processed:
            continue
        
        canonical = get_canonical_rule(rule)
        equiv_set = get_equivalent_rules(rule)
        
        classes[canonical] = equiv_set
        processed.update(equiv_set)
    
    return classes


def get_canonical_rules() -> List[int]:
    """Get list of canonical rules (88 total)"""
    classes = build_equivalence_classes()
    return sorted(list(classes.keys()))


def validate_symmetry() -> bool:
    """Validate that we get exactly 88 equivalence classes"""
    classes = build_equivalence_classes()
    n_classes = len(classes)
    total_rules = sum(len(equiv_set) for equiv_set in classes.values())
    
    assert n_classes == 88, f"Expected 88 classes, got {n_classes}"
    assert total_rules == 256, f"Expected 256 total rules, got {total_rules}"
    
    print(f"Validation passed: {n_classes} equivalence classes covering {total_rules} rules")
    return True
    