import numpy as np
import torch
from torch_geometric.data import Data


def rule_id_to_debruijn_graph(rule_id):
    """
    Convert a 32-bit CA rule into a De Bruijn graph.
    
    For radius-2 CA, the De Bruijn graph has 32 nodes (one per 5-bit state).
    Edges connect states based on neighborhood overlap and transition output.
    
    Args:
        rule_id: Integer in [0, 2^32-1] encoding the transition rule
        
    Returns:
        PyG Data object with:
            - x: node features (32 x 1, just node indices for now)
            - edge_index: directed edges based on transition structure
            - edge_attr: transition outputs (0 or 1)
            - y: global graph label (the rule_id itself)
    """
    rule_id = int(rule_id)
    
    rule_table = np.array([(rule_id >> i) & 1 for i in range(32)], dtype=np.int64)
    
    edge_list = []
    edge_attrs = []
    
    for source in range(32):
        source_bits = [(source >> (4 - i)) & 1 for i in range(5)]
        
        for new_bit in [0, 1]:
            target_bits = source_bits[1:] + [new_bit]
            target = sum(bit << (4 - i) for i, bit in enumerate(target_bits))
            
            output = rule_table[source]
            
            edge_list.append([source, target])
            edge_attrs.append(output)
    
    x = torch.arange(32, dtype=torch.float).unsqueeze(1)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(1)
    
    y = torch.tensor([rule_id], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def batch_debruijn_graphs(rule_ids):
    """
    Convert multiple rule IDs to a batch of graphs.
    
    Args:
        rule_ids: List or array of rule IDs
        
    Returns:
        List of PyG Data objects
    """
    return [rule_id_to_debruijn_graph(rid) for rid in rule_ids]