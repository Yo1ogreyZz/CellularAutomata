import numpy as np
import torch
from torch_geometric.data import Data


def rule_id_to_simple_graph(rule_id):
    """
    Convert a 32-bit CA rule into a simple 2-node De Bruijn graph.
    
    This is the "symbol graph" representation where:
    - 2 nodes: representing cell states 0 and 1
    - 32 edges: one for each possible 5-bit neighborhood configuration
    - Edge attributes encode: [b4, b3, b2, b1, b0, output]
    
    Edge construction:
    - For each 5-bit neighborhood (b4, b3, b2, b1, b0)
    - Center cell is b2 (source node)
    - Apply CA rule to get output (target node)
    - Edge goes from b2 to output
    - Edge attributes store the full neighborhood info
    
    Args:
        rule_id: Integer in [0, 2^32-1] encoding the transition rule
        
    Returns:
        PyG Data object with:
            - x: node features (2 x 1, values [0.0] and [1.0])
            - edge_index: 32 edges connecting nodes 0 and 1
            - edge_attr: [5-bit neighborhood, 1-bit output] for each edge (32 x 6)
            - y: global graph label (the rule_id itself)
    """
    rule_id = int(rule_id)
    
    rule_table = np.array([(rule_id >> i) & 1 for i in range(32)], dtype=np.int64)
    
    edge_list = []
    edge_attrs = []
    
    for neighborhood in range(32):
        bits = [(neighborhood >> (4 - i)) & 1 for i in range(5)]
        
        center_bit = bits[2]
        output = rule_table[neighborhood]
        
        source_node = center_bit
        target_node = output
        
        edge_list.append([source_node, target_node])
        
        edge_attr = bits + [output]
        edge_attrs.append(edge_attr)
    
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    y = torch.tensor([rule_id], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def batch_simple_graphs(rule_ids):
    """
    Convert multiple rule IDs to a batch of simple graphs.
    
    Args:
        rule_ids: List or array of rule IDs
        
    Returns:
        List of PyG Data objects
    """
    return [rule_id_to_simple_graph(rid) for rid in rule_ids]