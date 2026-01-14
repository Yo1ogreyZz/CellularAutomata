import torch
from torch_geometric.data import Data

def eca_rule_table(rule_number: int):
    return [(rule_number >> i) & 1 for i in range(8)]

def eca_dbg_edges(rule_number: int):
    """
    dBG edges for ECA:
    returns list of (src, dst, out_label) with src,dst in {0..3}, out_label in {0,1}
    """
    table = eca_rule_table(rule_number)
    edges = []
    for u in range(4):
        b0 = (u >> 1) & 1
        b1 = u & 1
        for a in (0, 1):
            code = (b0 << 2) | (b1 << 1) | a
            out = table[code]
            v = (b1 << 1) | a
            edges.append((u, v, out))
    return edges

def eca_subset_graph_data(rule_number: int) -> Data:
    """
    Subset graph for ECA:
      Nodes: all subsets of dBG vertices => 2^4 = 16 nodes, indexed by bitmask 0..15
      For each subset S and label lab in {0,1}, deterministic transition to S'
      Edge_attr: lab (0/1)
      Node features: indicator vector of size 4 (membership of {00,01,10,11})
    """
    dbg_edges = eca_dbg_edges(rule_number)

    # out_map[(src, lab)] = list of dst
    out_map = {(u, lab): [] for u in range(4) for lab in (0, 1)}
    for u, v, lab in dbg_edges:
        out_map[(u, lab)].append(v)

    num_nodes = 16  # 2^4

    # Node feature: 4-dim membership vector
    x = torch.zeros((num_nodes, 4), dtype=torch.float)
    for mask in range(num_nodes):
        for v in range(4):
            if (mask >> v) & 1:
                x[mask, v] = 1.0

    src_list, dst_list, lab_list = [], [], []

    for S in range(num_nodes):
        # elements in subset S
        elems = [v for v in range(4) if (S >> v) & 1]

        for lab in (0, 1):
            nxt = 0
            for v in elems:
                for v2 in out_map[(v, lab)]:
                    nxt |= (1 << v2)  # union
            src_list.append(S)
            dst_list.append(nxt)
            lab_list.append(lab)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)  # [2, 32]
    edge_attr = torch.tensor(lab_list, dtype=torch.long).view(-1, 1)   # [32, 1]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.y = torch.tensor([rule_number], dtype=torch.long)
    data.rule = rule_number
    data.graph_type = "subset_graph"
    return data