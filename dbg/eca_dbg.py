import torch
from torch_geometric.data import Data

def eca_rule_table(rule_number: int):
    """
    ECA 输出表：邻域按 000..111 编码为 0..7
    output = (rule_number >> code) & 1
    """
    return [(rule_number >> i) & 1 for i in range(8)]

def eca_de_bruijn_data(rule_number: int) -> Data:
    """
    De Bruijn graph for ECA (r=1,k=2):
      Nodes: 00,01,10,11  (4 nodes)
      Edges: u=b0b1 append a -> v=b1a   (8 edges)
      Edge_attr: rule output for neighborhood b0 b1 a
    """
    table = eca_rule_table(rule_number)

    # Node features: two bits of context (00,01,10,11)
    x = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float)

    src, dst, out = [], [], []
    for u in range(4):
        b0 = (u >> 1) & 1
        b1 = u & 1
        for a in (0, 1):
            code = (b0 << 2) | (b1 << 1) | a   # 0..7 for 000..111
            v = (b1 << 1) | a                  # shift + append
            src.append(u)
            dst.append(v)
            out.append(table[code])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(out, dtype=torch.long).view(-1, 1)  # [E,1]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.y = torch.tensor([rule_number], dtype=torch.long)  # graph label
    data.rule = rule_number
    data.graph_type = "de_bruijn"
    return data