import torch
from torch_geometric.data import Data

def eca_rule_table(rule_number: int):
    # neighborhood code: 000..111 -> 0..7
    return [(rule_number >> i) & 1 for i in range(8)]

def eca_dbg_edges(rule_number: int):
    """
    Return all dBG edges for ECA:
    edges: (u, v, out_label) where u,v in {0..3}, out_label in {0,1}
    """
    table = eca_rule_table(rule_number)
    edges = []
    for u in range(4):
        b0 = (u >> 1) & 1
        b1 = u & 1
        for a in (0, 1):
            code = (b0 << 2) | (b1 << 1) | a  # neighborhood b0 b1 a
            out = table[code]
            v = (b1 << 1) | a                 # next context
            edges.append((u, v, out))
    return edges

def eca_pair_graph_data(rule_number: int) -> Data:
    """
    Pair graph for ECA:
      Nodes: ordered pairs (u,v), u,v in {0..3} -> 16 nodes
      Edge: (u,v)->(u',v') exists if there are dBG edges u->u' and v->v'
            with the same output label.
      Edge_attr: that output label (0/1)
    """
    # Node features: concatenate context bits of u and v: [u0,u1,v0,v1]
    # Mapping: 0->00, 1->01, 2->10, 3->11
    ctx = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float)  # [4,2]
    x = torch.cat(
        [ctx.repeat_interleave(4, dim=0),  # u part
         ctx.repeat(4, 1)],                # v part
        dim=1
    )  # [16,4]

    dbg_edges = eca_dbg_edges(rule_number)

    # Index outgoing edges by (src, out_label) -> list of dst
    out_map = {(u, lab): [] for u in range(4) for lab in (0,1)}
    for u, v, lab in dbg_edges:
        out_map[(u, lab)].append(v)

    src_list, dst_list, lab_list = [], [], []

    # Pair node id: pid(u,v)=u*4+v
    for u in range(4):
        for v in range(4):
            pid_src = u * 4 + v
            for lab in (0, 1):
                u_nexts = out_map[(u, lab)]
                v_nexts = out_map[(v, lab)]
                if not u_nexts or not v_nexts:
                    continue
                for up in u_nexts:
                    for vp in v_nexts:
                        pid_dst = up * 4 + vp
                        src_list.append(pid_src)
                        dst_list.append(pid_dst)
                        lab_list.append(lab)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(lab_list, dtype=torch.long).view(-1, 1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.y = torch.tensor([rule_number], dtype=torch.long)
    data.rule = rule_number
    data.graph_type = "pair_graph"
    return data