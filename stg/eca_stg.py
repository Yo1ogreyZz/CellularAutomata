import torch
from torch_geometric.data import Data

def eca_output(rule_number: int, left: int, center: int, right: int) -> int:
    """ECA output for neighborhood (left, center, right) bits."""
    code = (left << 2) | (center << 1) | right  # 0..7
    return (rule_number >> code) & 1

def stg_next_state(rule_number: int, state: int, N: int) -> int:
    """
    Compute next global state for an N-cell ring (periodic boundary).
    state is an integer 0..(2^N - 1), bit i is cell i.
    """
    nxt = 0
    for i in range(N):
        center = (state >> i) & 1
        left   = (state >> ((i - 1) % N)) & 1   # i-1 (left neighbor)
        right  = (state >> ((i + 1) % N)) & 1   # i+1 (right neighbor)
        out = eca_output(rule_number, left, center, right)
        nxt |= (out << i)
    return nxt

def eca_stg_data(rule_number: int, N: int = 8) -> Data:
    """
    STG for ECA on N-cell ring:
      Nodes: all global configs => 2^N nodes
      Edges: state -> next_state (out-degree=1)
      Node features: N-bit vector of the configuration
      Edge_attr: optional (here none, but we can store 0)
    """
    num_nodes = 1 << N

    # Node features: binary config vector (num_nodes, N)
    x = torch.zeros((num_nodes, N), dtype=torch.float)
    for s in range(num_nodes):
        for i in range(N):
            x[s, i] = (s >> i) & 1

    # Edges: deterministic map
    src = torch.arange(num_nodes, dtype=torch.long)
    dst = torch.empty(num_nodes, dtype=torch.long)
    for s in range(num_nodes):
        dst[s] = stg_next_state(rule_number, s, N)

    edge_index = torch.stack([src, dst], dim=0)  # [2, num_nodes]

    data = Data(x=x, edge_index=edge_index)
    data.y = torch.tensor([rule_number], dtype=torch.long)
    data.rule = rule_number
    data.graph_type = "stg"
    data.N = N
    return data