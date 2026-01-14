import torch
from torch_geometric.data import Data


def rule_to_bits(rule_number: int) -> torch.Tensor:
    """Return truth table bits for neighborhoods 000..111 as shape [8]."""
    bits = torch.zeros(8, dtype=torch.long)
    for code in range(8):
        bits[code] = (rule_number >> code) & 1
    return bits


def eca_step(state: torch.Tensor, rule_bits: torch.Tensor, periodic: bool = True) -> torch.Tensor:
    """
    One synchronous ECA step (r=1), standard convention:
    left = i-1, center=i, right=i+1 (periodic wrap if enabled).
    state shape [N], values {0,1}.
    """
    N = state.numel()
    nxt = torch.zeros_like(state)
    for i in range(N):
        if periodic:
            left  = state[(i - 1) % N].item()
            cen   = state[i].item()
            right = state[(i + 1) % N].item()
        else:
            left  = state[i - 1].item() if i - 1 >= 0 else 0
            cen   = state[i].item()
            right = state[i + 1].item() if i + 1 < N else 0

        code = (left << 2) | (cen << 1) | right
        nxt[i] = rule_bits[code]
    return nxt


def simulate_eca(rule_number: int, N: int, T: int, periodic: bool,
                 num_seeds: int = 4, base_seed: int = 0) -> torch.Tensor:
    """
    Simulate num_seeds trajectories for T steps.
    Returns states shape [num_seeds, T+1, N].
    """
    rule_bits = rule_to_bits(rule_number)
    gen = torch.Generator().manual_seed(base_seed)

    states = torch.zeros((num_seeds, T + 1, N), dtype=torch.long)
    for k in range(num_seeds):
        cur = torch.randint(0, 2, (N,), generator=gen, dtype=torch.long)
        states[k, 0] = cur
        for t in range(T):
            cur = eca_step(cur, rule_bits, periodic=periodic)
            states[k, t + 1] = cur
    return states


def cell_lattice_graph_data(
    rule_number: int,
    N: int = 16,
    r: int = 1,
    periodic: bool = True,
    # rule-aug params
    T: int = 8,
    num_seeds: int = 4,
    base_seed: int = 0,
    use_rule_bits: bool = True
) -> Data:
    """
    Cell-Lattice Graph (spatial adjacency):
      Nodes: cells i = 0..N-1
      Edges: i -> j if j = i + dx, dx in [-r..r], dx != 0 (periodic/open)

    Rule-augmented node features:
      x[i] = [i_norm, s1, s2, ..., sK]
      where sk is the state of cell i at a fixed probe time (here we use t=T),
      under K different random initial conditions.

    Graph-level attrs:
      data.rule_bits: (8,) truth table, optional.
    """
    assert r >= 1, "r must be >= 1"
    if r != 1:
        # You can extend simulation to r>1 later; current ECA simulate assumes r=1
        raise ValueError("This rule-aug version currently supports r=1 only.")

    rule_bits = rule_to_bits(rule_number)
    traj = simulate_eca(rule_number, N=N, T=T, periodic=periodic,
                       num_seeds=num_seeds, base_seed=base_seed)  # [K, T+1, N]

    # Choose a probe time to inject into node features.
    # Using final time t=T gives a stronger “rule fingerprint” than t=0.
    probe_t = T
    probe_states = traj[:, probe_t, :]  # [K, N]

    # Node features: [i_norm, probe_states for K seeds]
    x = torch.zeros((N, 1 + num_seeds), dtype=torch.float)
    for i in range(N):
        x[i, 0] = i / max(1, (N - 1))
        x[i, 1:] = probe_states[:, i].to(torch.float)

    # Build lattice edges
    src_list, dst_list = [], []
    for i in range(N):
        for dx in range(-r, r + 1):
            if dx == 0:
                continue
            j = i + dx
            if periodic:
                j %= N
            else:
                if j < 0 or j >= N:
                    continue
            src_list.append(i)
            dst_list.append(j)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([rule_number], dtype=torch.long)
    )
    data.rule = int(rule_number)
    data.graph_type = "cell_lattice_rule"
    data.N = int(N)
    data.r = int(r)
    data.periodic = bool(periodic)

    # rule-aug metadata
    data.T = int(T)
    data.num_seeds = int(num_seeds)
    data.base_seed = int(base_seed)
    data.probe_t = int(probe_t)
    if use_rule_bits:
        data.rule_bits = rule_bits.clone()  # (8,)

    return data