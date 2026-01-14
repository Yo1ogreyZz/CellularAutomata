import torch
from torch_geometric.data import Data


def rule_to_bits(rule_number: int) -> torch.Tensor:
    """
    Return rule truth table bits for neighborhoods 000..111 as shape [8].
    bits[code] = output, where code = (l<<2)|(c<<1)|r.
    """
    bits = torch.zeros(8, dtype=torch.long)
    for code in range(8):
        bits[code] = (rule_number >> code) & 1
    return bits


def rule_influence(bits: torch.Tensor) -> torch.Tensor:
    """
    Compute whether output depends on each position (L,C,R).
    influence[p] = 1 if there exists a neighborhood where flipping that bit changes output.
    Order: [L, C, R] (3,)
    """
    infl = torch.zeros(3, dtype=torch.long)
    # enumerate all 8 neighborhoods
    for code in range(8):
        l = (code >> 2) & 1
        c = (code >> 1) & 1
        r = (code >> 0) & 1
        out = bits[code].item()

        # flip L
        codeL = ((1 - l) << 2) | (c << 1) | r
        if bits[codeL].item() != out:
            infl[0] = 1
        # flip C
        codeC = (l << 2) | ((1 - c) << 1) | r
        if bits[codeC].item() != out:
            infl[1] = 1
        # flip R
        codeR = (l << 2) | (c << 1) | (1 - r)
        if bits[codeR].item() != out:
            infl[2] = 1

        if infl.sum().item() == 3:
            break
    return infl


def eca_step(state: torch.Tensor, rule_bits: torch.Tensor, r: int = 1, periodic: bool = True) -> torch.Tensor:
    """
    One synchronous update for ECA (r=1, binary). state shape [W], values {0,1}.
    Uses standard convention: left = i-1, center=i, right=i+1 (with periodic wrap if enabled).
    """
    assert r == 1, "This implementation assumes r=1 (ECA)."
    W = state.numel()
    nxt = torch.zeros_like(state)

    for i in range(W):
        if periodic:
            left  = state[(i - 1) % W].item()
            cen   = state[i].item()
            right = state[(i + 1) % W].item()
        else:
            left  = state[i - 1].item() if i - 1 >= 0 else 0
            cen   = state[i].item()
            right = state[i + 1].item() if i + 1 < W else 0

        code = (left << 2) | (cen << 1) | right
        nxt[i] = rule_bits[code]
    return nxt


def simulate_eca(rule_number: int, W: int, T: int, periodic: bool,
                 num_seeds: int = 4, base_seed: int = 0) -> torch.Tensor:
    """
    Simulate num_seeds trajectories for T steps.
    Returns states shape [num_seeds, T+1, W] (includes t=0).
    """
    rule_bits = rule_to_bits(rule_number)

    gen = torch.Generator()
    gen.manual_seed(base_seed)

    states = torch.zeros((num_seeds, T + 1, W), dtype=torch.long)
    for k in range(num_seeds):
        init = torch.randint(0, 2, (W,), generator=gen, dtype=torch.long)
        states[k, 0] = init
        cur = init
        for t in range(T):
            cur = eca_step(cur, rule_bits, r=1, periodic=periodic)
            states[k, t + 1] = cur
    return states


def dependency_graph_data_with_rule(
    rule_number: int,
    W: int = 10,
    T: int = 8,
    r: int = 1,
    periodic: bool = True,
    num_seeds: int = 4,
    base_seed: int = 0
) -> Data:
    """
    Dependency graph over spacetime nodes (i,t) plus rule-dependent node/edge features.

    Nodes: (i,t) for i=0..W-1, t=0..T. Total W*(T+1).
    Edges: (j,t) -> (i,t+1) for j in [i-r..i+r] with wrap if periodic.

    Node features x: [i_norm, t_norm, s1, s2, ..., sK]
      where sk = state value at (i,t) for trajectory k.

    Edge features edge_attr: [dx_norm, influence_dx]
      dx_norm in {-1,0,1} for r=1 (stored as float), influence_dx is 0/1 computed from rule table.

    Graph attributes:
      - y: rule_number
      - rule_bits: (8,)
      - influence: (3,) for L,C,R
    """
    assert r == 1, "This version is for ECA (r=1)."
    rule_bits = rule_to_bits(rule_number)
    infl = rule_influence(rule_bits)  # [L,C,R]

    # simulate trajectories to inject rule info into nodes
    traj = simulate_eca(rule_number, W=W, T=T, periodic=periodic,
                        num_seeds=num_seeds, base_seed=base_seed)  # [K, T+1, W]

    num_nodes = W * (T + 1)

    def nid(i: int, t: int) -> int:
        return t * W + i

    # Node features: (i_norm, t_norm, states...)
    x = torch.zeros((num_nodes, 2 + num_seeds), dtype=torch.float)
    for t in range(T + 1):
        for i in range(W):
            idx = nid(i, t)
            x[idx, 0] = 0.0 if W == 1 else float(i) / float(W - 1)
            x[idx, 1] = 0.0 if T == 0 else float(t) / float(T)
            # inject trajectory states
            # traj[k, t, i] is 0/1 -> float
            x[idx, 2:] = traj[:, t, i].to(torch.float)

    # Edges + edge_attr
    src_list, dst_list, attr_list = [], [], []
    for t in range(T):
        for i in range(W):
            dst = nid(i, t + 1)
            for dx in (-1, 0, 1):
                j = i + dx
                if periodic:
                    j = j % W
                else:
                    if j < 0 or j >= W:
                        continue
                src = nid(j, t)
                src_list.append(src)
                dst_list.append(dst)

                # influence according to dx position: dx=-1 -> L, dx=0 -> C, dx=+1 -> R
                infl_bit = infl[0].item() if dx == -1 else (infl[1].item() if dx == 0 else infl[2].item())
                attr_list.append([float(dx), float(infl_bit)])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(attr_list, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([rule_number], dtype=torch.long)
    )

    data.rule = int(rule_number)
    data.rule_bits = rule_bits.clone()          # (8,)
    data.influence = infl.clone()               # (3,)
    data.graph_type = "dependency_rule"
    data.W = W
    data.T = T
    data.r = r
    data.periodic = bool(periodic)
    data.num_seeds = int(num_seeds)
    data.base_seed = int(base_seed)
    return data