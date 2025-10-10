# ca_toolbox.py
# Reproduces methods surveyed in Vispoel, Daly & Baetens (2022), Physica D 432:133074.
# Sections referenced inline (e.g., "Sec. 4.2") match the paper; figures noted where helpful.
# Implementations focus on 1D, binary CA (ECA and radius r>=1), periodic boundary,
# as that’s the core worked example in the review. Many functions generalize to r>1.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Iterable, Optional, Callable
import numpy as np
import itertools as it
from collections import Counter, defaultdict, deque
import math, zlib

__all__ = [
    "CA1D", "eca_rule_bits",
    "seed_simple", "seed_finite_block", "seed_random",
    "langton_lambda", "mean_field_cluster", "mu_sensitivity", "theta_obstruction",
    "z_reverse_parameter", "kolmogorov_rule_complexity",
    "mean_field_map_fun", "local_structure_step_order2", "two_point_corr",
    "difference_pattern_step", "sheveshevsky_velocities", "lyapunov_profiles",
    "compressibility_ratio", "power_spectrum",
    "next_state_int", "attractor_basins", "d_spectrum_tau",
    "de_bruijn_graph_labels", "ndfa_for_outputs", "subset_construction_dfa",
    "renyi_entropy_from_counts", "renyi_spatial_set_measure", "renyi_temporal_entropy"
]

# --------------------------- Core 1D CA engine ---------------------------

def eca_rule_bits(rule: int) -> np.ndarray:
    """
    ECA numbering (Wolfram): index = b_{i-1}<<2 | b_i<<1 | b_{i+1}.
    Bit index 7..0 correspond to neighborhoods 111,110,...,000.
    Returns uint8[8] with outputs for keys 0..7 (000..111).
    """
    bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)
    # bits[i] is output for neighborhood int i (000..111).
    return bits

@dataclass
class CA1D:
    """1D binary CA with radius r (neighborhood size n=2r+1), periodic boundary."""
    r: int = 1
    rule_table: Dict[int, int] = None  # map neighborhood int -> output int (0/1)
    N: int = 128  # lattice size
    
    @staticmethod
    def from_eca(rule_number: int, N: int=128) -> "CA1D":
        bits = eca_rule_bits(rule_number)
        # Map 3-bit neighborhood integer -> output
        tbl = {i: int(bits[i]) for i in range(8)}
        return CA1D(r=1, rule_table=tbl, N=N)
    
    @property
    def n(self) -> int:
        return 2*self.r + 1
    
    def step(self, x: np.ndarray) -> np.ndarray:
        """Apply one synchronous update (periodic BC). x shape (N,), dtype uint8."""
        N, r, n = self.N, self.r, self.n
        y = np.empty_like(x)
        # roll-based window to int neighborhood
        # build neighborhood int with LSB = rightmost neighbor
        # For r=1: idx = (left<<2)|(center<<1)|right
        nb_int = np.zeros(N, dtype=np.int32)
        for k, w in enumerate(range(-r, r+1)[::-1]):  # MSB first
            nb_int = (nb_int << 1) | np.roll(x, -w)  # -w because left shift in index
        # Dispatch via rule_table (for general r, rule_table must include all keys)
        vec = np.vectorize(lambda nb: self.rule_table.get(nb, 0))
        y = vec(nb_int).astype(np.uint8)
        return y
    
    def evolve(self, x0: np.ndarray, T: int) -> np.ndarray:
        """Return array of shape (T+1, N) including x0."""
        xs = np.empty((T+1, self.N), dtype=np.uint8)
        xs[0] = x0
        for t in range(T):
            xs[t+1] = self.step(xs[t])
        return xs

# --------------------------- Seeds (Sec. 5.2) ---------------------------

def seed_simple(N: int, val: int=1) -> np.ndarray:
    x = np.zeros(N, dtype=np.uint8)
    x[N//2] = val
    return x

def seed_finite_block(N: int, length: int=10, val: int=1) -> np.ndarray:
    x = np.zeros(N, dtype=np.uint8)
    s = (N - length)//2
    x[s:s+length] = val
    return x

def seed_random(N: int, p: float=0.5) -> np.ndarray:
    return (np.random.rand(N) < p).astype(np.uint8)

# --------------------------- Rule-table parameters (Sec. 4) ---------------------------

def langton_lambda(rule_table: Dict[int,int], k: int=2, q: int=0, n: int=3) -> float:
    """Langton's parameter λ = (k^n - n_q) / k^n (Sec. 4.2)."""
    K = k**n
    nq = sum(1 for nb in range(K) if rule_table.get(nb, 0) == q)
    return (K - nq) / K

def mean_field_cluster(rule_table: Dict[int,int], k: int=2, q: int=0, n: int=3) -> List[int]:
    """
    Mean-field parameters {M_i} (Sec. 4.3): for i=0..n, count neighborhoods with i non-q values
    that map to non-q. For binary q=0 this is "count #ones in neighborhood".
    """
    Ms = [0]*(n+1)
    for nb in range(k**n):
        out = rule_table.get(nb, 0)
        if out != q:
            # count i = # non-q in neighborhood
            if k == 2 and q == 0:
                i = int(bin(nb).count("1"))
            else:
                # general k: need digits in base-k; here we stick to binary focus
                raise NotImplementedError("General k not implemented.")
            Ms[i] += 1
    return Ms

def mu_sensitivity(rule_table: Dict[int,int], n: int=3) -> float:
    """
    μ-sensitivity (Sec. 4.5, Eq. 8–9): average #output flips when flipping one neighborhood bit.
    Returns μ in [0, 1/2] for binary.
    """
    K = 2**n
    total = 0
    for nb in range(K):
        y = rule_table.get(nb, 0)
        for pos in range(n):
            flipped = nb ^ (1 << pos)  # flip bit at position 'pos' (LSB=bit0=right neighbor)
            y2 = rule_table.get(flipped, 0)
            total += 1 if (y != y2) else 0
    return total / (n*K)

def theta_obstruction(rule_table: Dict[int,int], n: int=3) -> float:
    """
    Obstruction parameter Θ (Sec. 4.6, Eq. 10–11): fraction of (i,j) where additivity fails:
      x_i XOR x_j XOR x_{i⊕j} == 1.
    Θ=0 iff additive; reported values for ECA: 0, 9/32, 5/16, 3/8, 21/32.
    """
    K = 2**n
    xi = np.array([rule_table.get(i, 0) for i in range(K)], dtype=np.uint8)
    s = 0
    for i in range(K):
        for j in range(K):
            s += (xi[i] ^ xi[j] ^ xi[i ^ j])
    return s / (2 * K) / K  # 1/(2*2^n) * average over j; equivalent to Eq. (11)

def z_reverse_parameter(rule_table: Dict[int,int], r: int=1, N: int=64, trials: int=64, p: float=0.5) -> float:
    """
    Wuensche's Z-reverse (Sec. 4.4): Monte Carlo estimate of the probability that
    the next predecessor bit is *deterministic* in a sequential reverse construction.
    We simulate the subset-of-states propagation over last (n-1) bits (De Bruijn idea),
    and, at each step, check if ONLY one choice of the 'next bit' is possible across
    the current boundary-states set. Average this indicator over positions & trials.
    """
    n = 2*r+1
    # Precompute transition relation for predecessor builder.
    # State = last (n-1) predecessor bits (as int). Given next bit b in {0,1}, we can
    # form neighborhood (state<<1 | b) and check if rule output equals current-row bit.
    det_count = 0
    total = 0
    for _ in range(trials):
        y = (np.random.rand(N) < p).astype(np.uint8)  # current row
        # Initialize set of possible states for first (n-1) bits: all 2^(n-1) states
        states = set(range(2**(n-1)))
        for t in range(N):  # left-to-right fill
            # For each candidate state s, determine allowed 'next bit' choices b in {0,1}
            allow_b = {0: False, 1: False}
            new_states_by_b = {0: set(), 1: set()}
            for s in states:
                for b in (0,1):
                    nb = ((s << 1) | b) & ((1<<n) - 1)
                    if rule_table.get(nb, 0) == int(y[t]):
                        allow_b[b] = True
                        new_states_by_b[b].add(nb & ((1<<(n-1))-1))  # next state = last (n-1) bits
            if allow_b[0] ^ allow_b[1]:  # exactly one option holds
                det_count += 1
            # Advance states union; if both allowed, union; if none allowed, break early.
            next_states = set()
            if allow_b[0]: next_states |= new_states_by_b[0]
            if allow_b[1]: next_states |= new_states_by_b[1]
            if not next_states:
                # No predecessor consistent; in Wuensche's alg one would backtrack, but
                # for Z we just continue counting positions processed so far
                pass
            states = next_states if next_states else set(range(2**(n-1)))
            total += 1
    return det_count / total if total else 0.0

def kolmogorov_rule_complexity(rule_table: Dict[int,int], n: int=3) -> int:
    """
    Rule-table 'complexity' upper-bound via DEFLATE (Sec. 4.7): compress length of the
    rule table serialized as bytes (neighborhood order 0..2^n-1).
    """
    K = 2**n
    b = bytes([rule_table.get(i, 0) for i in range(K)])
    return len(zlib.compress(b, level=9))

# --------------------------- Local analyses (Sec. 5) ---------------------------

def mean_field_map_fun(rule_table: Dict[int,int], n: int=3) -> Callable[[float], float]:
    """
    Return f(ρ) = sum_{neigh with out=1} ρ^{#1} (1-ρ)^{#0}, the MF polynomial (Sec. 5.3, Eq. 21).
    """
    outs = [(nb, rule_table.get(nb,0)) for nb in range(2**n)]
    k1 = [nb for nb,y in outs if y==1]
    pow_table = [(bin(nb).count("1"), n - bin(nb).count("1")) for nb in k1]
    def f(rho: float) -> float:
        s = 0.0
        for a,b in pow_table:
            s += (rho**a) * ((1.0 - rho)**b)
        return s
    return f

def local_structure_step_order2(rule_table: Dict[int,int], rho2: Dict[Tuple[int,int], float]) -> Dict[Tuple[int,int], float]:
    """
    Local structure theory, order-2 (Sec. 5.4, Eq. 22–23) for r=1 (n=3):
      ρ_{ab}(t+1) = sum over length-4 blocks v0..v3 where φ(v0 v1 v2)=a AND φ(v1 v2 v3)=b
      of p(v0..v3) under the order-2 approximation
      p(v0..v3) = ρ_{v0 v1} ρ_{v1 v2} ρ_{v2 v3} / (ρ_{v1} ρ_{v2})
    """
    def phi3(v0,v1,v2):
        nb = (v0<<2)|(v1<<1)|v2
        return rule_table.get(nb,0)
    # single-site densities ρ_v = sum_u ρ_{uv} = sum_u ρ_{vu}
    rhov = {0: rho2[(0,0)] + rho2[(1,0)], 1: rho2[(0,1)] + rho2[(1,1)]}
    out = {(a,b):0.0 for a in (0,1) for b in (0,1)}
    for v0,v1,v2,v3 in it.product((0,1), repeat=4):
        p = 0.0
        denom = rhov[v1]*rhov[v2]
        if denom > 0:
            p = (rho2[(v0,v1)] * rho2[(v1,v2)] * rho2[(v2,v3)]) / denom
        a = phi3(v0,v1,v2)
        b = phi3(v1,v2,v3)
        out[(a,b)] += p
    # normalize against numeric drift
    s = sum(out.values())
    if s>0:
        for k in out: out[k] /= s
    return out

def two_point_corr(x: np.ndarray, r: int) -> float:
    """
    Two-point correlation C^(2)(r) (Sec. 5.5, Eq. 25) using s^(b) ∈ {+1,-1}.
    """
    s = np.where(x==1, 1.0, -1.0)
    s_shift = np.roll(s, -r)
    return float(np.mean(s*s_shift) - np.mean(s)*np.mean(s_shift))

def difference_pattern_step(ca: CA1D, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """
    Δs(t+1) = J(s(t),t) * Δs(t) over GF(2) (Sec. 5.6, Eq. 26–28).
    We implement J on-the-fly: output bit i only depends on window [i-r..i+r].
    """
    r = ca.r
    N = ca.N
    y = np.zeros_like(dx)
    # For each site i, perturb each site j in window; if output flips, J[i,j]=1
    # Accumulate y[i] = sum_j J[i,j] * dx[j] mod 2
    for i in range(N):
        # current neighborhood
        nb = 0
        for w in range(-r, r+1):
            nb = (nb<<1) | int(x[(i+w) % N])
        base = ca.rule_table.get(nb,0)
        acc = 0
        for off, w in enumerate(range(-r, r+1)):
            j = (i+w) % N
            if dx[j]==0: 
                continue
            nb_flip = nb ^ (1 << (r - w))  # bit index: MSB at w=-r
            fl = ca.rule_table.get(nb_flip,0)
            if fl != base:
                acc ^= 1
        y[i] = acc
    return y

def sheveshevsky_velocities(xs: np.ndarray, dXs: np.ndarray) -> Tuple[float,float]:
    """
    Approximate left/right cone velocities from difference pattern stack (Sec. 5.7.1).
    Returns (v_left, v_right) as sites per step divided by total steps.
    """
    T, N = dXs.shape[0]-1, dXs.shape[1]
    if T == 0: return (0.0, 0.0)
    leftmost = []
    rightmost = []
    for t in range(1, T+1):
        idx = np.where(dXs[t]>0)[0]
        if idx.size == 0:
            leftmost.append(0)
            rightmost.append(0)
        else:
            leftmost.append((idx.min() - idx[0]) % N)  # arbitrary reference
            rightmost.append((idx.max() - idx[0]) % N)
    return ( (min(leftmost) if leftmost else 0)/T, (max(rightmost) if rightmost else 0)/T )

def lyapunov_profiles(ca: CA1D, x0: np.ndarray, site0: int=0, T: int=256) -> Tuple[np.ndarray, float]:
    """
    Bagnoli et al. Lyapunov profiles (Sec. 5.7.2): evolve both x(t) and integer 'damage routes' n(t),
    where n(t+1) = J(t) @ n(t) (over Z, not mod 2). MLE = max_i (1/T * log n_i(T)).
    Returns (Λ_i(T), MLE).
    """
    N = ca.N
    x = x0.copy()
    # n(0): one route at single site0
    n = np.zeros(N, dtype=np.int64); n[site0] = 1
    for _ in range(T):
        # Build J action via difference of unit vectors: J * e_j is the next damage from a unit flip at j
        # But to avoid NxN, we can reuse the GF(2) version multiple times … here we compute column-wise sparsely:
        x_next = ca.step(x)
        n_next = np.zeros_like(n)
        # For each site j with n[j] routes, propagate those routes through J's column j
        # To get J[:,j], compute difference pattern with dx having 1 at j only:
        for j, cnt in enumerate(n):
            if cnt == 0: continue
            dx = np.zeros(N, dtype=np.uint8); dx[j]=1
            col = difference_pattern_step(ca, x, dx)  # GF(2) pattern of which outputs flip if input j flips
            n_next += cnt * col.astype(np.int64)
        x = x_next
        n = n_next
    with np.errstate(divide='ignore'):
        lam = np.where(n>0, np.log(n)/T, -np.inf)
    mle = float(np.max(lam))
    return lam, mle

def compressibility_ratio(configs: np.ndarray) -> float:
    """
    γ_CR(s(·,t)) (Sec. 5.8): compress concat of rows with DEFLATE; return compressed_len / raw_len.
    """
    T, N = configs.shape
    # pack bits to bytes
    bits = configs.astype(np.uint8).ravel().tolist()
    # group per 8
    by = bytearray()
    for i in range(0, len(bits), 8):
        b = 0
        for k in range(8):
            if i+k < len(bits):
                b |= (bits[i+k] & 1) << k
        by.append(b)
    raw = bytes(by)
    comp = zlib.compress(raw, level=9)
    return len(comp) / max(1, len(raw))

def power_spectrum(xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Power spectrum S(f) averaged over sites (Sec. 5.9; Eq. 34–35). xs shape (T, N).
    Returns (freqs, S), where freqs are 0..T//2.
    """
    T, N = xs.shape
    # Detrend by subtracting mean per site (optional)
    X = xs - xs.mean(axis=0, keepdims=True)
    # FFT across time for each site
    F = np.fft.rfft(X, axis=0) / T
    S = (np.abs(F)**2).mean(axis=1)  # average over sites
    freqs = np.arange(S.shape[0], dtype=float)
    return freqs, S

# --------------------------- Global (finite-CA) analyses (Sec. 6) ---------------------------

def next_state_int(ca: CA1D, s: int) -> int:
    """Compute next state (as int) from current state int (finite CA, small N)."""
    N = ca.N
    x = np.array([(s >> i) & 1 for i in range(N)], dtype=np.uint8)
    y = ca.step(x)
    out = 0
    for i in range(N):
        if y[i]: out |= (1 << i)
    return out

def attractor_basins(ca: CA1D, Nmax: int=20) -> Dict[str, object]:
    """
    Enumerate full state graph for N<=20; compute basins, cycles, in-degree, leaf-density & G-density (Sec. 6.1.2).
    Returns summary dict.
    """
    assert ca.N <= Nmax, "State-space too large; choose smaller N."
    total = 1 << ca.N
    succ = np.zeros(total, dtype=np.uint32)
    indeg = np.zeros(total, dtype=np.uint32)
    for s in range(total):
        t = next_state_int(ca, s)
        succ[s] = t
        indeg[t] += 1
    # Garden of Eden = states with indeg 0
    goe = int((indeg == 0).sum())
    # Find cycles and basins
    visited = np.zeros(total, dtype=bool)
    cycles = []
    basins = []
    for s in range(total):
        if visited[s]: continue
        # tortoise-hare to find cycle from s
        tort = s; hare = succ[succ[s]]
        while tort != hare:
            tort = succ[tort]; hare = succ[succ[hare]]
        # Find cycle start
        mu = 0; tort = s
        while tort != hare:
            tort = succ[tort]; hare = succ[hare]; mu += 1
        # Find cycle length
        lam = 1; hare = succ[tort]
        while tort != hare:
            hare = succ[hare]; lam += 1
        # Collect cycle nodes
        cycle = []
        cur = tort
        for _ in range(lam):
            cycle.append(cur); visited[cur]=True; cur = succ[cur]
        cycles.append(cycle)
        # BFS backwards to collect basin nodes
        basin_nodes = set(cycle)
        dq = deque(cycle)
        preds = defaultdict(list)
        for u in range(total):
            preds[succ[u]].append(u)
        while dq:
            v = dq.popleft()
            for u in preds[v]:
                if u not in basin_nodes:
                    basin_nodes.add(u); dq.append(u)
        basins.append(basin_nodes)
        for u in basin_nodes: visited[u] = True
    leaf_density = float((indeg == 0).sum()) / total
    return {
        "goe_count": goe,
        "g_density": goe/total,
        "indegree": indeg,
        "cycles": cycles,
        "basins": basins,
        "leaf_density": leaf_density,
        "succ": succ
    }

def d_spectrum_tau(cycle_states: List[int], N: int) -> int:
    """
    D-spectrum τ (Sec. 6.4, Eq. 43): number of distinct densities #1(x) in the periodic part P.
    """
    dens = set()
    for s in cycle_states:
        dens.add(bin(s).count("1"))
    return len(dens)

# ---- Formal languages via De Bruijn → NDFA → DFA (Ω(t), Sec. 6.3.2; Fig. 19–21)

def de_bruijn_graph_labels(rule_table: Dict[int,int], r: int=1) -> Dict[Tuple[int,int], List[int]]:
    """
    Build De Bruijn graph (nodes = (n-1)-bit ints), edges u->v labeled by output φ(neigh) (Sec. 6.3.2).
    Returns dict[(u,v)] -> [labels], labels ∈ {0,1}.
    """
    n = 2*r+1
    nodes = range(2**(n-1))
    edges = defaultdict(list)
    for u in nodes:
        for b in (0,1):
            nb = ((u << 1) | b) & ((1<<n)-1)
            lab = rule_table.get(nb, 0)
            v = nb & ((1<<(n-1))-1)
            edges[(u,v)].append(lab)
    return edges

def ndfa_for_outputs(edges: Dict[Tuple[int,int], List[int]]) -> Dict[int, Dict[int, List[int]]]:
    """
    NDFA over alphabet {0,1}: state space = De Bruijn nodes; transitions follow edges labeled by symbol.
    Returns: delta_ndfa[state][symbol] -> list(next_states)
    """
    nodes = set([u for (u,_) in edges] + [v for (_,v) in edges])
    delta = {s:{0:[],1:[]} for s in nodes}
    for (u,v), labs in edges.items():
        for a in labs:
            delta[u][a].append(v)
    return delta

def subset_construction_dfa(delta_ndfa: Dict[int, Dict[int, List[int]]], start_states: Optional[Iterable[int]]=None) -> Tuple[Dict[frozenset, Dict[int, frozenset]], frozenset]:
    """
    Subset construction: build DFA whose states are frozensets of NDFA states.
    Start states: all De Bruijn nodes (Ω(1) is generated from all initial configs).
    """
    if start_states is None:
        start_states = list(delta_ndfa.keys())
    q0 = frozenset(start_states)
    Q = {q0}
    work = [q0]
    delta_dfa = {}
    while work:
        S = work.pop()
        delta_dfa[S] = {}
        for a in (0,1):
            U = set()
            for s in S:
                U.update(delta_ndfa[s][a])
            U = frozenset(U)
            delta_dfa[S][a] = U
            if U not in Q:
                Q.add(U); work.append(U)
    return delta_dfa, q0

# ---- Rényi entropies (Sec. 6.5; Eq. 45–48)

def renyi_entropy_from_counts(counts: Counter, base_k: int, alpha: float) -> float:
    """
    Generic Rényi entropy from counts of discrete outcomes (configurations or blocks).
    Returns entropy per symbol (divide by length outside if needed).
    """
    total = sum(counts.values())
    if total == 0: return 0.0
    ps = np.array(list(counts.values()), dtype=float)/total
    if alpha == 1.0:
        H = -np.sum(ps * np.log(ps)) / np.log(base_k)
    elif alpha == 0.0:
        H = np.log(len(ps)) / np.log(base_k)
    else:
        H = (1.0/(1.0 - alpha)) * np.log(np.sum(ps**alpha)) / np.log(base_k)
    return float(H)

def renyi_spatial_set_measure(xs: np.ndarray, alpha: float=1.0, block: int=1, k: int=2) -> float:
    """
    Block Rényi entropy S_α(|B|, t) (Eq. 46) for a single configuration xs[t] using blocks of length 'block'.
    Returns per-site entropy by dividing by |B|.
    """
    x = xs.astype(np.uint8)
    N = x.size
    blks = []
    for i in range(N):
        v = 0
        for b in range(block):
            v = (v<<1) | int(x[(i+b) % N])
        blks.append(v)
    H = renyi_entropy_from_counts(Counter(blks), base_k=k, alpha=alpha)
    return H / block

def renyi_temporal_entropy(xs: np.ndarray, site: int, alpha: float=1.0, Tblock: int=8, k: int=2) -> float:
    """
    Temporal Rényi entropy at a site (Eq. 48) using blocks of length Tblock over time.
    Returns H/Tblock.
    """
    s = xs[:, site].astype(np.uint8)
    T = s.size
    seqs = []
    for t in range(T):
        v = 0
        for b in range(Tblock):
            v = (v<<1) | int(s[(t+b) % T])
        seqs.append(v)
    H = renyi_entropy_from_counts(Counter(seqs), base_k=k, alpha=alpha)
    return H / Tblock
