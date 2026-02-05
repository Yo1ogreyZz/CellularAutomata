"""
Radius=2 subset graph dataset builder (PyG InMemoryDataset, standard format) — 5-class fixed

Base:
- radius=2 de Bruijn graph has 16 nodes (all 4-bit contexts).
- Each 5-bit neighborhood induces a directed edge u->v with label out in {0,1}.

Subset graph (reachable subset construction):
- Nodes: reachable subsets of DBG nodes, represented by a 16-bit mask.
- Node feature x: 16-dim indicator vector of the subset (float32).
- Edges: for each subset node S, add transitions for symbol a in {0,1}:
    S --a--> δ(S,a), where δ(S,a)= union over u in S of N_a(u).
  Also optionally add reverse edges (dir=1) for message passing.
- Edge attr: [symbol, dir]
    symbol=0/1, dir=0 forward, dir=1 reverse

Fixed label mapping (5 classes):
  0 -> Class1-Homogeneous
  1 -> Class2-Propagate
  2 -> Class2-Stable
  3 -> Class3-Chaotic
  4 -> Class4-Complex
"""

import os
import glob
import json
import argparse
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data, InMemoryDataset


# -------------------------------------------------
# Fixed 5-class mapping (IMPORTANT)
# -------------------------------------------------
CLASSES_5 = [
    "Class1-Homogeneous",
    "Class2-Propagate",
    "Class2-Stable",
    "Class3-Chaotic",
    "Class4-Complex",
]
LABEL_MAP_5 = {c: i for i, c in enumerate(CLASSES_5)}


# -------------------------------------------------
# CA utilities (radius = 2 fixed)
# -------------------------------------------------
def decode_rule_bits(rule_id: int) -> List[int]:
    """Decode 32-bit truth table: index i corresponds to 5-bit neighborhood i (0..31)."""
    return [(rule_id >> i) & 1 for i in range(32)]

def bits_to_int(bits: List[int]) -> int:
    """Convert bit list to integer (big-endian)."""
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v

def neighborhood_bits(i: int) -> List[int]:
    """Return [b0,b1,b2,b3,b4] for i in [0..31], big-endian."""
    return [(i >> k) & 1 for k in reversed(range(5))]


# -------------------------------------------------
# Load + deduplicate records
# -------------------------------------------------
def load_unique_rule_labels(classification_dir: str) -> Dict[int, str]:
    """
    Returns { ruleId: className }.
    First occurrence wins.
    Assumption: JSON contains only the 5 class names in CLASSES_5.
    """
    rule_map: Dict[int, str] = {}

    paths = sorted(glob.glob(os.path.join(classification_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"No json files found in: {classification_dir}")

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            continue

        for r in data:
            if "ruleId" not in r or "className" not in r:
                continue
            rule_id = int(r["ruleId"])
            if rule_id not in rule_map:
                rule_map[rule_id] = str(r["className"])

    return rule_map


# -------------------------------------------------
# Build labeled DBG transitions as 16-bit masks
# -------------------------------------------------
def build_dbg_trans_masks(rule_id: int) -> Tuple[List[int], List[int]]:
    """
    Returns:
      trans0[u] = 16-bit mask of dest nodes v from u using edges with out=0
      trans1[u] = 16-bit mask of dest nodes v from u using edges with out=1
    """
    rule_bits = decode_rule_bits(rule_id)
    trans0 = [0 for _ in range(16)]
    trans1 = [0 for _ in range(16)]

    for i in range(32):
        b0, b1, b2, b3, b4 = neighborhood_bits(i)
        u = bits_to_int([b0, b1, b2, b3])
        v = bits_to_int([b1, b2, b3, b4])
        out = rule_bits[i]
        if out == 0:
            trans0[u] |= (1 << v)
        else:
            trans1[u] |= (1 << v)

    return trans0, trans1


def delta_subset(mask: int, trans: List[int]) -> int:
    """δ(mask, symbol) = OR_{u in mask} trans[u], where trans[u] is a 16-bit dest mask."""
    res = 0
    m = mask
    while m:
        lsb = m & -m
        u = lsb.bit_length() - 1
        res |= trans[u]
        m -= lsb
    return res


# -------------------------------------------------
# Build reachable subset graph (BFS) with cap
# -------------------------------------------------
def build_subset_graph(
    rule_id: int,
    max_nodes: int = 4096,
    add_reverse: bool = True,
    init: str = "singletons",  # "singletons" or "all"
) -> Data:
    """
    Nodes:
      reachable subset masks (<= max_nodes)

    Node feature:
      x[idx] = 16-dim indicator vector of the subset

    Edge attr:
      [symbol, dir] float32
    """
    trans0, trans1 = build_dbg_trans_masks(rule_id)

    # init seeds
    if init == "singletons":
        seeds = [(1 << u) for u in range(16)]
    elif init == "all":
        seeds = [(1 << 16) - 1]
    else:
        raise ValueError(f"Unknown init: {init}")

    mask_to_idx: Dict[int, int] = {}
    queue: List[int] = []

    def add_mask(m: int):
        if m not in mask_to_idx and len(mask_to_idx) < max_nodes:
            mask_to_idx[m] = len(mask_to_idx)
            queue.append(m)

    # include seeds + empty set
    for m in seeds:
        add_mask(m)
    add_mask(0)

    src: List[int] = []
    dst: List[int] = []
    eattr: List[List[float]] = []

    while queue:
        m = queue.pop(0)
        i = mask_to_idx[m]

        # symbol=0
        m0 = delta_subset(m, trans0)
        add_mask(m0)
        j0 = mask_to_idx[m0]
        src.append(i); dst.append(j0)
        eattr.append([0.0, 0.0])  # forward

        # symbol=1
        m1 = delta_subset(m, trans1)
        add_mask(m1)
        j1 = mask_to_idx[m1]
        src.append(i); dst.append(j1)
        eattr.append([1.0, 0.0])  # forward

        if len(mask_to_idx) >= max_nodes:
            break

    # add reverse edges (optional)
    if add_reverse:
        f_src = src.copy()
        f_dst = dst.copy()
        f_attr = eattr.copy()
        for s, d, a in zip(f_src, f_dst, f_attr):
            src.append(d)
            dst.append(s)
            eattr.append([a[0], 1.0])  # same symbol, reverse

    # node features: 16-bit indicator
    N = len(mask_to_idx)
    x = torch.zeros((N, 16), dtype=torch.float32)
    for mask, idx in mask_to_idx.items():
        # fill bits
        mm = mask
        while mm:
            lsb = mm & -mm
            u = lsb.bit_length() - 1
            x[idx, u] = 1.0
            mm -= lsb

    data = Data(
        x=x,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(eattr, dtype=torch.float32),  # [E,2]
    )
    data.rule_id = torch.tensor([rule_id], dtype=torch.long)
    data.num_subset_nodes = torch.tensor([N], dtype=torch.long)  # debug
    return data


# -------------------------------------------------
# PyG Dataset
# -------------------------------------------------
class Radius2SubsetDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        classification_dir: str,
        max_nodes: int = 4096,
        add_reverse: bool = True,
        init: str = "singletons",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.classification_dir = classification_dir
        self.max_nodes = max_nodes
        self.add_reverse = add_reverse
        self.init = init
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        meta = torch.load(self.processed_paths[1], weights_only=False)
        self.classes = meta["classes"]
        self.label_map = meta["label_map"]

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt", "meta.pt"]

    def process(self):
        rule_label_map = load_unique_rule_labels(self.classification_dir)

        classes = CLASSES_5
        label_map = LABEL_MAP_5

        data_list: List[Data] = []
        for rule_id, cls in rule_label_map.items():
            g = build_subset_graph(
                rule_id,
                max_nodes=self.max_nodes,
                add_reverse=self.add_reverse,
                init=self.init,
            )
            g.y = torch.tensor([label_map[cls]], dtype=torch.long)
            g.class_name = cls  # debug only
            data_list.append(g)

        data, slices = self.collate(data_list)

        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        torch.save({"classes": classes, "label_map": label_map}, self.processed_paths[1])


# -------------------------------------------------
# CLI
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-dir", default="classification_data")
    parser.add_argument("--root", default="output/pyg_radius2_subset_std")
    parser.add_argument("--max-nodes", type=int, default=4096)
    parser.add_argument("--init", choices=["singletons", "all"], default="singletons")
    parser.add_argument("--no-reverse", action="store_true")
    args = parser.parse_args()

    ds = Radius2SubsetDataset(
        root=args.root,
        classification_dir=args.classification_dir,
        max_nodes=args.max_nodes,
        add_reverse=(not args.no_reverse),
        init=args.init,
    )

    print("✅ Done")
    print(f"graphs        : {len(ds)}")
    print(f"classes       : {len(ds.classes)}  {ds.classes}")
    print(f"processed dir : {ds.processed_dir}")

    g0 = ds[0]
    print("\n--- sample[0] ---")
    print(g0)
    print("x shape       :", tuple(g0.x.shape))            # (N,16)
    print("edge_index    :", tuple(g0.edge_index.shape))  # (2,E)
    print("edge_attr     :", tuple(g0.edge_attr.shape))   # (E,2) [symbol,dir]
    print("rule_id       :", int(g0.rule_id.item()))
    print("y             :", int(g0.y.item()))
    print("class_name    :", getattr(g0, "class_name", None))
    print("num_subset_nodes:", int(getattr(g0, "num_subset_nodes", torch.tensor([-1])).item()))

if __name__ == "__main__":
    main()
