"""
Radius=2 SCC skeleton dataset builder (PyG InMemoryDataset, standard format) — 5-class fixed

Base graph for SCC:
- radius=2 de Bruijn graph (16 nodes), forward edges only (32).
- Each edge has output out in {0,1} from the rule truth table.

SCC skeleton:
- Nodes: SCCs of the forward DBG (condensation graph is a DAG).
- Node features x (float32):
    [ size_norm,
      out1_frac_in_scc,
      self_loop,
      avg_outdeg,
      avg_indeg ]
- Edges: between SCCs if any forward edge crosses SCC boundary.
- Edge attr (float32): [p1_between, dir]
    p1_between: fraction of cross edges with out=1
    dir: 0 forward, 1 reverse (optional)

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
# Build forward DBG edges (32) with out label
# -------------------------------------------------
def build_forward_dbg(rule_id: int) -> Tuple[List[int], List[int], List[int]]:
    """
    returns src, dst, out for 32 forward edges on 16 nodes
    """
    rule_bits = decode_rule_bits(rule_id)
    src, dst, outs = [], [], []
    for i in range(32):
        b0,b1,b2,b3,b4 = neighborhood_bits(i)
        u = bits_to_int([b0,b1,b2,b3])
        v = bits_to_int([b1,b2,b3,b4])
        out = int(rule_bits[i])
        src.append(u); dst.append(v); outs.append(out)
    return src, dst, outs


# -------------------------------------------------
# SCC (Kosaraju) on 16 nodes
# -------------------------------------------------
def kosaraju_scc(n: int, src: List[int], dst: List[int]) -> List[int]:
    """
    returns comp_id for each node in [0..n-1]
    """
    g = [[] for _ in range(n)]
    gr = [[] for _ in range(n)]
    for u,v in zip(src,dst):
        g[u].append(v)
        gr[v].append(u)

    visited = [False]*n
    order = []

    def dfs1(u: int):
        visited[u] = True
        for v in g[u]:
            if not visited[v]:
                dfs1(v)
        order.append(u)

    for u in range(n):
        if not visited[u]:
            dfs1(u)

    comp = [-1]*n
    def dfs2(u: int, cid: int):
        comp[u] = cid
        for v in gr[u]:
            if comp[v] == -1:
                dfs2(v, cid)

    cid = 0
    for u in reversed(order):
        if comp[u] == -1:
            dfs2(u, cid)
            cid += 1

    return comp  # length n, values 0..cid-1


# -------------------------------------------------
# Build SCC skeleton graph
# -------------------------------------------------
def build_scc_skeleton_graph(rule_id: int, add_reverse: bool = True) -> Data:
    """
    Node features:
      [ size_norm, out1_frac_in_scc, self_loop, avg_outdeg, avg_indeg ]
    Edge attr:
      [p1_between, dir]
    """
    src, dst, outs = build_forward_dbg(rule_id)
    comp = kosaraju_scc(16, src, dst)
    K = max(comp) + 1

    # nodes in each SCC
    nodes_in = [[] for _ in range(K)]
    for u,cid in enumerate(comp):
        nodes_in[cid].append(u)

    # compute degrees within SCC (forward only)
    outdeg = [0]*16
    indeg = [0]*16
    for u,v in zip(src,dst):
        outdeg[u] += 1
        indeg[v] += 1

    # SCC internal edge stats
    # count edges whose both ends inside same SCC
    in_edge_total = [0]*K
    in_edge_out1  = [0]*K
    has_internal_edge = [0]*K

    for u,v,o in zip(src,dst,outs):
        cu, cv = comp[u], comp[v]
        if cu == cv:
            in_edge_total[cu] += 1
            in_edge_out1[cu] += int(o == 1)
            has_internal_edge[cu] = 1  # at least one internal edge

    # node features
    x = torch.zeros((K, 5), dtype=torch.float32)
    for cid in range(K):
        size = len(nodes_in[cid])
        size_norm = size / 16.0
        tot = in_edge_total[cid]
        out1_frac = (in_edge_out1[cid] / tot) if tot > 0 else 0.0
        self_loop = float(has_internal_edge[cid])
        avg_out = sum(outdeg[u] for u in nodes_in[cid]) / max(size, 1)
        avg_in  = sum(indeg[u] for u in nodes_in[cid]) / max(size, 1)
        x[cid] = torch.tensor([size_norm, out1_frac, self_loop, avg_out, avg_in], dtype=torch.float32)

    # edges between SCCs + stats
    # map (cu,cv) -> (count, out1count)
    edge_stat: Dict[Tuple[int,int], List[int]] = {}
    for u,v,o in zip(src,dst,outs):
        cu, cv = comp[u], comp[v]
        if cu == cv:
            continue
        key = (cu, cv)
        if key not in edge_stat:
            edge_stat[key] = [0,0]
        edge_stat[key][0] += 1
        edge_stat[key][1] += int(o == 1)

    e_src, e_dst, eattr = [], [], []
    for (cu,cv), (cnt, out1cnt) in edge_stat.items():
        p1_between = out1cnt / cnt if cnt > 0 else 0.0
        e_src.append(cu); e_dst.append(cv)
        eattr.append([float(p1_between), 0.0])  # forward

    if add_reverse:
        f_src = e_src.copy()
        f_dst = e_dst.copy()
        f_attr = eattr.copy()
        for s,d,a in zip(f_src,f_dst,f_attr):
            e_src.append(d)
            e_dst.append(s)
            eattr.append([a[0], 1.0])  # reverse

    if len(e_src) == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr  = torch.empty((0,2), dtype=torch.float32)
    else:
        edge_index = torch.tensor([e_src, e_dst], dtype=torch.long)
        edge_attr  = torch.tensor(eattr, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.rule_id = torch.tensor([rule_id], dtype=torch.long)
    data.num_scc = torch.tensor([K], dtype=torch.long)  # debug
    return data


# -------------------------------------------------
# PyG Dataset
# -------------------------------------------------
class Radius2SCCSkeletonDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        classification_dir: str,
        add_reverse: bool = True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.classification_dir = classification_dir
        self.add_reverse = add_reverse
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        meta = torch.load(self.processed_paths[1], weights_only=False)
        self.classes = meta["classes"]
        self.label_map = meta["label_map"]
        self.feature_names = meta["feature_names"]

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
        feature_names = ["size_norm", "out1_frac_in_scc", "self_loop", "avg_outdeg", "avg_indeg"]

        data_list: List[Data] = []
        for rule_id, cls in rule_label_map.items():
            g = build_scc_skeleton_graph(rule_id, add_reverse=self.add_reverse)
            g.y = torch.tensor([label_map[cls]], dtype=torch.long)
            g.class_name = cls
            data_list.append(g)

        data, slices = self.collate(data_list)

        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        torch.save({"classes": classes, "label_map": label_map, "feature_names": feature_names}, self.processed_paths[1])


# -------------------------------------------------
# CLI
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-dir", default="classification_data")
    parser.add_argument("--root", default="output/pyg_radius2_scc_skeleton_std")
    parser.add_argument("--no-reverse", action="store_true")
    args = parser.parse_args()

    ds = Radius2SCCSkeletonDataset(
        root=args.root,
        classification_dir=args.classification_dir,
        add_reverse=(not args.no_reverse),
    )

    print("✅ Done")
    print(f"graphs        : {len(ds)}")
    print(f"classes       : {len(ds.classes)}  {ds.classes}")
    print(f"feature_names : {ds.feature_names}")
    print(f"processed dir : {ds.processed_dir}")

    g0 = ds[0]
    print("\n--- sample[0] ---")
    print(g0)
    print("x shape       :", tuple(g0.x.shape))            # (K,5)
    print("edge_index    :", tuple(g0.edge_index.shape))  # (2,E)
    print("edge_attr     :", tuple(g0.edge_attr.shape))   # (E,2) [p1_between,dir]
    print("rule_id       :", int(g0.rule_id.item()))
    print("y             :", int(g0.y.item()))
    print("class_name    :", getattr(g0, "class_name", None))
    print("num_scc       :", int(getattr(g0, "num_scc", torch.tensor([-1])).item()))

if __name__ == "__main__":
    main()
