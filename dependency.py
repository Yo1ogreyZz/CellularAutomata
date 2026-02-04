"""
Enhanced Dependency Graph builder for radius=2 CA (ring + strength) -> Standard PyG InMemoryDataset (5-class fixed)

Nodes: 6
  0..4 : input positions [-2,-1,0,+1,+2]
  5    : output node (center at t+1)

Edges:
  (A) Ring edges among 0..4 (periodic):
      i -> (i+1)%5 and i -> (i-1)%5
  (B) Dependency edges:
      pos -> 5 with strength in [0,1]
      strength = (#assignments where flipping pos changes output) / 16

Node features x (float32), dim=7:
  [type_onehot(2) || pos_onehot(5)]
    input nodes: type=[1,0], pos onehot
    output node: type=[0,1], pos all zeros

Edge features edge_attr (float32), dim=3:
  [edge_type, strength, dir]
    ring edge: edge_type=0, strength=0, dir in {-1,+1}
    dep  edge: edge_type=1, strength in [0,1], dir=0

Output:
  processed/data.pt = (data, slices)
  processed/meta.pt = {classes,label_map}

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
from typing import Dict, List

import torch
import torch.nn.functional as F
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


# -------------------------
# CA utilities (radius=2)
# -------------------------
def decode_rule_bits(rule_id: int) -> List[int]:
    return [(rule_id >> i) & 1 for i in range(32)]

def bits5_to_index(b0: int, b1: int, b2: int, b3: int, b4: int) -> int:
    return (b0 << 4) | (b1 << 3) | (b2 << 2) | (b3 << 1) | b4


# -------------------------
# Load + deduplicate records
# -------------------------
def load_unique_rule_labels(classification_dir: str) -> Dict[int, str]:
    """
    Returns {ruleId: className}. First occurrence wins.
    Assumption: JSON contains only these 5 class names.
    """
    rule_map: Dict[int, str] = {}
    paths = sorted(glob.glob(os.path.join(classification_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"No json files in {classification_dir}")

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


# -------------------------
# Dependency strength
# -------------------------
def dependency_strength(rule_bits: List[int], pos: int) -> float:
    """
    pos: 0..4 for [-2,-1,0,+1,+2]
    strength = fraction of assignments of other 4 bits where flipping pos changes output
    """
    other_positions = [j for j in range(5) if j != pos]
    changed = 0

    for mask in range(16):
        bits = [0, 0, 0, 0, 0]
        for k, j in enumerate(other_positions):
            bits[j] = (mask >> (3 - k)) & 1

        bits[pos] = 0
        idx0 = bits5_to_index(bits[0], bits[1], bits[2], bits[3], bits[4])
        y0 = rule_bits[idx0]

        bits[pos] = 1
        idx1 = bits5_to_index(bits[0], bits[1], bits[2], bits[3], bits[4])
        y1 = rule_bits[idx1]

        if y0 != y1:
            changed += 1

    return changed / 16.0


# -------------------------
# Build enhanced dependency graph
# -------------------------
def build_dependency_graph(rule_id: int) -> Data:
    rule_bits = decode_rule_bits(rule_id)

    # ----- node features: type(2) + pos_onehot(5) => 7 dims -----
    x = torch.zeros((6, 7), dtype=torch.float32)

    # input nodes 0..4
    x[0:5, 0] = 1.0  # type input
    pos_oh = F.one_hot(torch.arange(5), num_classes=5).to(torch.float32)  # [5,5]
    x[0:5, 2:7] = pos_oh

    # output node 5
    x[5, 1] = 1.0  # type output
    # pos_onehot for output stays zero

    src, dst, eattr = [], [], []

    # ----- (A) ring edges among 0..4 -----
    for i in range(5):
        j = (i + 1) % 5
        k = (i - 1) % 5

        # i -> j  (dir=+1)
        src.append(i); dst.append(j)
        eattr.append([0.0, 0.0, +1.0])

        # i -> k  (dir=-1)
        src.append(i); dst.append(k)
        eattr.append([0.0, 0.0, -1.0])

    # ----- (B) dependency edges pos -> 5 with strength -----
    for pos in range(5):
        s = dependency_strength(rule_bits, pos)  # [0,1]
        if s > 0.0:
            src.append(pos); dst.append(5)
            eattr.append([1.0, float(s), 0.0])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(eattr, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.rule_id = torch.tensor([rule_id], dtype=torch.long)
    return data


# -------------------------
# Standard PyG InMemoryDataset
# -------------------------
class Radius2DependencyDataset(InMemoryDataset):
    def __init__(self, root: str, classification_dir: str, transform=None, pre_transform=None, pre_filter=None):
        self.classification_dir = classification_dir
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        meta = torch.load(self.processed_paths[1], weights_only=False)
        self.classes = meta["classes"]
        self.label_map = meta["label_map"]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt", "meta.pt"]

    def process(self):
        rule_label_map = load_unique_rule_labels(self.classification_dir)

        # ✅ fixed mapping (5-class)
        classes = CLASSES_5
        label_map = LABEL_MAP_5

        data_list: List[Data] = []
        for rule_id, cls in rule_label_map.items():
            # assumption: cls always in LABEL_MAP_5
            g = build_dependency_graph(rule_id)
            g.y = torch.tensor([label_map[cls]], dtype=torch.long)
            g.class_name = cls  # debug
            data_list.append(g)

        data, slices = self.collate(data_list)

        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        torch.save({"classes": classes, "label_map": label_map}, self.processed_paths[1])


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-dir", default="classification_data")
    parser.add_argument("--root", default="output/pyg_radius2_dependency_std")
    args = parser.parse_args()

    ds = Radius2DependencyDataset(root=args.root, classification_dir=args.classification_dir)

    print("✅ Done")
    print("graphs        :", len(ds))
    print("classes       :", len(ds.classes), ds.classes)
    print("processed dir :", ds.processed_dir)

    g0 = ds[0]
    print("\n--- sample[0] ---")
    print(g0)
    print("x shape       :", tuple(g0.x.shape))            # (6,7)
    print("edge_index    :", tuple(g0.edge_index.shape))
    print("edge_attr     :", tuple(g0.edge_attr.shape))   # (~(10 + dep_edges), 3)
    print("rule_id       :", int(g0.rule_id.item()))
    print("y             :", int(g0.y.item()))
    print("class_name    :", getattr(g0, "class_name", None))

    etype = g0.edge_attr[:, 0]
    ring_n = int((etype == 0).sum().item())
    dep_n = int((etype == 1).sum().item())
    print("ring edges    :", ring_n, "(expect 10)")
    print("dep edges     :", dep_n, "(0..5 depending on rule)")


if __name__ == "__main__":
    main()
