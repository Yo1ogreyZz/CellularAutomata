"""
symbol0/1 graph builder (radius=2) -> Standard PyG InMemoryDataset (5-class fixed)

Graph:
- Nodes: 2 nodes represent center cell state at time t: {0,1}
- Edges: one per 5-bit neighborhood (32 total)
         forward:  src = center(t)  -> dst = center(t+1) = rule output
         reverse:  dst -> src  (for better message passing)
- Edge attr: [l2, l1, r1, r2, dir]
    dir=0 forward edge, dir=1 reverse edge
- Deduplicate by ruleId, keep ruleId + className
- Output: processed/data.pt (data,slices) + processed/meta.pt (classes,label_map)

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


# -------------------------------------------------
# CA utilities (radius = 2)
# -------------------------------------------------
def decode_rule_bits(rule_id: int) -> List[int]:
    """Decode 32-bit truth table (index i in [0..31])."""
    return [(rule_id >> i) & 1 for i in range(32)]

def neighborhood_bits(i: int) -> List[int]:
    """Return [l2, l1, c, r1, r2] big-endian."""
    return [(i >> k) & 1 for k in reversed(range(5))]


# -------------------------------------------------
# Load + deduplicate records
# -------------------------------------------------
def load_unique_rule_labels(classification_dir: str) -> Dict[int, str]:
    """
    Returns: { ruleId : className }, first occurrence wins.
    Assumption: JSON contains only the 5 class names in CLASSES_5.
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


# -------------------------------------------------
# Build symbol0/1 graph (with reverse edges)
# -------------------------------------------------
def build_symbol01_graph(rule_id: int) -> Data:
    """
    Nodes:
      0 -> center state 0
      1 -> center state 1

    Edges (forward + reverse):
      forward: center(t)=c -> out=rule(l2 l1 c r1 r2)
      reverse: out -> c

    Edge attr:
      [l2, l1, r1, r2, dir]  (float32)
      dir=0 forward, dir=1 reverse
    """
    rule_bits = decode_rule_bits(rule_id)

    # node features: one-hot(2)
    x = F.one_hot(torch.tensor([0, 1], dtype=torch.long), num_classes=2).to(torch.float32)  # [2,2]

    src, dst, edge_attr = [], [], []

    # forward edges: 32
    for i in range(32):
        l2, l1, c, r1, r2 = neighborhood_bits(i)
        out = rule_bits[i]

        src.append(c)
        dst.append(out)
        edge_attr.append([float(l2), float(l1), float(r1), float(r2), 0.0])

    # reverse edges: 32
    for i in range(32):
        l2, l1, c, r1, r2 = neighborhood_bits(i)
        out = rule_bits[i]

        src.append(out)
        dst.append(c)
        edge_attr.append([float(l2), float(l1), float(r1), float(r2), 1.0])

    data = Data(
        x=x,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),  # [64,5]
    )

    data.rule_id = torch.tensor([rule_id], dtype=torch.long)
    return data


# -------------------------------------------------
# Standard PyG InMemoryDataset
# -------------------------------------------------
class Radius2Symbol01Dataset(InMemoryDataset):
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
            g = build_symbol01_graph(rule_id)
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
    parser.add_argument("--root", default="output/pyg_radius2_symbol01_std")
    args = parser.parse_args()

    ds = Radius2Symbol01Dataset(root=args.root, classification_dir=args.classification_dir)

    print("✅ Done")
    print(f"graphs        : {len(ds)}")
    print(f"classes       : {len(ds.classes)}  {ds.classes}")
    print(f"processed dir : {ds.processed_dir}")

    g0 = ds[0]
    print("\n--- sample[0] ---")
    print(g0)
    print("x shape       :", tuple(g0.x.shape))           # (2,2)
    print("edge_index    :", tuple(g0.edge_index.shape)) # (2,64)
    print("edge_attr     :", tuple(g0.edge_attr.shape))  # (64,5)
    print("rule_id       :", int(g0.rule_id.item()))
    print("y             :", int(g0.y.item()))
    print("class_name    :", getattr(g0, "class_name", None))


if __name__ == "__main__":
    main()
