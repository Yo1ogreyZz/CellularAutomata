"""
Radius=2 de Bruijn graph dataset builder (PyG InMemoryDataset, standard format) — 5-class fixed

Changes vs your 8-class version:
- Uses fixed 5-class label mapping (DO NOT sort class names)
- Keeps everything else identical (graph structure, reverse edges, edge_attr, node_feature options)

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

def int_to_bits(i: int, n: int) -> List[int]:
    """Return big-endian n-bit representation."""
    return [(i >> k) & 1 for k in reversed(range(n))]

def neighborhood_bits(i: int) -> List[int]:
    """Return [b0, b1, b2, b3, b4] for i in [0..31], big-endian."""
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
# Build de Bruijn graph (radius=2)
# -------------------------------------------------
def build_debruijn_graph(
    rule_id: int,
    node_feature: str = "onehot16",   # "onehot16" or "bits4"
    add_reverse: bool = True
) -> Data:
    """
    Nodes (16):
      all 4-bit contexts

    Directed edges (32):
      one per 5-bit neighborhood (b0..b4)
      shift: (b0 b1 b2 b3) -> (b1 b2 b3 b4)

    edge_attr = [b0,b1,b2,b3,b4,out,dir] float32
    also adds reverse edges with dir=1 (optional)
    """
    rule_bits = decode_rule_bits(rule_id)

    # ----- nodes -----
    if node_feature == "onehot16":
        x = F.one_hot(torch.arange(16, dtype=torch.long), num_classes=16).to(torch.float32)  # [16,16]
    elif node_feature == "bits4":
        x_list = [int_to_bits(i, 4) for i in range(16)]
        x = torch.tensor(x_list, dtype=torch.float32)  # [16,4]
    else:
        raise ValueError(f"Unknown node_feature: {node_feature}")

    # ----- edges -----
    src, dst, eattr = [], [], []

    # forward edges
    for i in range(32):
        b0, b1, b2, b3, b4 = neighborhood_bits(i)
        u = bits_to_int([b0, b1, b2, b3])
        v = bits_to_int([b1, b2, b3, b4])
        out = rule_bits[i]
        src.append(u)
        dst.append(v)
        eattr.append([float(b0), float(b1), float(b2), float(b3), float(b4), float(out), 0.0])

    # reverse edges (optional)
    if add_reverse:
        for i in range(32):
            b0, b1, b2, b3, b4 = neighborhood_bits(i)
            u = bits_to_int([b0, b1, b2, b3])
            v = bits_to_int([b1, b2, b3, b4])
            out = rule_bits[i]
            src.append(v)
            dst.append(u)
            eattr.append([float(b0), float(b1), float(b2), float(b3), float(b4), float(out), 1.0])

    edge_index = torch.tensor([src, dst], dtype=torch.long)   # [2, E]
    edge_attr = torch.tensor(eattr, dtype=torch.float32)      # [E, 7]

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    data.rule_id = torch.tensor([rule_id], dtype=torch.long)
    return data


# -------------------------------------------------
# PyG Dataset
# -------------------------------------------------
class Radius2DeBruijnDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        classification_dir: str,
        node_feature: str = "onehot16",
        add_reverse: bool = True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.classification_dir = classification_dir
        self.node_feature = node_feature
        self.add_reverse = add_reverse
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

        # ✅ fixed mapping (5-class)
        classes = CLASSES_5
        label_map = LABEL_MAP_5

        data_list: List[Data] = []

        for rule_id, cls in rule_label_map.items():
            # assumption: cls always in LABEL_MAP_5
            g = build_debruijn_graph(
                rule_id,
                node_feature=self.node_feature,
                add_reverse=self.add_reverse
            )
            g.y = torch.tensor([label_map[cls]], dtype=torch.long)
            g.class_name = cls  # debug

            if self.pre_filter is not None and not self.pre_filter(g):
                continue
            if self.pre_transform is not None:
                g = self.pre_transform(g)

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
    parser.add_argument("--classification-dir", default="classification_data", help="Directory with JSON files")
    parser.add_argument("--root", default="output/pyg_radius2_debruijn_std", help="Dataset root (will create processed/)")
    parser.add_argument("--node-feature", default="onehot16", choices=["onehot16", "bits4"])
    parser.add_argument("--no-reverse", action="store_true", help="Disable reverse edges")
    args = parser.parse_args()

    ds = Radius2DeBruijnDataset(
        root=args.root,
        classification_dir=args.classification_dir,
        node_feature=args.node_feature,
        add_reverse=(not args.no_reverse),
    )

    print("✅ Done")
    print(f"   graphs        : {len(ds)}")
    print(f"   classes       : {len(ds.classes)}  {ds.classes}")
    print(f"   processed dir : {ds.processed_dir}")
    print(f"   data file     : {ds.processed_paths[0]}")
    print(f"   meta file     : {ds.processed_paths[1]}")

    g0 = ds[0]
    print("\n--- sample[0] ---")
    print(g0)
    print(f"x shape        : {tuple(g0.x.shape)}")
    print(f"edge_index     : {tuple(g0.edge_index.shape)}")
    print(f"edge_attr      : {tuple(g0.edge_attr.shape)}  (columns: b0..b4, out, dir)")
    print(f"y              : {int(g0.y.item())}")
    print(f"rule_id        : {int(g0.rule_id.item())}")
    print(f"class_name     : {getattr(g0, 'class_name', None)}")


if __name__ == "__main__":
    main()
