"""
Radius=2 invariants token dataset builder (PyG InMemoryDataset, standard format) — 5-class fixed

Graph:
- One node per rule (a "token graph")
- Node feature x: invariants/statistics from the 32-entry truth table
- Optional self-loop edge (kept minimal)

Features (9-dim):
  0: p1         (output=1 density)
  1: sens_mean  (avg sensitivity over 5 bits)
  2: infl_l2
  3: infl_l1
  4: infl_c
  5: infl_r1
  6: infl_r2
  7: lr_sym     (left-right mirror symmetry flag)
  8: comp_sym   (complement symmetry flag)

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

def neighborhood_bits(i: int) -> List[int]:
    """Return [l2,l1,c,r1,r2] big-endian for i in [0..31]."""
    return [(i >> k) & 1 for k in reversed(range(5))]

def bits_to_index(bits: List[int]) -> int:
    """bits is big-endian length 5 -> index in [0..31]."""
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v

def mirror_bits(bits: List[int]) -> List[int]:
    """[l2,l1,c,r1,r2] -> [r2,r1,c,l1,l2]"""
    return [bits[4], bits[3], bits[2], bits[1], bits[0]]

def complement_bits(bits: List[int]) -> List[int]:
    """bitwise complement"""
    return [1 - b for b in bits]


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
# Compute invariants from truth table
# -------------------------------------------------
def compute_invariants(rule_id: int) -> torch.Tensor:
    """
    Returns x: shape [1, 9] float32
    """
    tb = decode_rule_bits(rule_id)  # length 32, index is 5-bit neighborhood
    tb = [int(v) for v in tb]

    # (A) density of ones
    p1 = sum(tb) / 32.0

    # (B) influence per bit
    infl = []
    for bit_pos in range(5):
        # flip this bit for all 32 inputs and count output changes
        changes = 0
        for i in range(32):
            bits = neighborhood_bits(i)
            bits_flip = bits.copy()
            bits_flip[bit_pos] = 1 - bits_flip[bit_pos]
            j = bits_to_index(bits_flip)
            if tb[i] != tb[j]:
                changes += 1
        infl.append(changes / 32.0)  # probability output changes if flip this bit

    sens_mean = sum(infl) / 5.0

    # (C) symmetries
    # left-right mirror: rule(bits) == rule(mirror(bits)) for all bits
    lr_ok = 1.0
    for i in range(32):
        bits = neighborhood_bits(i)
        j = bits_to_index(mirror_bits(bits))
        if tb[i] != tb[j]:
            lr_ok = 0.0
            break

    # complement symmetry: rule(bits) == 1 - rule(complement(bits))
    comp_ok = 1.0
    for i in range(32):
        bits = neighborhood_bits(i)
        j = bits_to_index(complement_bits(bits))
        if tb[i] != (1 - tb[j]):
            comp_ok = 0.0
            break

    feat = [p1, sens_mean] + infl + [lr_ok, comp_ok]
    x = torch.tensor([feat], dtype=torch.float32)  # [1,9]
    return x


def build_invariant_token_graph(rule_id: int, add_self_loop: bool = True) -> Data:
    """
    One-node graph with invariants feature.
    Edge_attr kept minimal.
    """
    x = compute_invariants(rule_id)  # [1,9]

    if add_self_loop:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)  # one self-loop
        edge_attr = torch.tensor([[1.0]], dtype=torch.float32)   # dummy attr
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.rule_id = torch.tensor([rule_id], dtype=torch.long)
    return data


# -------------------------------------------------
# PyG Dataset
# -------------------------------------------------
class Radius2InvariantsDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        classification_dir: str,
        add_self_loop: bool = True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.classification_dir = classification_dir
        self.add_self_loop = add_self_loop
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
        feature_names = [
            "p1", "sens_mean",
            "infl_l2", "infl_l1", "infl_c", "infl_r1", "infl_r2",
            "lr_sym", "comp_sym"
        ]

        data_list: List[Data] = []
        for rule_id, cls in rule_label_map.items():
            g = build_invariant_token_graph(rule_id, add_self_loop=self.add_self_loop)
            g.y = torch.tensor([label_map[cls]], dtype=torch.long)
            g.class_name = cls
            data_list.append(g)

        data, slices = self.collate(data_list)

        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        torch.save(
            {"classes": classes, "label_map": label_map, "feature_names": feature_names},
            self.processed_paths[1]
        )


# -------------------------------------------------
# CLI
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification-dir", default="classification_data")
    parser.add_argument("--root", default="output/pyg_radius2_invariants_std")
    parser.add_argument("--no-self-loop", action="store_true")
    args = parser.parse_args()

    ds = Radius2InvariantsDataset(
        root=args.root,
        classification_dir=args.classification_dir,
        add_self_loop=(not args.no_self_loop),
    )

    print("✅ Done")
    print(f"graphs        : {len(ds)}")
    print(f"classes       : {len(ds.classes)}  {ds.classes}")
    print(f"feature_names : {ds.feature_names}")
    print(f"processed dir : {ds.processed_dir}")

    g0 = ds[0]
    print("\n--- sample[0] ---")
    print(g0)
    print("x shape       :", tuple(g0.x.shape))           # (1,9)
    print("x             :", g0.x.squeeze(0).tolist())
    print("edge_index    :", tuple(g0.edge_index.shape))
    print("edge_attr     :", tuple(g0.edge_attr.shape))
    print("rule_id       :", int(g0.rule_id.item()))
    print("y             :", int(g0.y.item()))
    print("class_name    :", getattr(g0, "class_name", None))


if __name__ == "__main__":
    main()
