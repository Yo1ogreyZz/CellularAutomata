"""
symbol0/1 graph builder (radius=2)

Preprocessing rules:
- deduplicate by ruleId
- keep only: ruleId + className
- no radius checks
- output PyG in-memory dataset
"""

import os
import glob
import json
import argparse
from typing import Dict, List

import torch
from torch_geometric.data import Data


# -------------------------------------------------
# CA utilities (radius = 2 fixed)
# -------------------------------------------------
def decode_rule_bits(rule_id: int) -> List[int]:
    """Decode 32-bit truth table"""
    return [(rule_id >> i) & 1 for i in range(32)]

def neighborhood_bits(i: int) -> List[int]:
    """Return [l2, l1, c, r1, r2]"""
    return [(i >> k) & 1 for k in reversed(range(5))]


# -------------------------------------------------
# Load + deduplicate records
# -------------------------------------------------
def load_unique_rule_labels(classification_dir: str) -> Dict[int, str]:
    """
    Returns:
        { ruleId : className }
    First occurrence wins.
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
            rule_id = int(r["ruleId"])
            if rule_id not in rule_map:
                rule_map[rule_id] = str(r["className"])

    return rule_map


# -------------------------------------------------
# Build symbol0/1 graph
# -------------------------------------------------
def build_symbol01_graph(rule_id: int) -> Data:
    """
    Nodes:
      0 -> symbol 0
      1 -> symbol 1

    Edges:
      one per neighborhood pattern
      direction: center -> next_center
      edge_attr: [l2, l1, r1, r2]
    """

    rule_bits = decode_rule_bits(rule_id)

    # node features: one-hot
    x = torch.tensor([
        [1.0, 0.0],  # 0
        [0.0, 1.0],  # 1
    ], dtype=torch.float)

    src, dst, edge_attr = [], [], []

    for i in range(32):
        l2, l1, c, r1, r2 = neighborhood_bits(i)
        src.append(c)
        dst.append(rule_bits[i])
        edge_attr.append([l2, l1, r1, r2])

    data = Data(
        x=x,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )

    data.rule_id = rule_id
    return data


# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):
    rule_label_map = load_unique_rule_labels(args.classification_dir)

    # build label encoding
    classes = sorted(set(rule_label_map.values()))
    label_map = {c: i for i, c in enumerate(classes)}

    dataset: List[Data] = []

    for rule_id, cls in rule_label_map.items():
        g = build_symbol01_graph(rule_id)
        g.y = torch.tensor([label_map[cls]], dtype=torch.long)
        dataset.append(g)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "pyg_radius2_symbol01_dedup.pt")
    torch.save(dataset, out_path)

    print("✅ Done")
    print(f"   unique rules : {len(dataset)}")
    print(f"   saved to     : {out_path}")


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classification-dir",
        default="classification_data",
        help="Directory with JSON classification files"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory"
    )
    args = parser.parse_args()
    main(args)