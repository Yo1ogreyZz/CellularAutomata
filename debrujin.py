"""
de Bruijn graph builder for radius=2 cellular automata

Preprocessing rules:
- deduplicate by ruleId
- keep only: ruleId + className
- no radius checks
- output PyG in-memory dataset

Graph definition:
- Nodes: all length-4 binary contexts (16 nodes)
- Edges: one per length-5 neighborhood (32 edges)
- Direction: spatial shift (b0 b1 b2 b3) -> (b1 b2 b3 b4)
- Edge attribute: rule output c' ∈ {0,1}
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

def bits_to_int(bits: List[int]) -> int:
    """Convert bit list to integer"""
    v = 0
    for b in bits:
        v = (v << 1) | b
    return v

def neighborhood_bits(i: int) -> List[int]:
    """Return [b0, b1, b2, b3, b4]"""
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
# Build de Bruijn graph (radius=2)
# -------------------------------------------------
def build_debruijn_graph(rule_id: int) -> Data:
    """
    Nodes:
      16 nodes, each a 4-bit context

    Edges:
      32 edges, one per 5-bit neighborhood
      (b0 b1 b2 b3) -> (b1 b2 b3 b4)

    Edge attribute:
      c' = rule(b0 b1 b2 b3 b4) ∈ {0,1}
    """

    rule_bits = decode_rule_bits(rule_id)

    # ----- nodes -----
    # node feature: explicit 4-bit context (uint8)
    x = []
    for i in range(16):
        bits = [(i >> k) & 1 for k in reversed(range(4))]
        x.append(bits)
    x = torch.tensor(x, dtype=torch.uint8)   # [16, 4]

    # ----- edges -----
    src, dst, edge_attr = [], [], []

    for i in range(32):
        b0, b1, b2, b3, b4 = neighborhood_bits(i)

        left_node  = bits_to_int([b0, b1, b2, b3])
        right_node = bits_to_int([b1, b2, b3, b4])

        src.append(left_node)
        dst.append(right_node)

        # rule output stored on edge
        edge_attr.append(rule_bits[i])

    data = Data(
        x=x,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.uint8).unsqueeze(1),  # [32,1]
    )

    data.rule_id = rule_id
    return data


# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):
    rule_label_map = load_unique_rule_labels(args.classification_dir)

    # label encoding
    classes = sorted(set(rule_label_map.values()))
    label_map = {c: i for i, c in enumerate(classes)}

    dataset: List[Data] = []

    for rule_id, cls in rule_label_map.items():
        g = build_debruijn_graph(rule_id)
        g.y = torch.tensor([label_map[cls]], dtype=torch.long)
        dataset.append(g)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "pyg_radius2_debruijn_dedup.pt")
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