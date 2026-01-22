"""
Dependency graph builder for radius=2 cellular automata

Preprocessing rules:
- deduplicate by ruleId
- keep only: ruleId + className
- no radius checks
- output PyG in-memory dataset

Dependency definition (binary):
For position i in {-2,-1,0,+1,+2}, there is an edge i -> out iff
exists two neighborhoods differing only at i that produce different outputs.

Graph definition:
- Nodes: 6
  node 0..4 : input positions [-2,-1,0,+1,+2] (in that order)
  node 5    : output node (center at t+1)
- Directed edges: subset of {0..4} -> 5
- edge_attr: uint8 scalar 1 for each existing dependency edge
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
    """Decode 32-bit truth table: output for neighborhood index i is (rule_id >> i) & 1"""
    return [(rule_id >> i) & 1 for i in range(32)]

def bits5_to_index(b0: int, b1: int, b2: int, b3: int, b4: int) -> int:
    """[b0..b4] -> integer in [0,31] (b0 is MSB)"""
    return (b0 << 4) | (b1 << 3) | (b2 << 2) | (b3 << 1) | b4


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
# Dependency test (binary)
# -------------------------------------------------
def position_is_dependent(rule_bits: List[int], pos: int) -> bool:
    """
    pos: 0..4 corresponding to [-2,-1,0,+1,+2]
    Checks if output depends on this input bit.
    """
    # Enumerate all assignments of the other 4 bits (16 cases)
    # We'll build two neighborhoods: one with x_pos=0, one with x_pos=1
    # and compare their outputs.
    other_positions = [j for j in range(5) if j != pos]

    for mask in range(16):
        bits = [0, 0, 0, 0, 0]

        # fill other 4 positions from mask (MSB..LSB over other_positions)
        for k, j in enumerate(other_positions):
            # k in [0..3], we map mask bit (3-k) to keep stable ordering
            bits[j] = (mask >> (3 - k)) & 1

        # compare outputs for x_pos = 0 vs 1
        bits[pos] = 0
        idx0 = bits5_to_index(bits[0], bits[1], bits[2], bits[3], bits[4])
        y0 = rule_bits[idx0]

        bits[pos] = 1
        idx1 = bits5_to_index(bits[0], bits[1], bits[2], bits[3], bits[4])
        y1 = rule_bits[idx1]

        if y0 != y1:
            return True

    return False


# -------------------------------------------------
# Build dependency graph (radius=2)
# -------------------------------------------------
def build_dependency_graph(rule_id: int) -> Data:
    """
    Nodes:
      0..4 : input positions [-2,-1,0,+1,+2]
      5    : output node (center at t+1)

    Edges:
      pos -> 5 exists iff dependency holds for pos.

    Node features x (uint8):
      one-hot for node type:
        [1,0] = input node
        [0,1] = output node

    edge_attr (uint8):
      scalar 1 per edge (existence)
    """
    rule_bits = decode_rule_bits(rule_id)

    # node features (6,2) uint8
    x = torch.zeros((6, 2), dtype=torch.uint8)
    x[0:5, 0] = 1  # inputs
    x[5, 1] = 1    # output

    src, dst, edge_attr = [], [], []

    for pos in range(5):
        if position_is_dependent(rule_bits, pos):
            src.append(pos)
            dst.append(5)
            edge_attr.append(1)

    # handle no-edge case (PyG allows empty edge_index)
    if len(src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr_t = torch.empty((0, 1), dtype=torch.uint8)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.uint8).unsqueeze(1)  # [E,1]

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr_t,
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
        g = build_dependency_graph(rule_id)
        g.y = torch.tensor([label_map[cls]], dtype=torch.long)
        dataset.append(g)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "pyg_radius2_dependency_dedup.pt")
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