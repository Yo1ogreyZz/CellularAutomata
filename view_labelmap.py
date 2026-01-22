"""
View label map for CA rule classification.

This script reproduces exactly the same label encoding logic
used in:
- symbol01 builder
- de Bruijn builder
- dependency builder
cd
It shows:
  y -> className
mapping, so that saved PyG labels can be interpreted correctly.
"""

import os
import glob
import json
import argparse


# -------------------------------------------------
# Load + deduplicate rule labels (same logic)
# -------------------------------------------------
def load_unique_rule_labels(classification_dir: str):
    """
    Returns:
        { ruleId : className }
    First occurrence wins.
    """
    rule_map = {}

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
# Main
# -------------------------------------------------
def main(args):
    rule_label_map = load_unique_rule_labels(args.classification_dir)

    # EXACT same encoding rule as graph builders
    classes = sorted(set(rule_label_map.values()))
    label_map = {i: c for i, c in enumerate(classes)}

    print("=" * 60)
    print("Label map (y -> className)")
    print("=" * 60)

    for y, class_name in label_map.items():
        print(f"{y:2d} -> {class_name}")

    print("=" * 60)
    print(f"Total unique rules : {len(rule_label_map)}")
    print(f"Total classes      : {len(classes)}")
    print("=" * 60)


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
    args = parser.parse_args()
    main(args)