"""Split cached .pt file into separate view files."""

import os
import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Input .pt file")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    args = ap.parse_args()

    samples = torch.load(args.pt, weights_only=False)
    os.makedirs(args.out_dir, exist_ok=True)

    symbol_list = []
    debruijn_list = []
    dependency_list = []
    labels = []
    rule_ids = []

    for s in samples:
        symbol_list.append(s["symbol"])
        debruijn_list.append(s["debruijn"])
        dependency_list.append(s["dependency"])
        labels.append(s["label"])
        rule_ids.append(s["rule_id"])

    torch.save(symbol_list, os.path.join(args.out_dir, "symbol.pt"))
    torch.save(debruijn_list, os.path.join(args.out_dir, "debruijn.pt"))
    torch.save(dependency_list, os.path.join(args.out_dir, "dependency.pt"))
    torch.save({"label": labels, "rule_id": rule_ids}, os.path.join(args.out_dir, "meta.pt"))

    print(f"Saved to {args.out_dir}/:")
    print("  - symbol.pt")
    print("  - debruijn.pt")
    print("  - dependency.pt")
    print("  - meta.pt")


if __name__ == "__main__":
    main()
