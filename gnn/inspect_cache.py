"""Inspect cached .pt file structure."""

import argparse
import torch


def summarize_data(d):
    info = {}
    info["num_nodes"] = int(d.num_nodes) if getattr(d, "num_nodes", None) is not None else None
    info["x"] = tuple(d.x.shape) if getattr(d, "x", None) is not None else None
    info["edge_index"] = tuple(d.edge_index.shape) if getattr(d, "edge_index", None) is not None else None
    info["edge_attr"] = tuple(d.edge_attr.shape) if getattr(d, "edge_attr", None) is not None else None
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    samples = torch.load(args.pt, weights_only=False)
    k = min(args.k, len(samples))

    print(f"Loaded {len(samples)} samples from {args.pt}\n")

    for i in range(k):
        s = samples[i]
        print(f"sample[{i}]: rule={int(s['rule_id'])}, label={int(s['label'])}")
        for name in ["symbol", "debruijn", "dependency"]:
            if name in s:
                print(f"  {name}: {summarize_data(s[name])}")
        print()


if __name__ == "__main__":
    main()
