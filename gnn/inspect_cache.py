import argparse
import torch

def summarize_data(d):
    # d 是 PyG Data
    info = {}
    info["num_nodes"] = int(d.num_nodes) if getattr(d, "num_nodes", None) is not None else None
    info["x"] = tuple(d.x.shape) if getattr(d, "x", None) is not None else None
    info["edge_index"] = tuple(d.edge_index.shape) if getattr(d, "edge_index", None) is not None else None
    info["edge_attr"] = tuple(d.edge_attr.shape) if getattr(d, "edge_attr", None) is not None else None
    return info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    samples = torch.load(args.pt, weights_only=False)
    k = min(args.k, len(samples))

    print(f"Loaded {len(samples)} samples from {args.pt}")
    print(f"Showing first {k} samples:\n")

    for i in range(k):
        s = samples[i]
        keys = list(s.keys())
        print(f"=== sample[{i}] keys = {keys}")
        print(f"  rule_id: {int(s['rule_id'])} | label: {int(s['label'])}")

        for name in ["symbol", "lattice", "debruijn", "dependency"]:
            d = s[name]
            info = summarize_data(d)
            print(f"  {name}: {info}")

        if "features" in s:
            print(f"  features: {tuple(s['features'].shape)}")
        if "prediction_entropy" in s:
            print(f"  prediction_entropy: {float(s['prediction_entropy'])}")

        print()

if __name__ == "__main__":
    main()