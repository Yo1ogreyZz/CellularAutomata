"""Cache PyG graph data to .pt file for faster loading."""

import os
import argparse
import torch
from tqdm import tqdm


def cache_from_csv(csv_path, out_pt, dependency_T=4, dependency_W=7):
    """Generate PyG graphs from CSV and save to .pt file."""
    from .dataset import CAMultiViewDataset
    
    ds = CAMultiViewDataset(
        csv_path,
        split_indices=None,
        dependency_T=dependency_T,
        dependency_W=dependency_W,
    )

    samples = []
    for i in tqdm(range(len(ds)), desc="Generating graphs"):
        s = ds[i]
        samples.append({
            "symbol": s["symbol"],
            "debruijn": s["debruijn"],
            "dependency": s["dependency"],
            "label": s["label"],
            "rule_id": s["rule_id"],
        })

    os.makedirs(os.path.dirname(out_pt) or ".", exist_ok=True)
    torch.save(samples, out_pt)
    print(f"Saved {len(samples)} samples to: {out_pt}")


def load_cached(pt_path):
    """Load cached PyG data from .pt file."""
    samples = torch.load(pt_path, weights_only=False)
    print(f"Loaded {len(samples)} samples from: {pt_path}")
    return samples


def main():
    ap = argparse.ArgumentParser(description="Cache PyG graphs to .pt file")
    ap.add_argument("--csv", required=True, help="Input CSV dataset")
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument("--T", type=int, default=4, help="dependency_T")
    ap.add_argument("--W", type=int, default=7, help="dependency_W")
    args = ap.parse_args()

    cache_from_csv(args.csv, args.out, dependency_T=args.T, dependency_W=args.W)


if __name__ == "__main__":
    main()
