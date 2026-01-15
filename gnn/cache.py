import os
import argparse
import torch
from tqdm import tqdm

from .dataset import CAMultiViewDataset

def cache_dataset_to_pt(csv_path, out_pt, lattice_N=8, dependency_T=4, dependency_W=7):
    ds = CAMultiViewDataset(
        csv_path,
        split_indices=None,
        lattice_N=lattice_N,
        dependency_T=dependency_T,
        dependency_W=dependency_W,
    )

    samples = []
    for i in tqdm(range(len(ds)), desc="Caching"):
        s = ds[i]
        item = {
            "symbol": s["symbol"],
            "lattice": s["lattice"],
            "debruijn": s["debruijn"],
            "dependency": s["dependency"],
            "label": s["label"],
            "rule_id": s["rule_id"],
        }
        if "features" in s:
            item["features"] = s["features"]
        if "prediction_entropy" in s:
            item["prediction_entropy"] = s["prediction_entropy"]

        samples.append(item)

    os.makedirs(os.path.dirname(out_pt) or ".", exist_ok=True)
    torch.save(samples, out_pt)
    print(f"\n✅ Saved {len(samples)} samples to: {out_pt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV dataset")
    ap.add_argument("--out", required=True, help="Output .pt path")
    ap.add_argument("--N", type=int, default=8, help="lattice_N")
    ap.add_argument("--T", type=int, default=4, help="dependency_T")
    ap.add_argument("--W", type=int, default=7, help="dependency_W")
    args = ap.parse_args()

    cache_dataset_to_pt(args.csv, args.out, lattice_N=args.N, dependency_T=args.T, dependency_W=args.W)


if __name__ == "__main__":
    main()