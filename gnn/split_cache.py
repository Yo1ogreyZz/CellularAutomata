import os
import argparse
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Input ca.pt")
    ap.add_argument("--out_dir", required=True, help="Output folder")
    args = ap.parse_args()

    samples = torch.load(args.pt, weights_only=False)

    os.makedirs(args.out_dir, exist_ok=True)

    symbol_list = []
    lattice_list = []
    debruijn_list = []
    dependency_list = []
    labels = []
    rule_ids = []
    extras = []

    for s in samples:
        symbol_list.append(s["symbol"])
        lattice_list.append(s["lattice"])
        debruijn_list.append(s["debruijn"])
        dependency_list.append(s["dependency"])
        labels.append(s["label"])
        rule_ids.append(s["rule_id"])

        # 可选字段统一丢进 extras
        e = {}
        if "features" in s: e["features"] = s["features"]
        if "prediction_entropy" in s: e["prediction_entropy"] = s["prediction_entropy"]
        extras.append(e)

    torch.save(symbol_list, os.path.join(args.out_dir, "symbol.pt"))
    torch.save(lattice_list, os.path.join(args.out_dir, "lattice.pt"))
    torch.save(debruijn_list, os.path.join(args.out_dir, "debruijn.pt"))
    torch.save(dependency_list, os.path.join(args.out_dir, "dependency.pt"))
    torch.save({"label": labels, "rule_id": rule_ids, "extras": extras},
               os.path.join(args.out_dir, "meta.pt"))

    print("✅ Saved:")
    print(" - symbol.pt")
    print(" - lattice.pt")
    print(" - debruijn.pt")
    print(" - dependency.pt")
    print(" - meta.pt (label/rule_id/optional extras)")

if __name__ == "__main__":
    main()