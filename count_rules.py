import os
import glob
import json
import argparse


def main(args):
    paths = sorted(glob.glob(os.path.join(args.classification_dir, "*.json")))
    if not paths:
        raise FileNotFoundError(f"No json files in {args.classification_dir}")

    total_records = 0
    unique_rule_ids = set()

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            continue

        total_records += len(data)

        for r in data:
            unique_rule_ids.add(int(r["ruleId"]))

    print("=" * 60)
    print("Rule statistics")
    print("=" * 60)
    print(f"JSON files found          : {len(paths)}")
    print(f"Total rule records        : {total_records} (with duplicates)")
    print(f"Unique ruleIds (deduped)  : {len(unique_rule_ids)}")
    print(f"Duplicate records         : {total_records - len(unique_rule_ids)}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classification-dir",
        default="classification_data",
        help="Directory with JSON classification files"
    )
    args = parser.parse_args()
    main(args)