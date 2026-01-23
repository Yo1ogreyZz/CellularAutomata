#!/usr/bin/env python3
"""
Convert JSON classification data to CSV format for GNN training.

Function:
    1. load all json files in the input directory
    2. convert each json file to a pandas dataframe
    3. merge all dataframes into a single dataframe
    4. save the dataframe to a csv file

Usage:
    python json_to_csv.py -d . -o dataset.csv
    python json_to_csv.py -d . -o dataset.csv --include-ic  # include IC subtypes
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict


# Mapping from JSON className to 5-class labels
CLASS_MAPPING = {
    'Class1-Homogeneous': 'Homogeneous',
    'Class2-Stable': 'Stable',
    'Class2-Propagate': 'Propagate',
    'Class3-Chaotic': 'Chaotic',
    'Class4-Complex': 'Complex',
    # IC subtypes - map to Stable by default (can be changed)
    'Class2-IC-ConvTime': 'Propagate',
    'Class2-IC-Period': 'Propagate',
    'Class2-IC-Regime': 'Propagate',
}


def load_all_json(input_dir):
    records = []
    for f in sorted(Path(input_dir).glob('ca-classifications*.json')):
        with open(f) as fp:
            data = json.load(fp)
            records.extend(data)
        print(f"Loaded {f.name}: {len(data)} records")
    return records


def convert_to_csv(records, include_ic=False):
    """Convert JSON records to DataFrame with proper format."""
    
    # Group by ruleId, check for conflicts
    rule_groups = defaultdict(list)
    for r in records:
        rule_groups[r['ruleId']].append(r)
    
    rows = []
    conflicts = []
    skipped_ic = 0
    
    for rule_id, items in rule_groups.items():
        class_names = set(item['className'] for item in items)
        
        # Check for conflicts (same rule, different classes)
        if len(class_names) > 1:
            conflicts.append((rule_id, class_names))
            print(f'find conflicts for {rule_id}')
            continue
        
        raw_class = items[0]['className']
        
        # Skip IC types if not including
        if not include_ic and raw_class.startswith('Class2-IC'):
            skipped_ic += 1
            continue
        
        # Map to 5-class
        if raw_class not in CLASS_MAPPING:
            print(f"Warning: Unknown class {raw_class}, skipping")
            continue
            
        mapped_class = CLASS_MAPPING[raw_class]
        
        rows.append({
            'ruleId': rule_id,
            'className': mapped_class,
        })
    
    if conflicts:
        print(f"\nSkipped {len(conflicts)} rules with conflicting labels")
    if skipped_ic:
        print(f"Skipped {skipped_ic} IC-subtype rules (use --include-ic to include)")
    
    df = pd.DataFrame(rows)
    
    # Create count columns (for compatibility with existing dataset.py)
    # Since we have single labels, set count=1 for the class
    for cls in ['Homogeneous', 'Stable', 'Propagate', 'Chaotic', 'Complex']:
        df[f'count_{cls}'] = (df['className'] == cls).astype(int)
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--input-dir', default='.', help='Directory with JSON files')
    parser.add_argument('-o', '--output', default='merged_dataset.csv', help='Output CSV path')
    parser.add_argument('--include-ic', action='store_true', help='Include IC subtypes (mapped to Stable)')
    args = parser.parse_args()
    
    records = load_all_json(args.input_dir)
    print(f"\nTotal records: {len(records)}")
    
    df = convert_to_csv(records, include_ic=args.include_ic)
    df.to_csv(args.output, index=False)

    print(f"Final samples: {len(df)}")
    print("\nClass distribution:")
    print(df['className'].value_counts())

if __name__ == '__main__':
    main()
