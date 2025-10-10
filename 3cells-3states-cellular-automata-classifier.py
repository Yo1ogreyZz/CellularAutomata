import numpy as np
import json
import sqlite3
import os
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime, timezone
from itertools import permutations

# Create output directory
OUTPUT_DIR = "CA_1D_3cells_3states_Classification_DB"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

DB_PATH = os.path.join(OUTPUT_DIR, "classification.db")
JSON_PATH = os.path.join(OUTPUT_DIR, "classification_db.json")


def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS classes
                   (
                       class_id
                       INTEGER
                       PRIMARY
                       KEY,
                       canonical_rule
                       TEXT
                       NOT
                       NULL,
                       class_size
                       INTEGER
                       NOT
                       NULL,
                       created_at
                       TIMESTAMP
                       DEFAULT
                       CURRENT_TIMESTAMP
                   )
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS rules
                   (
                       rule_number
                       TEXT
                       PRIMARY
                       KEY,
                       class_id
                       INTEGER
                       NOT
                       NULL,
                       is_canonical
                       INTEGER
                       DEFAULT
                       0,
                       classified_at
                       TIMESTAMP
                       DEFAULT
                       CURRENT_TIMESTAMP,
                       FOREIGN
                       KEY
                   (
                       class_id
                   ) REFERENCES classes
                   (
                       class_id
                   )
                       )
                   ''')

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS metadata
                   (
                       key
                       TEXT
                       PRIMARY
                       KEY,
                       value
                       TEXT
                       NOT
                       NULL,
                       updated_at
                       TIMESTAMP
                       DEFAULT
                       CURRENT_TIMESTAMP
                   )
                   ''')

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_rule_class ON rules(class_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_canonical ON rules(is_canonical)')

    conn.commit()
    conn.close()


def get_database_stats():
    """Get current database statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM classes')
    num_classes = cursor.fetchone()[0]

    cursor.execute('SELECT COUNT(*) FROM rules')
    num_rules = cursor.fetchone()[0]

    cursor.execute("SELECT value FROM metadata WHERE key = 'last_classified_rule'")
    metadata_result = cursor.fetchone()
    last_classified = int(metadata_result[0]) if metadata_result else -1

    conn.close()

    return {
        'num_classes': num_classes,
        'num_rules': num_rules,
        'last_classified': last_classified
    }


def get_rule_lookup(rule_number, n_configs=27):
    """
    Convert rule number to lookup table
    For 3-state 3-cells: 3^3 = 27 configurations
    Rule expressed in base-3 (ternary)
    """
    # Convert to ternary (base-3) with 27 digits
    ternary = np.base_repr(rule_number, base=3).zfill(n_configs)

    lookup = {}
    for i in range(n_configs):
        # Configuration in ternary: LCR (3 cells)
        config = np.base_repr(i, base=3).zfill(3)
        lookup[config] = int(ternary[n_configs - 1 - i])

    return lookup


def apply_mirror(config_str):
    """
    Apply left-right mirror
    For 3 cells: LCR -> RCL
    """
    return config_str[::-1]


def apply_state_permutation(config_str, perm):
    """
    Apply state permutation
    perm: tuple like (0,1,2) or (2,0,1) etc.
    """
    return ''.join(str(perm[int(c)]) for c in config_str)


def transform_rule_lookup(rule_lookup, transform_func):
    """Apply a transformation to the entire rule lookup table"""
    new_lookup = {}
    for config_str, output in rule_lookup.items():
        transformed_config = transform_func(config_str)
        new_lookup[transformed_config] = output
    return new_lookup


def apply_state_permutation_to_lookup(rule_lookup, perm):
    """
    Apply state permutation to both input AND output
    """
    new_lookup = {}
    for config, output in rule_lookup.items():
        # Transform input configuration
        new_config = apply_state_permutation(config, perm)
        # Transform output state
        new_output = perm[output]
        new_lookup[new_config] = new_output
    return new_lookup


def rule_lookup_to_number(rule_lookup):
    """Convert rule lookup to rule number (base-3)"""
    ternary_str = ''
    for i in range(27):
        config = np.base_repr(i, base=3).zfill(3)
        ternary_str += str(rule_lookup[config])

    # Convert ternary string to decimal
    return int(ternary_str[::-1], base=3)


def get_all_equivalent_rules(rule_number):
    """
    Get all equivalent rules through symmetry transformations
    - Left-right mirror (2 variations)
    - State permutations (6 variations: all permutations of 0,1,2)
    Total: 2 × 6 = 12 maximum equivalent rules
    """
    rule_lookup = get_rule_lookup(rule_number)

    equivalent_rules = set()
    equivalent_rules.add(rule_number)

    # All state permutations
    state_perms = list(permutations([0, 1, 2]))  # 6 permutations

    for perm in state_perms:
        # Apply state permutation
        perm_lookup = apply_state_permutation_to_lookup(rule_lookup, perm)
        perm_rule = rule_lookup_to_number(perm_lookup)
        equivalent_rules.add(perm_rule)

        # Apply mirror + state permutation
        mirrored_perm_lookup = transform_rule_lookup(perm_lookup, apply_mirror)
        mirrored_perm_rule = rule_lookup_to_number(mirrored_perm_lookup)
        equivalent_rules.add(mirrored_perm_rule)

    return equivalent_rules


def is_rule_classified(rule_number, conn):
    """Check if a rule is already classified"""
    cursor = conn.cursor()
    cursor.execute('SELECT class_id FROM rules WHERE rule_number = ?', (str(rule_number),))
    result = cursor.fetchone()
    return result is not None


def get_next_class_id(conn):
    """Get the next available class ID"""
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(class_id) FROM classes')
    result = cursor.fetchone()[0]
    return 0 if result is None else result + 1


def count_classified_in_range(start, end, conn):
    """Count how many rules in the range are already classified"""
    cursor = conn.cursor()
    cursor.execute('''
                   SELECT COUNT(*)
                   FROM rules
                   WHERE CAST(rule_number AS INTEGER) >= ?
                     AND CAST(rule_number AS INTEGER) <= ?
                   ''', (start, end))
    return cursor.fetchone()[0]


def classify_rules_batch(start, end):
    """Classify all rules from start to end"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check already classified
    classified_count = count_classified_in_range(start, end, conn)
    total_in_range = end - start + 1

    if classified_count == total_in_range:
        print(f"\n✓ All rules in range {start}-{end} are already classified!")
        stats = get_database_stats()
        print(f"  Database contains {stats['num_classes']:,} classes")
        print(f"  Total classified rules: {stats['num_rules']:,}")
        conn.close()
        return

    if classified_count > 0:
        print(f"\n✓ {classified_count:,} rules in range {start}-{end} already classified")
        print(f"  Will process remaining {total_in_range - classified_count:,} rules...")

    print(f"\nClassifying rules {start} to {end} ({total_in_range:,} rules)")
    print(f"{'=' * 80}\n")

    new_classes = 0
    skipped_rules = 0
    classified_rules = 0
    start_time = time.time()

    # Process with progress bar
    with tqdm(total=total_in_range, desc="Classifying", unit="rule") as pbar:
        for rule_num in range(start, end + 1):
            # Check if already classified
            if is_rule_classified(rule_num, conn):
                skipped_rules += 1
                pbar.update(1)
                continue

            # Get all equivalent rules
            equivalent_rules = get_all_equivalent_rules(rule_num)
            canonical_rule = min(equivalent_rules)

            # Check if any equivalent rule is already classified
            class_id = None
            for equiv_rule in equivalent_rules:
                cursor.execute('SELECT class_id FROM rules WHERE rule_number = ?',
                               (str(equiv_rule),))
                result = cursor.fetchone()
                if result:
                    class_id = result[0]
                    break

            # Create new class if needed
            if class_id is None:
                class_id = get_next_class_id(conn)
                cursor.execute('''
                               INSERT INTO classes (class_id, canonical_rule, class_size)
                               VALUES (?, ?, ?)
                               ''', (class_id, str(canonical_rule), len(equivalent_rules)))
                new_classes += 1

            # Add all equivalent rules to database
            for equiv_rule in equivalent_rules:
                if not is_rule_classified(equiv_rule, conn):
                    is_canonical = 1 if equiv_rule == canonical_rule else 0
                    cursor.execute('''
                                   INSERT
                                   OR IGNORE INTO rules (rule_number, class_id, is_canonical)
                        VALUES (?, ?, ?)
                                   ''', (str(equiv_rule), class_id, is_canonical))
                    classified_rules += 1

            # Commit every 1000 rules
            if rule_num % 1000 == 0:
                conn.commit()

            pbar.update(1)

    # Final commit
    conn.commit()

    # Update metadata
    cursor.execute('''
        INSERT OR REPLACE INTO metadata (key, value, updated_at)
        VALUES ('last_classified_rule', ?, ?)
    ''', (str(end), datetime.now(timezone.utc).isoformat()))

    stats = get_database_stats()
    cursor.execute('''
        INSERT OR REPLACE INTO metadata (key, value, updated_at)
        VALUES ('total_classes', ?, ?)
    ''', (str(stats['num_classes']), datetime.now(timezone.utc).isoformat()))

    conn.commit()
    conn.close()

    # Export to JSON
    export_to_json()

    # Print statistics
    elapsed_time = time.time() - start_time
    final_stats = get_database_stats()

    print(f"\n{'=' * 80}")
    print(f"Classification Complete!")
    print(f"{'=' * 80}")
    print(f"Time elapsed: {elapsed_time / 60:.2f} minutes ({elapsed_time:.1f} seconds)")
    if elapsed_time > 0:
        print(f"Processing speed: {total_in_range / elapsed_time:.1f} rules/second")
    print(f"\nStatistics:")
    print(f"  Rules in target range: {total_in_range:,}")
    print(f"  New rules classified: {classified_rules:,}")
    print(f"  Rules skipped (already classified): {skipped_rules:,}")
    print(f"  New classes created: {new_classes:,}")
    print(f"  Total classes in database: {final_stats['num_classes']:,}")
    print(f"  Total rules in database: {final_stats['num_rules']:,}")
    print(f"\nOutput files:")
    print(f"  Database: {DB_PATH}")
    print(f"  JSON: {JSON_PATH}")
    print(f"{'=' * 80}")


def export_to_json():
    """Export database to JSON format"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT class_id, canonical_rule, class_size FROM classes ORDER BY class_id')
    classes = {}

    for class_id, canonical_rule, class_size in cursor.fetchall():
        cursor.execute('SELECT rule_number FROM rules WHERE class_id = ? ORDER BY CAST(rule_number AS INTEGER)',
                       (class_id,))
        members = [int(row[0]) for row in cursor.fetchall()]

        classes[str(class_id)] = {
            'canonical': int(canonical_rule),
            'members': members,
            'class_size': class_size
        }

    cursor.execute('SELECT rule_number, class_id FROM rules')
    rule_to_class = {row[0]: row[1] for row in cursor.fetchall()}

    db_json = {
        'classes': classes,
        'rule_to_class': rule_to_class,
        'next_class_id': len(classes),
        'exported_at': datetime.now(timezone.utc).isoformat()
    }

    with open(JSON_PATH, 'w') as f:
        json.dump(db_json, f, indent=2)

    conn.close()


def query_rule(rule_number):
    """Query information about a specific rule"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT class_id, is_canonical FROM rules WHERE rule_number = ?',
                   (str(rule_number),))
    result = cursor.fetchone()

    if not result:
        print(f"\n❌ Rule {rule_number} has not been classified yet.")
        print(f"   Run classification first to include this rule.")
        conn.close()
        return

    class_id, is_canonical = result

    cursor.execute('SELECT canonical_rule, class_size FROM classes WHERE class_id = ?',
                   (class_id,))
    canonical_rule, class_size = cursor.fetchone()

    cursor.execute('SELECT rule_number FROM rules WHERE class_id = ? ORDER BY CAST(rule_number AS INTEGER)',
                   (class_id,))
    members = [int(row[0]) for row in cursor.fetchall()]

    conn.close()

    print(f"\n{'=' * 80}")
    print(f"Rule {rule_number} Information")
    print(f"{'=' * 80}")
    print(f"Class ID: {class_id}")
    print(f"Canonical Rule: {canonical_rule}")
    print(f"Is Canonical: {'Yes' if is_canonical else 'No'}")
    print(f"Class Size: {class_size} equivalent rules")
    print(f"\nEquivalent Rules:")

    for i in range(0, min(len(members), 50), 10):
        print(f"  {', '.join(map(str, members[i:i + 10]))}")

    if len(members) > 50:
        print(f"  ... and {len(members) - 50} more")

    print(f"{'=' * 80}")


def main():
    total_rules = 3 ** 27

    print("=" * 80)
    print("1D 3-cells 3-states CA Rule Classification System")
    print("=" * 80)
    print(f"Output directory: ./{OUTPUT_DIR}/")
    print(f"\nTotal possible rules: 3^27 = {total_rules:,}")
    print("\nClassification based on:")
    print("  • Left-right mirror symmetry")
    print("  • State permutations (6 permutations of 0,1,2)")
    print("  • Maximum equivalence class size: 12 rules")
    print("=" * 80)

    # Initialize database
    init_database()

    # Show current database status
    stats = get_database_stats()
    if stats['num_rules'] > 0:
        print(f"\nCurrent Database Status:")
        print(f"  Classes: {stats['num_classes']:,}")
        print(f"  Classified Rules: {stats['num_rules']:,}")
        if stats['last_classified'] >= 0:
            print(f"  Last Classified Range: 0 to {stats['last_classified']:,}")

    try:
        while True:
            print("\n" + "=" * 80)
            print("Options:")
            print("  1. Classify rules (from 0 to specified rule number)")
            print("  2. Query rule information")
            print("  3. Exit")
            print("=" * 80)

            choice = input("\nEnter choice (1-3): ").strip()

            if choice == '1':
                end_rule = int(input("\nClassify from rule 0 to: "))
                if end_rule < 0:
                    print("Invalid rule number!")
                    continue
                classify_rules_batch(0, end_rule)

            elif choice == '2':
                rule_num = int(input("\nEnter rule number to query: "))
                query_rule(rule_num)

            elif choice == '3':
                print("\n👋 Goodbye!")
                break

            else:
                print("Invalid choice!")

    except ValueError as e:
        print(f"\n❌ Invalid input: {e}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted! Progress has been saved to database.")


if __name__ == "__main__":
    main()