import numpy as np
import json
import sqlite3
import os
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime, timezone

# Create output directory
OUTPUT_DIR = "CA_2D_3x3_Classification_DB"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

DB_PATH = os.path.join(OUTPUT_DIR, "classification.db")
JSON_PATH = os.path.join(OUTPUT_DIR, "classification_db.json")


def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create classes table
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

    # Create rules table
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

    # Create metadata table
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

    # Create indices for faster queries
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

    # Get max rule that is actually a canonical rule (more reliable indicator)
    cursor.execute('''
                   SELECT MAX(CAST(canonical_rule AS INTEGER))
                   FROM classes
                   WHERE CAST(canonical_rule AS INTEGER) < 1000000000
                   ''')
    max_canonical = cursor.fetchone()[0]

    # Also check metadata for last classified rule
    cursor.execute("SELECT value FROM metadata WHERE key = 'last_classified_rule'")
    metadata_result = cursor.fetchone()
    last_classified = int(metadata_result[0]) if metadata_result else -1

    conn.close()

    return {
        'num_classes': num_classes,
        'num_rules': num_rules,
        'max_canonical': max_canonical if max_canonical else -1,
        'last_classified': last_classified
    }


def get_rule_lookup_2d(rule_number, n_states=512):
    """Convert rule number to lookup table"""
    binary = format(rule_number, f'0{n_states}b')
    lookup = {}
    for i in range(n_states):
        config = format(i, '09b')
        lookup[config] = int(binary[n_states - 1 - i])
    return lookup


def config_to_3x3(config_str):
    """Convert 9-bit binary string to 3x3 numpy array"""
    return np.array([int(config_str[i]) for i in range(9)]).reshape(3, 3)


def array_3x3_to_config(arr):
    """Convert 3x3 numpy array to 9-bit binary string"""
    return ''.join(str(int(arr.flatten()[i])) for i in range(9))


def apply_rotation_90(arr):
    return np.rot90(arr, 1)


def apply_rotation_180(arr):
    return np.rot90(arr, 2)


def apply_rotation_270(arr):
    return np.rot90(arr, 3)


def apply_flip_lr(arr):
    return np.fliplr(arr)


def apply_flip_ud(arr):
    return np.flipud(arr)


def apply_transpose(arr):
    return arr.T


def apply_antitranspose(arr):
    return np.flipud(arr.T)


def transform_rule_lookup(rule_lookup, transform_func):
    """Apply a spatial transformation to the entire rule lookup table"""
    new_lookup = {}
    for config_str, output in rule_lookup.items():
        arr = config_to_3x3(config_str)
        transformed_arr = transform_func(arr)
        transformed_config = array_3x3_to_config(transformed_arr)
        new_lookup[transformed_config] = output
    return new_lookup


def invert_rule_lookup(rule_lookup):
    """Apply state inversion (complementary)"""
    inverted = {}
    for config, output in rule_lookup.items():
        inverted_config = ''.join('1' if c == '0' else '0' for c in config)
        inverted[inverted_config] = 1 - output
    return inverted


def rule_lookup_to_number(rule_lookup):
    """Convert rule lookup to rule number"""
    binary_str = ''
    for i in range(512):
        config = format(i, '09b')
        binary_str += str(rule_lookup[config])
    return int(binary_str[::-1], 2)


def get_all_equivalent_rules(rule_number):
    """Get all equivalent rules through symmetry transformations"""
    rule_lookup = get_rule_lookup_2d(rule_number)

    transformations = [
        lambda x: x,
        apply_flip_lr,
        apply_flip_ud,
        apply_rotation_90,
        apply_rotation_180,
        apply_rotation_270,
        apply_transpose,
        apply_antitranspose,
    ]

    equivalent_rules = set()

    # Apply all spatial transformations
    for transform_func in transformations:
        transformed_lookup = transform_rule_lookup(rule_lookup, transform_func)
        rule_num = rule_lookup_to_number(transformed_lookup)
        equivalent_rules.add(rule_num)

    # Apply state inversion to all spatial transforms
    for transform_func in transformations:
        transformed_lookup = transform_rule_lookup(rule_lookup, transform_func)
        inverted_lookup = invert_rule_lookup(transformed_lookup)
        rule_num = rule_lookup_to_number(inverted_lookup)
        equivalent_rules.add(rule_num)

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
    """Count how many rules in the range [start, end] are already classified"""
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

    # Check how many rules in the target range are already classified
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
            else:
                # Update class size
                cursor.execute('''
                               UPDATE classes
                               SET class_size = (SELECT COUNT(*) FROM rules WHERE class_id = ?) + ?
                               WHERE class_id = ?
                               ''', (class_id, len(equivalent_rules), class_id))

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

    # Print class distribution BEFORE closing connection
    print_class_distribution(conn)

    # NOW close the connection
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

    # Get all classes
    cursor.execute('SELECT class_id, canonical_rule, class_size FROM classes ORDER BY class_id')
    classes = {}

    for class_id, canonical_rule, class_size in cursor.fetchall():
        # Get all members of this class
        cursor.execute('SELECT rule_number FROM rules WHERE class_id = ? ORDER BY CAST(rule_number AS INTEGER)',
                       (class_id,))
        members = [int(row[0]) for row in cursor.fetchall()]

        classes[str(class_id)] = {
            'canonical': int(canonical_rule),
            'members': members,
            'class_size': class_size
        }

    # Get rule to class mapping
    cursor.execute('SELECT rule_number, class_id FROM rules')
    rule_to_class = {row[0]: row[1] for row in cursor.fetchall()}

    # Create JSON structure
    db_json = {
        'classes': classes,
        'rule_to_class': rule_to_class,
        'next_class_id': len(classes),
        'exported_at': datetime.now(timezone.utc).isoformat()
    }

    # Save to file
    with open(JSON_PATH, 'w') as f:
        json.dump(db_json, f, indent=2)

    conn.close()


def print_class_distribution(conn=None):
    """Print distribution of class sizes"""
    should_close = False
    if conn is None:
        conn = sqlite3.connect(DB_PATH)
        should_close = True

    cursor = conn.cursor()
    cursor.execute('''
                   SELECT class_size, COUNT(*) as count
                   FROM classes
                   GROUP BY class_size
                   ORDER BY class_size
                   ''')

    print(f"\nClass Size Distribution:")
    results = cursor.fetchall()
    if results:
        for size, count in results:
            print(f"  Size {size:2d}: {count:,} classes")
    else:
        print("  No classes found yet.")

    if should_close:
        conn.close()


def query_by_rule(rule_number):
    """Query class information by rule number"""
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

    # Get class information
    cursor.execute('SELECT canonical_rule, class_size FROM classes WHERE class_id = ?',
                   (class_id,))
    canonical_rule, class_size = cursor.fetchone()

    # Get all members
    cursor.execute('SELECT rule_number FROM rules WHERE class_id = ? ORDER BY CAST(rule_number AS INTEGER)',
                   (class_id,))
    members = [int(row[0]) for row in cursor.fetchall()]

    conn.close()

    # Print information
    print(f"\n{'=' * 80}")
    print(f"Query Result: Rule {rule_number}")
    print(f"{'=' * 80}")
    print(f"Class ID: {class_id}")
    print(f"Canonical Rule: {canonical_rule}")
    print(f"Is Canonical: {'✓ Yes' if is_canonical else '✗ No'}")
    print(f"Class Size: {class_size} equivalent rules")
    print(f"\nAll Equivalent Rules in Class {class_id}:")

    # Print members in groups of 10
    for i in range(0, len(members), 10):
        group = members[i:i + 10]
        # Highlight the queried rule
        formatted = []
        for m in group:
            if m == rule_number:
                formatted.append(f"[{m}]")  # Highlight with brackets
            else:
                formatted.append(str(m))
        print(f"  {', '.join(formatted)}")

    print(f"{'=' * 80}")


def query_by_class(class_id):
    """Query all rules in a specific class"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if class exists
    cursor.execute('SELECT canonical_rule, class_size, created_at FROM classes WHERE class_id = ?',
                   (class_id,))
    result = cursor.fetchone()

    if not result:
        print(f"\n❌ Class {class_id} does not exist in the database.")
        conn.close()
        return

    canonical_rule, class_size, created_at = result

    # Get all members
    cursor.execute('''
                   SELECT rule_number, is_canonical, classified_at
                   FROM rules
                   WHERE class_id = ?
                   ORDER BY CAST(rule_number AS INTEGER)
                   ''', (class_id,))
    members = cursor.fetchall()

    conn.close()

    # Print information
    print(f"\n{'=' * 80}")
    print(f"Query Result: Class {class_id}")
    print(f"{'=' * 80}")
    print(f"Canonical Rule: {canonical_rule}")
    print(f"Class Size: {class_size} rules")
    print(f"Created At: {created_at}")
    print(f"\nAll Rules in this Class:")
    print(f"{'-' * 80}")

    # Print detailed table
    print(f"{'Rule Number':<20} {'Type':<15} {'Classified At':<30}")
    print(f"{'-' * 80}")

    for rule_num, is_canonical, classified_at in members:
        rule_type = "★ CANONICAL" if is_canonical else "  Equivalent"
        print(f"{rule_num:<20} {rule_type:<15} {classified_at:<30}")

    print(f"{'-' * 80}")
    print(f"Total: {len(members)} rules")
    print(f"{'=' * 80}")


def list_all_classes(limit=50):
    """List all classes in the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM classes')
    total_classes = cursor.fetchone()[0]

    if total_classes == 0:
        print("\n❌ No classes found in database. Run classification first.")
        conn.close()
        return

    cursor.execute('''
                   SELECT class_id, canonical_rule, class_size
                   FROM classes
                   ORDER BY class_id LIMIT ?
                   ''', (limit,))

    classes = cursor.fetchall()
    conn.close()

    print(f"\n{'=' * 80}")
    print(f"All Classes in Database (Showing {min(limit, total_classes)} of {total_classes})")
    print(f"{'=' * 80}")
    print(f"{'Class ID':<12} {'Canonical Rule':<20} {'Class Size':<15}")
    print(f"{'-' * 80}")

    for class_id, canonical_rule, class_size in classes:
        print(f"{class_id:<12} {canonical_rule:<20} {class_size:<15}")

    print(f"{'-' * 80}")

    if total_classes > limit:
        print(f"\n... and {total_classes - limit} more classes")
        print(f"Use query option to see specific classes")

    print(f"{'=' * 80}")


def query_menu():
    """Interactive query menu"""
    while True:
        print(f"\n{'=' * 80}")
        print("Query Options:")
        print("  1. Query by Rule Number (find which class a rule belongs to)")
        print("  2. Query by Class ID (see all rules in a class)")
        print("  3. List All Classes (overview)")
        print("  4. Back to Main Menu")
        print(f"{'=' * 80}")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == '1':
            try:
                rule_num = int(input("\nEnter rule number: "))
                query_by_rule(rule_num)
            except ValueError:
                print("❌ Invalid rule number!")

        elif choice == '2':
            try:
                class_id = int(input("\nEnter class ID: "))
                query_by_class(class_id)
            except ValueError:
                print("❌ Invalid class ID!")

        elif choice == '3':
            try:
                limit_input = input("\nShow how many classes? (default 50): ").strip()
                limit = int(limit_input) if limit_input else 50
                list_all_classes(limit)
            except ValueError:
                print("❌ Invalid number!")

        elif choice == '4':
            break

        else:
            print("❌ Invalid choice!")


def main():
    print("=" * 80)
    print("2D 3x3 CA Rule Classification System")
    print("=" * 80)
    print(f"Output directory: ./{OUTPUT_DIR}/")
    print("\nClassification based on:")
    print("  • Spatial symmetries (8 transformations)")
    print("  • State inversion (complementary)")
    print("  • Maximum equivalence class size: 16 rules")
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
            print("Main Menu:")
            print("  1. Classify rules (from 0 to specified rule number)")
            print("  2. Query database (search by rule or class)")
            print("  3. Exit")
            print("=" * 80)

            choice = input("\nEnter choice (1-3): ").strip()

            if choice == '1':
                end_rule = int(input("\nClassify from rule 0 to: "))
                if end_rule < 0:
                    print("❌ Invalid rule number!")
                    continue
                classify_rules_batch(0, end_rule)

            elif choice == '2':
                query_menu()

            elif choice == '3':
                print("\n👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice!")

    except ValueError as e:
        print(f"\n❌ Invalid input: {e}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted! Progress has been saved to database.")


if __name__ == "__main__":
    main()