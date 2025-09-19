import csv
import math


def rule_to_base_k(rule_number: int, R: int, K: int) -> list[int]:
    """
    Converts a decimal rule number to its base-K lookup table representation.

    Example: for R=1, K=2, Rule 30 -> [0, 0, 0, 1, 1, 1, 1, 0]
    The list index corresponds to the base-K value of the neighborhood (e.g., '111' -> 7, '000' -> 0).
    """
    num_neighborhoods = K ** (2 * R + 1)
    if rule_number >= K ** num_neighborhoods:
        raise ValueError(f"Rule number {rule_number} is out of bounds.")

    base_k_digits = []
    temp_num = rule_number
    for _ in range(num_neighborhoods):
        base_k_digits.append(temp_num % K)
        temp_num //= K

    return base_k_digits[::-1]


def base_k_to_rule(rule_table: list[int], K: int) -> int:
    """
    Converts a base-K lookup table back to its decimal rule number.
    """
    rule_number = 0
    for digit in rule_table:
        rule_number = rule_number * K + digit
    return rule_number


def apply_mirror(rule_table: list[int], R: int, K: int) -> list[int]:
    """
    Applies the "left-right reflection" (mirror) symmetry to a rule table.
    R_mirror(xyz) = R(zyx)
    """
    neighborhood_size = 2 * R + 1
    num_neighborhoods = len(rule_table)
    mirrored_table = [0] * num_neighborhoods

    for i in range(num_neighborhoods):
        # Convert index i to its neighborhood representation
        neighborhood = []
        temp_i = i
        for _ in range(neighborhood_size):
            neighborhood.append(temp_i % K)
            temp_i //= K
        neighborhood = neighborhood[::-1]

        # Flip the neighborhood
        mirrored_neighborhood = neighborhood[::-1]

        # Calculate the index of the flipped neighborhood
        mirrored_index = 0
        for digit in mirrored_neighborhood:
            mirrored_index = mirrored_index * K + digit

        # The value at the mirrored_index of the new table is the value at index i of the old table
        mirrored_table[mirrored_index] = rule_table[i]

    return mirrored_table


def apply_complement(rule_table: list[int], R: int, K: int) -> list[int]:
    """
    Applies the "state inversion" (complementary) symmetry to a rule table.
    R_comp(xyz) = NOT( R(NOT x, NOT y, NOT z) )
    """
    neighborhood_size = 2 * R + 1
    num_neighborhoods = len(rule_table)
    complemented_table = [0] * num_neighborhoods

    for i in range(num_neighborhoods):
        # 1. Invert the current neighborhood's input
        # The index of the inverted neighborhood of i is (num_neighborhoods - 1 - i)
        complemented_input_index = num_neighborhoods - 1 - i

        # 2. Look up the output of the old rule for this inverted input
        original_output = rule_table[complemented_input_index]

        # 3. Invert this output
        complemented_output = (K - 1) - original_output

        # 4. Store it in the new table
        complemented_table[i] = complemented_output

    return complemented_table


def classify_rules(R: int, K: int):
    """
    Classifies all 1D cellular automata rules for a given R and K.
    """
    if not (R >= 0 and K > 1):
        raise ValueError("R must be non-negative and K must be greater than 1.")

    num_rules = K ** (K ** (2 * R + 1))

    classified = [False] * num_rules
    equivalence_classes = []

    print(f"Classifying {num_rules} rules for R={R}, K={K}...")

    for i in range(num_rules):
        if not classified[i]:
            # Found a new, unclassified rule
            new_class = set()

            # 1. Original rule
            original_table = rule_to_base_k(i, R, K)
            new_class.add(i)

            # 2. Mirrored version
            mirrored_table = apply_mirror(original_table, R, K)
            mirrored_rule = base_k_to_rule(mirrored_table, K)
            new_class.add(mirrored_rule)

            # 3. Complemented version
            complemented_table = apply_complement(original_table, R, K)
            complemented_rule = base_k_to_rule(complemented_table, K)
            new_class.add(complemented_rule)

            # 4. Mirrored and complemented version
            mirrored_complemented_table = apply_complement(mirrored_table, R, K)
            mirrored_complemented_rule = base_k_to_rule(mirrored_complemented_table, K)
            new_class.add(mirrored_complemented_rule)

            # Mark all rules in this newly found equivalence class as classified
            for rule_in_class in new_class:
                if rule_in_class < num_rules:
                    classified[rule_in_class] = True

            equivalence_classes.append(sorted(list(new_class)))

    return equivalence_classes


def main():
    """
    Main function: sets parameters, runs the classification, and outputs the results.
    """
    # --- Parameters ---
    R = 2  # Neighborhood radius
    K = 2  # Number of states
    # ------------------

    # Run the classification
    classes = classify_rules(R, K)

    # Print results to the command line
    print("\n--- Equivalence Class Classification Result ---")
    print(f"Total number of unique equivalence classes: {len(classes)}")
    for idx, eq_class in enumerate(classes):
        print(f"Class {idx + 1} (Size: {len(eq_class)}): {eq_class}")

    # Save results to a CSV file
    csv_filename = f'rule_classification_R{R}_K{K}.csv'
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Class_ID', 'Class_Size', 'Rules_In_Class'])
            for idx, eq_class in enumerate(classes):
                # Convert the list of rules to a semicolon-separated string
                rules_str = "; ".join(map(str, eq_class))
                writer.writerow([idx + 1, len(eq_class), rules_str])
        print(f"\nClassification results successfully saved to '{csv_filename}'")
    except IOError as e:
        print(f"\nError saving to CSV file: {e}")


if __name__ == '__main__':
    main()