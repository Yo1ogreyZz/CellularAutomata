import json
import csv


# ==============================================================================
#  We reuse the core transformation functions from classify_rules.py
#  to accurately determine the relationship between any two rules.
# ==============================================================================

def rule_to_base_k(rule_number: int, R: int, K: int) -> list[int]:
    """Converts a decimal rule number to its base-K lookup table representation."""
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
    """Converts a base-K lookup table back to its decimal rule number."""
    rule_number = 0
    for digit in rule_table:
        rule_number = rule_number * K + digit
    return rule_number


def apply_mirror(rule_table: list[int], R: int, K: int) -> list[int]:
    """Applies the 'left-right reflection' (mirror) symmetry to a rule table."""
    neighborhood_size = 2 * R + 1
    num_neighborhoods = len(rule_table)
    mirrored_table = [0] * num_neighborhoods
    for i in range(num_neighborhoods):
        neighborhood = []
        temp_i = i
        for _ in range(neighborhood_size):
            neighborhood.append(temp_i % K)
            temp_i //= K
        neighborhood = neighborhood[::-1]
        mirrored_neighborhood = neighborhood[::-1]
        mirrored_index = 0
        for digit in mirrored_neighborhood:
            mirrored_index = mirrored_index * K + digit
        mirrored_table[mirrored_index] = rule_table[i]
    return mirrored_table


def apply_complement(rule_table: list[int], R: int, K: int) -> list[int]:
    """Applies the 'state inversion' (complementary) symmetry to a rule table."""
    num_neighborhoods = len(rule_table)
    complemented_table = [0] * num_neighborhoods
    for i in range(num_neighborhoods):
        complemented_input_index = num_neighborhoods - 1 - i
        original_output = rule_table[complemented_input_index]
        complemented_output = (K - 1) - original_output
        complemented_table[i] = complemented_output
    return complemented_table


def create_graph_data(R: int, K: int):
    """
    Processes rule classification to generate graph data (nodes and edges)
    suitable for GNN training.
    """
    num_rules = K ** (K ** (2 * R + 1))

    # --- 1. Create Nodes ---
    # Each rule is a node in our graph.
    nodes = [{'id': i} for i in range(num_rules)]

    # --- 2. Create Edges ---
    # Edges represent the symmetry transformations between rules.
    edges = []

    print(f"Generating graph data for R={R}, K={K}...")

    for i in range(num_rules):
        original_table = rule_to_base_k(i, R, K)

        # Mirror Edge
        mirrored_table = apply_mirror(original_table, R, K)
        mirrored_rule = base_k_to_rule(mirrored_table, K)
        # To avoid duplicate edges like (A,B) and (B,A), we only add one.
        # Self-loops (A,A) are important and are kept.
        if i <= mirrored_rule:
            edges.append({
                'source': i,
                'target': mirrored_rule,
                'type': 'mirror'
            })

        # Complementary Edge
        complemented_table = apply_complement(original_table, R, K)
        complemented_rule = base_k_to_rule(complemented_table, K)
        if i <= complemented_rule:
            edges.append({
                'source': i,
                'target': complemented_rule,
                'type': 'complement'
            })

    graph_data = {
        'parameters': {'R': R, 'K': K},
        'nodes': nodes,
        'edges': edges
    }

    return graph_data


def main():
    """
    Main function to run the data processing and save the output.
    """
    # --- Parameters ---
    R = 1
    K = 2
    # ------------------

    # Generate the graph-structured data
    graph_data = create_graph_data(R, K)

    # Save the data to a JSON file
    output_filename = f'graph_data_R{R}_K{K}.json'
    try:
        with open(output_filename, 'w') as f:
            json.dump(graph_data, f, indent=4)
        print(f"\nGraph data successfully saved to '{output_filename}'")

        # Explain the content of the saved file
        print("\n--- Structure of the saved JSON file ---")
        print(f"Total nodes: {len(graph_data['nodes'])}")
        print(f"Total edges: {len(graph_data['edges'])}")
        print("\nSaved Keys and their meaning:")
        print("  - 'parameters': Contains the R and K values for this dataset.")
        print("  - 'nodes': A list of all rules, treated as nodes in the graph.")
        print("    - 'id': The rule number (e.g., 30).")
        print("  - 'edges': A list of all transformation relationships.")
        print("    - 'source': The starting rule number of the edge.")
        print("    - 'target': The ending rule number of the edge.")
        print("    - 'type': The type of transformation ('mirror' or 'complement').")
        print("\nThis format is ideal for loading into GNN libraries like PyTorch Geometric or DGL.")

    except IOError as e:
        print(f"\nError saving to JSON file: {e}")


if __name__ == '__main__':
    main()