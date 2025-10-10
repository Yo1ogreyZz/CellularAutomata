import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os
from pathlib import Path

# Create output directory
OUTPUT_DIR = "CA_1D_5cells_Rules"
Path(OUTPUT_DIR).mkdir(exist_ok=True)


def get_rule_lookup(rule_number, n_states=32):
    """
    Convert rule number to lookup table
    For 5 cells, there are 2^5=32 possible neighborhood configurations
    """
    binary = format(rule_number, f'0{n_states}b')
    lookup = {}
    for i in range(n_states):
        config = format(i, '05b')  # 5-bit binary
        lookup[config] = int(binary[n_states - 1 - i])
    return lookup


def visualize_rule(rule_number):
    """
    Visualize the transition diagram of 1D 5-cell rule
    """
    lookup = get_rule_lookup(rule_number)

    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Draw title
    title = f'1D 5-Cell Cellular Automaton - Rule {rule_number}'
    ax.text(16, 2.7, title, ha='center', va='top', fontsize=16, weight='bold')

    # Draw all 32 configurations
    cell_size = 0.15
    y_position = 1.5

    for i in range(32):
        x_base = i + 0.5
        config = format(i, '05b')

        # Draw input configuration (top 5 cells)
        for j, bit in enumerate(config):
            x = x_base + (j - 2) * cell_size
            color = 'black' if bit == '1' else 'white'
            rect = Rectangle((x - cell_size / 2, y_position),
                             cell_size, cell_size,
                             facecolor=color,
                             edgecolor='black',
                             linewidth=1)
            ax.add_patch(rect)

        # Draw arrow
        ax.arrow(x_base, y_position - 0.2, 0, -0.3,
                 head_width=0.1, head_length=0.1, fc='gray', ec='gray')

        # Draw output (bottom center cell)
        output = lookup[config]
        color = 'black' if output == 1 else 'white'
        rect = Rectangle((x_base - cell_size / 2, y_position - 0.7),
                         cell_size, cell_size,
                         facecolor=color,
                         edgecolor='black',
                         linewidth=2)
        ax.add_patch(rect)

        # Label configuration number
        ax.text(x_base, y_position - 1.0, str(i),
                ha='center', va='top', fontsize=8)

    plt.tight_layout()

    # Save to the designated folder with rule number as filename
    filename = os.path.join(OUTPUT_DIR, f'rule_{rule_number}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'Image saved: {filename}')
    plt.close()


def main():
    # Calculate 2^32
    total_rules = 2 ** 32

    print("=" * 70)
    print("1D 5-Cell Cellular Automaton Rule Visualizer")
    print("=" * 70)
    print(f"Total number of possible rules: 2^32 = {total_rules:,}")
    print(f"That's exactly: 4,294,967,296 rules")
    print(f"Or in scientific notation: {total_rules:.2e}")
    print("-" * 70)
    print(f"Output directory: ./{OUTPUT_DIR}/")
    print(f"All 32 neighborhood configurations will be visualized per rule")
    print(f"Suggested rule range: 0 to 1,000,000 for practical exploration")
    print("-" * 70)

    try:
        rule_input = input("Enter rule number(s) to visualize (comma-separated): ")
        rule_numbers = [int(x.strip()) for x in rule_input.split(',')]

        for rule_num in rule_numbers:
            if 0 <= rule_num < total_rules:
                print(f"\nGenerating visualization for rule {rule_num}...")
                visualize_rule(rule_num)
            else:
                print(f"Rule {rule_num} is out of valid range (0 to {total_rules - 1})!")

        print("\n" + "=" * 70)
        print(f"All visualizations saved to: ./{OUTPUT_DIR}/")
        print("=" * 70)

    except ValueError:
        print("Invalid input format! Please enter valid numbers.")
    except KeyboardInterrupt:
        print("\n\nProgram terminated.")


if __name__ == "__main__":
    main()