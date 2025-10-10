import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
import os
from pathlib import Path

# Suppress font warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Create output directory
OUTPUT_DIR = "CA_2D_3x3_Rules"
Path(OUTPUT_DIR).mkdir(exist_ok=True)


def get_rule_lookup_2d(rule_number, n_states=512):
    """
    Convert rule number to lookup table
    For 3x3 (9 cells), there are 2^9=512 possible neighborhood configurations
    """
    binary = format(rule_number, f'0{n_states}b')
    lookup = {}
    for i in range(n_states):
        config = format(i, '09b')  # 9-bit binary
        lookup[config] = int(binary[n_states - 1 - i])
    return lookup


def visualize_rule_2d(rule_number, max_configs=512):
    """
    Visualize the transition diagram of 2D 3x3 rule
    Shows all 512 configurations by default
    """
    lookup = get_rule_lookup_2d(rule_number)

    # Calculate layout - optimize for all 512 configs
    cols = 16  # More columns for better layout
    rows = (max_configs + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()  # Flatten for easier indexing

    fig.suptitle(f'2D 3x3 Cellular Automaton - Rule {rule_number} (Showing {max_configs} configs)',
                 fontsize=20, weight='bold', y=0.995)

    for idx in range(max_configs):
        ax = axes[idx]

        config = format(idx, '09b')
        output = lookup[config]

        # Draw 3x3 grid
        grid = np.array([[int(config[i]) for i in range(j * 3, j * 3 + 3)] for j in range(3)])

        ax.set_xlim(0, 4)
        ax.set_ylim(0, 2)
        ax.axis('off')

        # Draw input configuration (top)
        for i in range(3):
            for j in range(3):
                color = 'black' if grid[i, j] == 1 else 'white'
                rect = Rectangle((j + 0.5, 1.5 - i * 0.4), 0.35, 0.35,
                                 facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(rect)

        # Draw arrow
        ax.arrow(2, 0.3, 0, -0.15, head_width=0.15, head_length=0.08,
                 fc='gray', ec='gray')

        # Draw output (bottom center cell)
        color = 'black' if output == 1 else 'white'
        rect = Rectangle((1.825, 0.0), 0.35, 0.35,
                         facecolor=color, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        # Label configuration number
        ax.text(2, -0.2, f'#{idx}', ha='center', va='top', fontsize=7)

    # Hide extra subplots
    for idx in range(max_configs, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save to the designated folder with rule number as filename
    filename = os.path.join(OUTPUT_DIR, f'rule_{rule_number}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'Image saved: {filename}')
    plt.close()


def main():
    # Calculate 2^512
    total_rules = 2 ** 512

    print("=" * 70)
    print("2D 3x3 Cellular Automaton Rule Visualizer")
    print("=" * 70)
    print(f"Total number of possible rules: 2^512 = {total_rules}")
    print(f"That's approximately {total_rules:.2e}")
    print(f"Or in words: ~1.34 × 10^154 (a number with 155 digits!)")
    print("-" * 70)
    print(f"Output directory: ./{OUTPUT_DIR}/")
    print(f"Default: Visualize all 512 neighborhood configurations per rule")
    print("-" * 70)

    try:
        rule_input = input("Enter rule number(s) to visualize (comma-separated): ")
        rule_numbers = [int(x.strip()) for x in rule_input.split(',')]

        configs_input = input("How many configurations per rule? (1-512, default 512): ").strip()
        max_configs = int(configs_input) if configs_input else 512
        max_configs = min(max(1, max_configs), 512)

        for rule_num in rule_numbers:
            if 0 <= rule_num < total_rules:
                print(f"\nGenerating visualization for rule {rule_num}...")
                visualize_rule_2d(rule_num, max_configs)
            else:
                print(f"Rule {rule_num} is invalid! Must be between 0 and {total_rules - 1}")

        print("\n" + "=" * 70)
        print(f"All visualizations saved to: ./{OUTPUT_DIR}/")
        print("=" * 70)

    except ValueError:
        print("Invalid input format! Please enter valid numbers.")
    except KeyboardInterrupt:
        print("\n\nProgram terminated.")


if __name__ == "__main__":
    main()