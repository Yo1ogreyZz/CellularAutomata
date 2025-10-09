import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings

# Suppress font warnings
warnings.filterwarnings('ignore', category=UserWarning)

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

def visualize_rule_2d(rule_number, max_configs=64):
    """
    Visualize the transition diagram of 2D 3x3 rule
    Since there are 512 configurations, only show the first 64 by default
    """
    lookup = get_rule_lookup_2d(rule_number)
    
    # Calculate layout
    cols = 8
    rows = (max_configs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 2.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'2D 3x3 Cellular Automaton - Rule {rule_number} (First {max_configs} configs)', 
                 fontsize=16, weight='bold')
    
    for idx in range(max_configs):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        config = format(idx, '09b')
        output = lookup[config]
        
        # Draw 3x3 grid
        grid = np.array([[int(config[i]) for i in range(j*3, j*3+3)] for j in range(3)])
        
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
        ax.text(2, -0.2, f'#{idx}', ha='center', va='top', fontsize=8)
    
    # Hide extra subplots
    for idx in range(max_configs, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    filename = f'rule_2d_3x3_{rule_number}_first{max_configs}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f'Image saved: {filename}')
    plt.close()

def main():
    print("=" * 60)
    print("2D 3x3 Cellular Automaton Rule Visualizer")
    print("=" * 60)
    print("Rule range: 0 to 2^512 - 1")
    print("Note: Total number of rules is 2^512 (approx 1.34×10^154), an astronomical number!")
    print("Suggested rule range: 0 to 10^20")
    print("Due to the large number of configurations (512), each rule displays")
    print("the first 64 configurations by default")
    print("-" * 60)
    
    try:
        rule_input = input("Enter rule number(s) to visualize (comma-separated): ")
        rule_numbers = [int(x.strip()) for x in rule_input.split(',')]
        
        configs_input = input("How many configurations per rule? (1-512, default 64): ").strip()
        max_configs = int(configs_input) if configs_input else 64
        max_configs = min(max(1, max_configs), 512)
        
        for rule_num in rule_numbers:
            if rule_num >= 0:
                print(f"\nGenerating visualization for rule {rule_num}...")
                visualize_rule_2d(rule_num, max_configs)
            else:
                print(f"Rule {rule_num} is invalid!")
    
    except ValueError:
        print("Invalid input format! Please enter valid numbers.")
    except KeyboardInterrupt:
        print("\n\nProgram terminated.")

if __name__ == "__main__":
    main()