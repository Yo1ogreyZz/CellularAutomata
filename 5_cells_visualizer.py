import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import json
import sqlite3
import os
from collections import defaultdict
from tqdm import tqdm
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
import warnings
from datetime import datetime, timezone

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Directories
DB_DIR = "CA_1D_5cells_Classification_DB"
DB_PATH = os.path.join(DB_DIR, "classification.db")
JSON_PATH = os.path.join(DB_DIR, "classification_db.json")
OUTPUT_DIR = "CA_1D_5cells_Graphs"
GRAPH_DIR = os.path.join(OUTPUT_DIR, "class_graphs")
EDGE_DIR = os.path.join(OUTPUT_DIR, "edge_data")

Path(GRAPH_DIR).mkdir(parents=True, exist_ok=True)
Path(EDGE_DIR).mkdir(parents=True, exist_ok=True)


def get_rule_lookup_1d(rule_number, n_states=32):
    """Convert rule number to lookup table"""
    binary = format(rule_number, f'0{n_states}b')
    lookup = {}
    for i in range(n_states):
        config = format(i, '05b')
        lookup[config] = int(binary[n_states - 1 - i])
    return lookup


def build_graph_from_rule(rule_number):
    """
    Build directed multigraph from 1D CA rule
    - Nodes: center states (0, 1)
    - Edges: transitions with neighbor configurations (excluding center)

    For 5-cells: LL L C R RR (2 left + center + 2 right)
    Neighbors only: LL L _ R RR (4 neighbors)
    """
    rule_lookup = get_rule_lookup_1d(rule_number)
    G = nx.MultiDiGraph()

    # Add nodes
    G.add_node(0, label="State 0")
    G.add_node(1, label="State 1")

    edge_data = []
    transition_summary = defaultdict(list)

    # For each of 32 configurations
    for i in range(32):
        config = format(i, '05b')

        # Parse configuration: LL L C R RR
        LL, L, C, R, RR = [int(config[j]) for j in range(5)]
        center = C

        # Get next state
        next_state = rule_lookup[config]

        # Neighbors (excluding center): LL L _ R RR
        neighbor_config = f"{LL}{L}{R}{RR}"

        # Create edge label
        edge_label = f"{LL}{L}_{R}{RR}"

        # Add edge
        G.add_edge(center, int(next_state),
                   label=edge_label,
                   neighbors=neighbor_config,
                   full_config=config)

        # Track transition
        transition_key = (center, int(next_state))
        transition_summary[transition_key].append(neighbor_config)

        # Store edge data
        edge_data.append({
            'from_state': center,
            'to_state': int(next_state),
            'neighbor_config': neighbor_config,
            'edge_label': edge_label,
            'full_config': config,
            'config_decimal': i
        })

    return G, edge_data, transition_summary


def draw_self_loop(ax, x, y, width, color, alpha, direction='left'):
    """Draw a self-loop arc at the BOTTOM layer"""
    loop_size = 2.0

    if direction == 'left':
        center_x = x - 1.5
        center_y = y + 1.5
        theta1 = -30
        theta2 = 270
    else:
        center_x = x + 1.5
        center_y = y + 1.5
        theta1 = -90
        theta2 = 210

    arc = Arc((center_x, center_y), loop_size * 2, loop_size * 2,
              angle=0, theta1=theta1, theta2=theta2,
              color=color, linewidth=width, alpha=alpha, zorder=1)
    ax.add_patch(arc)

    return center_x, center_y


def visualize_class_graph(class_id, canonical_rule, class_size, output_filename):
    """Visualize the state transition graph"""
    # Build graph from canonical rule
    G, edge_data, transition_summary = build_graph_from_rule(canonical_rule)

    fig, ax = plt.subplots(figsize=(22, 20))

    # Set background
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')

    # Position nodes
    pos = {0: (-5, 0), 1: (5, 0)}

    # Count transitions
    edge_counts = defaultdict(int)
    for u, v in G.edges():
        edge_counts[(u, v)] += 1

    # Calculate line widths
    max_width = 20
    min_width = 2

    edge_styles = {}
    for (u, v), count in edge_counts.items():
        percentage = (count / 32) * 100  # 32 total configurations for 5-cells
        if count == 0:
            width = 0
        else:
            width = min_width + (max_width - min_width) * (count / 32)
        edge_styles[(u, v)] = {
            'count': count,
            'percentage': percentage,
            'width': width
        }

    # Draw self-loops FIRST (bottom layer)
    for (u, v), style in edge_styles.items():
        count = style['count']
        width = style['width']

        if count == 0:
            continue

        if u == v:
            edge_color = '#51cf66'
            alpha = 0.85
            direction = 'left' if u == 0 else 'right'
            draw_self_loop(ax, pos[u][0], pos[u][1], width, edge_color, alpha, direction)

    # Draw nodes
    for node in [0, 1]:
        x, y = pos[node]
        shadow = plt.Circle((x + 0.2, y - 0.2), 1.4,
                            facecolor='gray', alpha=0.25, zorder=2)
        ax.add_patch(shadow)

        node_color = '#ff6b6b' if node == 0 else '#4ecdc4'

        circle = plt.Circle((x, y), 1.4,
                            facecolor=node_color, alpha=0.95, zorder=3,
                            edgecolor='white', linewidth=6)
        ax.add_patch(circle)

        inner_circle = plt.Circle((x, y), 1.15,
                                  facecolor=node_color, alpha=0.75, zorder=4,
                                  edgecolor='none')
        ax.add_patch(inner_circle)

    # Node labels
    node_labels = {0: "STATE 0", 1: "STATE 1"}

    for node, label in node_labels.items():
        x, y = pos[node]
        ax.text(x, y, label,
                ha='center', va='center',
                fontsize=24, fontweight='bold',
                color='white', zorder=5)

    # Draw edges and collect label positions
    label_positions = {}

    for (u, v), style in edge_styles.items():
        count = style['count']
        percentage = style['percentage']
        width = style['width']

        if count == 0:
            continue

        if u == v:
            # Store label position for self-loop
            direction = 'left' if u == 0 else 'right'
            if direction == 'left':
                loop_center_x = pos[u][0] - 1.5
                loop_center_y = pos[u][1] + 1.5
            else:
                loop_center_x = pos[u][0] + 1.5
                loop_center_y = pos[u][1] + 1.5
            label_positions[(u, v)] = (loop_center_x, loop_center_y)
        else:
            # State transition
            edge_color = '#339af0'
            arc_rad = 0.3
            alpha = 0.9

            arc_style = f"arc3,rad={arc_rad}"
            nx.draw_networkx_edges(
                G, pos, [(u, v)],
                connectionstyle=arc_style,
                edge_color=edge_color,
                alpha=alpha,
                arrows=True,
                arrowsize=30 + width * 1.2,
                width=width,
                node_size=0,
                arrowstyle='->',
                min_source_margin=140,
                min_target_margin=140
            )

            # Calculate label position
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            mid_x = (x1 + x2) / 2

            # 0→1 below, 1→0 above
            if u == 0 and v == 1:
                mid_y = (y1 + y2) / 2 - 1.5
            elif u == 1 and v == 0:
                mid_y = (y1 + y2) / 2 + 1.1
            else:
                mid_y = (y1 + y2) / 2 + 1.1

            label_positions[(u, v)] = (mid_x, mid_y)

    # Draw percentage labels
    for (u, v), style in edge_styles.items():
        count = style['count']
        percentage = style['percentage']

        if count == 0:
            continue

        if (u, v) not in label_positions:
            continue

        label_x, label_y = label_positions[(u, v)]

        edge_color = '#51cf66' if u == v else '#339af0'
        label_text = f'{percentage:.1f}%'

        if percentage < 1:
            fontsize = 11
        elif percentage < 5:
            fontsize = 13
        else:
            fontsize = 15

        ax.text(label_x, label_y, label_text,
                ha='center', va='center',
                fontsize=fontsize, fontweight='bold',
                color='white',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor=edge_color,
                    alpha=0.95,
                    edgecolor='white',
                    linewidth=2.5
                ),
                zorder=7)

    # Title
    title = f'State Transition Graph - Equivalence Class #{class_id}\n'
    title += f'Canonical Rule: {canonical_rule} • Class Size: {class_size} equivalent rules'

    plt.title(title, fontsize=22, weight='bold', pad=40,
              color='#212529', family='sans-serif')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#51cf66', label='State Preserved (self-loop)', alpha=0.85, edgecolor='none'),
        mpatches.Patch(facecolor='#339af0', label='State Changed (transition)', alpha=0.9, edgecolor='none')
    ]

    legend = ax.legend(handles=legend_elements, loc='lower left',
                       fontsize=14, framealpha=0.95,
                       edgecolor='#dee2e6', fancybox=True,
                       shadow=True, borderpad=1.2)
    legend.get_frame().set_facecolor('#f8f9fa')
    legend.get_frame().set_linewidth(2)

    plt.axis('off')
    ax.set_xlim(-9, 9)
    ax.set_ylim(-4.5, 5.5)

    ax.axhline(y=0, color='#dee2e6', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='#dee2e6', linestyle='--', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    plt.savefig(output_filename, dpi=200, bbox_inches='tight',
                facecolor='#f8f9fa', edgecolor='none')
    plt.close()

    return edge_data


def save_edge_data(class_id, edge_data):
    """Save edge/transition data to JSON"""
    edge_filename = os.path.join(EDGE_DIR, f'class_{class_id}_transitions.json')

    organized_data = {
        'class_id': class_id,
        'total_transitions': len(edge_data),
        'transitions_by_type': {
            '0_to_0': [],
            '0_to_1': [],
            '1_to_0': [],
            '1_to_1': []
        },
        'all_transitions': edge_data
    }

    for trans in edge_data:
        key = f"{trans['from_state']}_to_{trans['to_state']}"
        organized_data['transitions_by_type'][key].append(trans)

    with open(edge_filename, 'w') as f:
        json.dump(organized_data, f, indent=2)


def visualize_all_classes_from_db():
    """Visualize all classes from the database"""
    if not os.path.exists(DB_PATH):
        print(f"\n❌ Database not found at {DB_PATH}")
        print("Please run the classifier first!")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM classes')
    total_classes = cursor.fetchone()[0]

    if total_classes == 0:
        print("\n❌ No classes found in database!")
        conn.close()
        return

    cursor.execute('SELECT class_id, canonical_rule, class_size FROM classes ORDER BY class_id')
    classes = cursor.fetchall()
    conn.close()

    print(f"\n{'=' * 80}")
    print(f"Generating State Transition Graphs for All Classes")
    print(f"{'=' * 80}")
    print(f"Total classes to visualize: {total_classes}")
    print(f"Output directory: ./{GRAPH_DIR}/")
    print(f"Transition data directory: ./{EDGE_DIR}/")
    print(f"{'=' * 80}\n")

    for class_id, canonical_rule, class_size in tqdm(classes, desc="Generating graphs", unit="class"):
        output_filename = os.path.join(GRAPH_DIR, f'class_{class_id}.png')

        try:
            edge_data = visualize_class_graph(class_id, int(canonical_rule), class_size, output_filename)
            save_edge_data(class_id, edge_data)
        except Exception as e:
            tqdm.write(f"  ⚠️  Error processing class {class_id}: {e}")

    print(f"\n{'=' * 80}")
    print(f"✓ All graphs generated successfully!")
    print(f"  Graphs: ./{GRAPH_DIR}/")
    print(f"  Transition data: ./{EDGE_DIR}/")
    print(f"{'=' * 80}")


def visualize_specific_classes(class_ids):
    """Visualize specific classes by their IDs"""
    if not os.path.exists(DB_PATH):
        print(f"\n❌ Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print(f"\n{'=' * 80}")
    print(f"Generating Graphs for Selected Classes")
    print(f"{'=' * 80}")

    for class_id in class_ids:
        cursor.execute('SELECT canonical_rule, class_size FROM classes WHERE class_id = ?', (class_id,))
        result = cursor.fetchone()

        if not result:
            print(f"⚠️  Class {class_id} not found in database")
            continue

        canonical_rule, class_size = result
        print(f"\n📊 Processing Class {class_id}...")

        output_filename = os.path.join(GRAPH_DIR, f'class_{class_id}.png')

        try:
            edge_data = visualize_class_graph(class_id, int(canonical_rule), class_size, output_filename)
            save_edge_data(class_id, edge_data)
            print(f"  ✓ Graph saved: {output_filename}")
            print(f"  ✓ Transition data saved")
        except Exception as e:
            print(f"  ⚠️  Error: {e}")

    conn.close()

    print(f"\n{'=' * 80}")
    print(f"✓ Selected graphs generated!")
    print(f"{'=' * 80}")


def query_class_transitions(class_id):
    """Query and display transition information for a specific class"""
    edge_file = os.path.join(EDGE_DIR, f'class_{class_id}_transitions.json')

    if not os.path.exists(edge_file):
        print(f"\n❌ Transition data not found for class {class_id}")
        print("Generate the graph first!")
        return

    with open(edge_file, 'r') as f:
        data = json.load(f)

    print(f"\n{'=' * 80}")
    print(f"Transition Analysis - Class {class_id}")
    print(f"{'=' * 80}")
    print(f"Total Transitions: {data['total_transitions']}")
    print(f"\nTransition Breakdown:")
    print(f"{'-' * 80}")

    for trans_type, transitions in data['transitions_by_type'].items():
        count = len(transitions)
        percentage = (count / 32) * 100
        from_state, to_state = trans_type.split('_to_')

        if from_state == to_state:
            change_type = "State Preserved"
            symbol = "✓"
        else:
            change_type = "State Changed"
            symbol = "⟳"

        bar = '█' * int(percentage / 5) + '░' * (20 - int(percentage / 5))

        print(f"\n{symbol} {trans_type.replace('_', ' → ').upper()}: {count} transitions ({percentage:.1f}%)")
        print(f"  {bar}")
        print(f"  Type: {change_type}")

        if count > 0 and count <= 10:
            print(f"  Sample neighbor configs:")
            for i, trans in enumerate(transitions[:5]):
                print(f"    {i + 1}. {trans['neighbor_config']} → State {to_state}")

    print(f"\n{'=' * 80}")


def main():
    print("=" * 80)
    print("1D 5-cells CA State Transition Graph Visualizer")
    print("=" * 80)
    print(f"Output: {OUTPUT_DIR}/")
    print("\nFeatures:")
    print("  • Line thickness = transition probability")
    print("  • 0→1 labels below, 1→0 labels above")
    print("  • Self-loops at bottom layer")
    print("  • 32 configurations total (2^5)")
    print("=" * 80)

    if not os.path.exists(DB_PATH):
        print(f"\n❌ Database not found!")
        print(f"Please run ca_1d_5cells_classifier.py first to classify rules.")
        return

    try:
        while True:
            print("\n" + "=" * 80)
            print("Options:")
            print("  1. Visualize ALL classes (generate all graphs)")
            print("  2. Visualize specific classes (by class ID)")
            print("  3. Query transition data for a class")
            print("  4. Exit")
            print("=" * 80)

            choice = input("\nEnter choice (1-4): ").strip()

            if choice == '1':
                visualize_all_classes_from_db()

            elif choice == '2':
                class_input = input("\nEnter class ID(s) (comma-separated): ").strip()
                class_ids = [int(x.strip()) for x in class_input.split(',')]
                visualize_specific_classes(class_ids)

            elif choice == '3':
                class_id = int(input("\nEnter class ID: "))
                query_class_transitions(class_id)

            elif choice == '4':
                print("\n👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice!")

    except ValueError as e:
        print(f"\n❌ Invalid input: {e}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted!")


if __name__ == "__main__":
    main()