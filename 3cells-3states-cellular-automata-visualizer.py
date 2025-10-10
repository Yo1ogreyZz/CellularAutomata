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
from matplotlib.patches import Arc, FancyBboxPatch
import warnings
from datetime import datetime, timezone

warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Directories
DB_DIR = "CA_1D_3cells_3states_Classification_DB"
DB_PATH = os.path.join(DB_DIR, "classification.db")
JSON_PATH = os.path.join(DB_DIR, "classification_db.json")
OUTPUT_DIR = "CA_1D_3cells_3states_Graphs"
GRAPH_DIR = os.path.join(OUTPUT_DIR, "class_graphs")
EDGE_DIR = os.path.join(OUTPUT_DIR, "edge_data")

Path(GRAPH_DIR).mkdir(parents=True, exist_ok=True)
Path(EDGE_DIR).mkdir(parents=True, exist_ok=True)


def get_rule_lookup(rule_number, n_configs=27):
    """Convert rule number to lookup table"""
    ternary = np.base_repr(rule_number, base=3).zfill(n_configs)

    lookup = {}
    for i in range(n_configs):
        config = np.base_repr(i, base=3).zfill(3)
        lookup[config] = int(ternary[n_configs - 1 - i])

    return lookup


def build_graph_from_rule(rule_number):
    """
    Build directed multigraph from 3-state 3-cells CA rule
    - Nodes: states (0, 1, 2)
    - Edges: transitions with neighbor configurations
    """
    rule_lookup = get_rule_lookup(rule_number)
    G = nx.MultiDiGraph()

    # Add 3 nodes
    G.add_node(0, label="State 0")
    G.add_node(1, label="State 1")
    G.add_node(2, label="State 2")

    edge_data = []
    transition_summary = defaultdict(list)

    # For each of 27 configurations
    for i in range(27):
        config = np.base_repr(i, base=3).zfill(3)

        # Parse: L C R
        L, C, R = [int(config[j]) for j in range(3)]
        center = C

        # Get next state
        next_state = rule_lookup[config]

        # Neighbors: L _ R
        neighbor_config = f"{L}{R}"
        edge_label = f"{L}_{R}"

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


def draw_self_loop(ax, x, y, width, color, alpha, position='top'):
    """Draw a self-loop arc"""
    loop_size = 2.0

    if position == 'left':
        center_x = x - 1.5
        center_y = y + 1.5
        theta1 = -30
        theta2 = 270
    elif position == 'right':
        center_x = x + 1.5
        center_y = y + 1.5
        theta1 = -90
        theta2 = 210
    else:  # top
        center_x = x
        center_y = y + 2.0
        theta1 = 20
        theta2 = 340

    arc = Arc((center_x, center_y), loop_size * 2, loop_size * 2,
              angle=0, theta1=theta1, theta2=theta2,
              color=color, linewidth=width, alpha=alpha, zorder=1)
    ax.add_patch(arc)

    return center_x, center_y


def get_label_position_for_edge(u, v, pos, edge_counts):
    """
    智能计算标签位置，避免重叠
    对于对向的边（如0→1和1→0），将标签分别放在上下

    关键逻辑：
    - 0→1 的标签应该在下方（因为0→1的箭头画在下面）
    - 1→0 的标签应该在上方（因为1→0的箭头画在上面）
    """
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # 检查是否有反向边
    reverse_edge = (v, u)
    has_reverse = edge_counts.get(reverse_edge, 0) > 0

    if has_reverse:
        # 有反向边：需要分开标签位置
        # 关键：根据当前边是"上弧"还是"下弧"来决定

        # 对于水平的边（0↔1），使用特殊规则
        if (u, v) in [(0, 1), (1, 0)]:
            if u == 0 and v == 1:
                # 0→1：标签放下方
                offset_y = -1.5
            else:  # u == 1 and v == 0
                # 1→0：标签放上方
                offset_y = 1.2
        # 对于对角的边（0↔2, 1↔2）
        elif (u, v) in [(0, 2), (2, 0)]:
            if u == 0 and v == 2:
                # 0→2：标签放左侧
                offset_y = 1.2
            else:
                # 2→0：标签放右侧
                offset_y = -1.2
        elif (u, v) in [(1, 2), (2, 1)]:
            if u == 1 and v == 2:
                # 1→2：标签放右侧
                offset_y = 1.2
            else:
                # 2→1：标签放左侧
                offset_y = -1.2
        else:
            # 默认
            offset_y = 1.0
    else:
        # 没有反向边：标签放在弧线上方
        offset_y = 1.0

    return mid_x, mid_y + offset_y


def visualize_class_graph(class_id, canonical_rule, class_size, output_filename):
    """Visualize the state transition graph for 3-state CA"""
    G, edge_data, transition_summary = build_graph_from_rule(canonical_rule)

    fig, ax = plt.subplots(figsize=(24, 22))

    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#ffffff')

    # Position 3 nodes in a triangle
    pos = {
        0: (-6, -2),  # Bottom left
        1: (6, -2),  # Bottom right
        2: (0, 4)  # Top
    }

    # Count transitions
    edge_counts = defaultdict(int)
    for u, v in G.edges():
        edge_counts[(u, v)] += 1

    # Calculate line widths
    max_width = 20
    min_width = 2

    edge_styles = {}
    for (u, v), count in edge_counts.items():
        percentage = (count / 27) * 100  # 27 total configurations
        if count == 0:
            width = 0
        else:
            width = min_width + (max_width - min_width) * (count / 27)
        edge_styles[(u, v)] = {
            'count': count,
            'percentage': percentage,
            'width': width
        }

    # Draw self-loops FIRST
    for (u, v), style in edge_styles.items():
        count = style['count']
        width = style['width']

        if count == 0 or u != v:
            continue

        # Colors for 3 states
        colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
        edge_color = colors[u]
        alpha = 0.85

        # Position self-loops
        if u == 0:
            position = 'left'
        elif u == 1:
            position = 'right'
        else:
            position = 'top'

        draw_self_loop(ax, pos[u][0], pos[u][1], width, edge_color, alpha, position)

    # Draw nodes
    node_colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
    node_labels_text = ["STATE 0", "STATE 1", "STATE 2"]

    for node in [0, 1, 2]:
        x, y = pos[node]

        # Shadow
        shadow = plt.Circle((x + 0.2, y - 0.2), 1.5,
                            facecolor='gray', alpha=0.25, zorder=2)
        ax.add_patch(shadow)

        # Main node
        circle = plt.Circle((x, y), 1.5,
                            facecolor=node_colors[node], alpha=0.95, zorder=3,
                            edgecolor='white', linewidth=6)
        ax.add_patch(circle)

        # Inner circle
        inner_circle = plt.Circle((x, y), 1.25,
                                  facecolor=node_colors[node], alpha=0.75, zorder=4,
                                  edgecolor='none')
        ax.add_patch(inner_circle)

        # Label
        ax.text(x, y, node_labels_text[node],
                ha='center', va='center',
                fontsize=22, fontweight='bold',
                color='white', zorder=5)

    # Draw regular edges and collect label positions
    label_positions = {}

    for (u, v), style in edge_styles.items():
        count = style['count']
        percentage = style['percentage']
        width = style['width']

        if count == 0:
            continue

        if u == v:
            # Store label position for self-loop
            if u == 0:
                loop_x = pos[u][0] - 1.5
                loop_y = pos[u][1] + 1.5
            elif u == 1:
                loop_x = pos[u][0] + 1.5
                loop_y = pos[u][1] + 1.5
            else:
                loop_x = pos[u][0]
                loop_y = pos[u][1] + 2.0
            label_positions[(u, v)] = (loop_x, loop_y)
        else:
            # State transition
            edge_color = '#339af0'

            # Calculate arc radius based on direction
            if (u, v) in [(0, 1), (1, 0)]:
                arc_rad = 0.2
            elif (u, v) in [(0, 2), (2, 0)]:
                arc_rad = -0.2
            else:
                arc_rad = 0.2

            arc_style = f"arc3,rad={arc_rad}"
            alpha = 0.85

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
                min_source_margin=150,
                min_target_margin=150
            )

            # 智能计算标签位置
            label_x, label_y = get_label_position_for_edge(u, v, pos, edge_counts)
            label_positions[(u, v)] = (label_x, label_y)

    # Draw percentage labels - 确保所有非零边都有标签
    for (u, v), style in edge_styles.items():
        count = style['count']
        percentage = style['percentage']

        # ✅ 关键：只要 count > 0 就绘制标签
        if count == 0:
            continue

        if (u, v) not in label_positions:
            # 如果没有计算位置，使用默认位置
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            label_x = (x1 + x2) / 2
            label_y = (y1 + y2) / 2
        else:
            label_x, label_y = label_positions[(u, v)]

        if u == v:
            edge_color = node_colors[u]
        else:
            edge_color = '#339af0'

        label_text = f'{percentage:.1f}%'

        # 根据百分比调整字体大小
        fontsize = 14

        # ✅ 强制绘制所有非零标签
        ax.text(label_x, label_y, label_text,
                ha='center', va='center',
                fontsize=fontsize, fontweight='bold',
                color='white',
                bbox=dict(
                    boxstyle='round,pad=0.4',
                    facecolor=edge_color,
                    alpha=0.95,
                    edgecolor='white',
                    linewidth=2.0
                ),
                zorder=7)

    # Title
    title = f'State Transition Graph - Equivalence Class #{class_id}\n'
    title += f'Canonical Rule: {canonical_rule} • Class Size: {class_size} equivalent rules'

    plt.title(title, fontsize=22, weight='bold', pad=40,
              color='#212529', family='sans-serif')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#ff6b6b', label='State 0', alpha=0.85, edgecolor='none'),
        mpatches.Patch(facecolor='#4ecdc4', label='State 1', alpha=0.85, edgecolor='none'),
        mpatches.Patch(facecolor='#95e1d3', label='State 2', alpha=0.85, edgecolor='none'),
        mpatches.Patch(facecolor='#339af0', label='Transitions', alpha=0.85, edgecolor='none')
    ]

    legend = ax.legend(handles=legend_elements, loc='lower right',
                       fontsize=14, framealpha=0.95,
                       edgecolor='#dee2e6', fancybox=True,
                       shadow=True, borderpad=1.2)
    legend.get_frame().set_facecolor('#f8f9fa')
    legend.get_frame().set_linewidth(2)

    plt.axis('off')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 7)

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
        'transitions_by_type': {},
        'all_transitions': edge_data
    }

    # Categorize by all 9 transition types (0→0, 0→1, ..., 2→2)
    for i in range(3):
        for j in range(3):
            key = f'{i}_to_{j}'
            organized_data['transitions_by_type'][key] = []

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
        except Exception as e:
            print(f"  ⚠️  Error: {e}")

    conn.close()

    print(f"\n{'=' * 80}")
    print(f"✓ Selected graphs generated!")
    print(f"{'=' * 80}")


def main():
    print("=" * 80)
    print("1D 3-cells 3-states CA State Transition Graph Visualizer")
    print("=" * 80)
    print(f"Output: {OUTPUT_DIR}/")
    print("\nFeatures:")
    print("  • 3 states with distinct colors")
    print("  • Triangle node layout")
    print("  • ALL edges labeled (including thin lines)")
    print("  • Fixed: 0→1 label below, 1→0 label above")
    print("=" * 80)

    if not os.path.exists(DB_PATH):
        print(f"\n❌ Database not found!")
        print(f"Please run ca_1d_3cells_3states_classifier.py first.")
        return

    try:
        while True:
            print("\n" + "=" * 80)
            print("Options:")
            print("  1. Visualize ALL classes")
            print("  2. Visualize specific classes (by class ID)")
            print("  3. Exit")
            print("=" * 80)

            choice = input("\nEnter choice (1-3): ").strip()

            if choice == '1':
                visualize_all_classes_from_db()

            elif choice == '2':
                class_input = input("\nEnter class ID(s) (comma-separated): ").strip()
                class_ids = [int(x.strip()) for x in class_input.split(',')]
                visualize_specific_classes(class_ids)

            elif choice == '3':
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