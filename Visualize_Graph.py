import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_rule_graph(R: int, K: int):
    """
    Loads the graph data from the JSON file and creates an enhanced visualization
    of the rule relationship network.
    """
    input_filename = f'graph_data_R{R}_K{K}.json'
    # New output filename for the final version
    output_filename = f'graph_visualization_R{R}_K{K}.png'

    # --- 1. Load Graph Data ---
    try:
        with open(input_filename, 'r') as f:
            graph_data = json.load(f)
        print(f"Successfully loaded graph data from '{input_filename}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
        print("Please run 'process_classification.py' first to generate it.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_filename}'.")
        return

    # --- 2. Create a NetworkX Graph ---
    G = nx.Graph()

    # Add nodes
    for node in graph_data['nodes']:
        G.add_node(node['id'])

    # Add edges and prepare edge colors
    edge_colors = []
    color_map = {'mirror': '#1f77b4', 'complement': '#d62728'}  # Blue and Red (Unchanged)

    for edge in graph_data['edges']:
        G.add_edge(edge['source'], edge['target'], type=edge['type'])

    # Re-iterate through edges in the graph to ensure color order is correct
    for u, v, data in G.edges(data=True):
        edge_colors.append(color_map.get(data['type'], 'gray'))

    print("Graph created with {} nodes and {} edges.".format(G.number_of_nodes(), G.number_of_edges()))

    # --- 3. Visualize the Graph ---
    print("Generating final visualization... (this may take a moment)")
    plt.figure(figsize=(24, 24))

    # Use a spring layout algorithm to position nodes
    # Increased k from 0.35 to 0.45 to further increase distance between nodes
    pos = nx.spring_layout(G, k=0.45, iterations=70, seed=42)

    # Draw the graph with enhanced visual properties
    nx.draw(G,
            pos,
            with_labels=True,
            node_color='#ffcc7f',  # Node color unchanged
            node_size=800,
            edge_color=edge_colors,  # Edge color unchanged
            width=2.5,  # Increased edge width from 2.0 to 2.5
            font_size=9,
            font_color='black',
            font_weight='bold')

    # --- 4. Create a Legend (Unchanged) ---
    mirror_patch = mpatches.Patch(color=color_map['mirror'], label='Mirror Transformation')
    complement_patch = mpatches.Patch(color=color_map['complement'], label='Complement Transformation')

    plt.legend(handles=[mirror_patch, complement_patch],
               loc='upper right',
               fontsize='x-large',
               title='Edge Types',
               title_fontsize='xx-large',
               facecolor='lightgrey',
               framealpha=0.9)

    plt.title(f'Rule Relationship Graph for R={R}, K={K}', fontsize=28, pad=20)

    # --- 5. Save the Image ---
    try:
        plt.savefig(output_filename, format='PNG', dpi=300, bbox_inches='tight')
        print(f"Visualization successfully saved to '{output_filename}'")
    except Exception as e:
        print(f"Error saving visualization: {e}")

    plt.close()


def main():
    """
    Main function to set parameters and run the visualization.
    """
    # --- Parameters ---
    R = 1
    K = 2
    # ------------------

    visualize_rule_graph(R, K)


if __name__ == '__main__':
    main()