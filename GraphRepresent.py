import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any
from itertools import product
import pandas as pd
import json


class CAToGraph:
    """
    Convert Wolfram Elementary Cellular Automata (ECA) rules to a multi-digraph.

    Defaults assume ECA: num_states=2, radius=1.
    """
    def __init__(self, num_states: int = 2, radius: int = 1):
        self.num_states = num_states
        self.radius = radius
        self.neighborhood_size = 2 * radius + 1
        if not (self.num_states == 2 and self.radius == 1):
            print("[Warn] Current rule table only supports Wolfram ECA ordering (2 states, radius=1).")

    # ---------- Rule table (ECA) ----------
    def rule_number_to_table(self, rule_number: int) -> Dict[Tuple[int, ...], int]:
        """
        Wolfram ECA convention:
        Neighborhood order: 111,110,101,100,011,010,001,000
        Map to bits of rule_number from MSB to LSB.
        """
        neighborhoods = [
            (1, 1, 1),
            (1, 1, 0),
            (1, 0, 1),
            (1, 0, 0),
            (0, 1, 1),
            (0, 1, 0),
            (0, 0, 1),
            (0, 0, 0),
        ]
        if not (0 <= rule_number <= 255):
            raise ValueError("Wolfram ECA rule number must be in [0, 255].")

        rule_binary = format(rule_number, '08b')  # MSB -> LSB, e.g., 30 -> '00011110'
        rule_outputs = [int(bit) for bit in rule_binary]

        rule_table: Dict[Tuple[int, ...], int] = {}
        for neighborhood, output in zip(neighborhoods, rule_outputs):
            rule_table[neighborhood] = output

        return rule_table

    # ---------- Neighbor enumerations ----------
    def generate_neighbor_configs(self) -> List[Tuple[int, ...]]:
        """
        All possible (left_neighbors + right_neighbors) tuples, length = 2 * radius.
        For radius=1, it's [(l,r)] with 4 combinations for binary states.
        """
        neighbor_positions = 2 * self.radius
        return list(product(range(self.num_states), repeat=neighbor_positions))

    def neighbor_config_to_matrix(self, config: Tuple[int, ...]) -> np.ndarray:
        """
        For radius=1, return [[l], [r]] to keep backward compatibility with your flattening code.
        For radius>1, returns a 2 x R matrix: [left; right].
        """
        left_neighbors = config[:self.radius]
        right_neighbors = config[self.radius:]

        if self.radius == 1:
            return np.array([[left_neighbors[0]], [right_neighbors[0]]])
        else:
            return np.array([list(left_neighbors), list(right_neighbors)])

    # ---------- Graph build ----------
    def build_graph_from_rule_table(self, rule_table: Dict[Tuple[int, ...], int], rule_id: int) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()

        # Nodes are center states
        for state in range(self.num_states):
            G.add_node(state, label=f"center={state}")

        neighbor_configs = self.generate_neighbor_configs()

        for center_state in range(self.num_states):
            for neighbor_config in neighbor_configs:
                left_neighbors = neighbor_config[:self.radius]
                right_neighbors = neighbor_config[self.radius:]
                full_neighborhood = left_neighbors + (center_state,) + right_neighbors

                if full_neighborhood in rule_table:
                    next_state = rule_table[full_neighborhood]
                    neighbor_matrix = self.neighbor_config_to_matrix(neighbor_config)

                    G.add_edge(
                        center_state,
                        next_state,
                        neighbor_config=neighbor_config,
                        neighbor_matrix=neighbor_matrix,
                        neighbor_string=self.format_neighbor_string(neighbor_config),
                        full_neighborhood=full_neighborhood,
                        rule_id=rule_id
                    )
        return G

    def format_neighbor_string(self, config: Tuple[int, ...]) -> str:
        left = ''.join(map(str, config[:self.radius]))
        right = ''.join(map(str, config[self.radius:]))
        return f'{left}|{right}'

    def build_graph_from_rule_number(self, rule_number: int) -> nx.MultiDiGraph:
        rule_table = self.rule_number_to_table(rule_number)
        return self.build_graph_from_rule_table(rule_table, rule_number)

    # ---------- GNN features ----------
    def extract_gnn_features(self, G: nx.MultiDiGraph) -> Dict[str, Any]:
        edges = list(G.edges(keys=True))
        edge_index = [[e[0] for e in edges], [e[1] for e in edges]]

        edge_attr = []
        rule_ids = []
        for u, v, key in edges:
            edge_data = G[u][v][key]
            neighbor_matrix = np.array(edge_data['neighbor_matrix'])
            edge_attr.append(neighbor_matrix.flatten())
            rule_ids.append(edge_data['rule_id'])

        return {
            'edge_index': np.array(edge_index),
            'edge_attr': np.array(edge_attr, dtype=object),
            'rule_ids': np.array(rule_ids),
            'num_nodes': G.number_of_nodes(),
            'num_edges': len(edges),
        }

    def get_graph_representation(self, rule_number: int) -> Dict[str, Any]:
        """
        Get the graph representation of a rule, including basic statistics and structural information.
        """
        graph = self.build_graph_from_rule_number(rule_number)
        features = self.extract_gnn_features(graph)

        # Basic graph statistics
        graph_stats = {
            'rule_number': rule_number,
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'is_strongly_connected': nx.is_strongly_connected(graph),
        }

        # Node degree statistics
        in_degrees = dict(graph.in_degree())
        out_degrees = dict(graph.out_degree())
        graph_stats.update({
            'max_in_degree': max(in_degrees.values()) if in_degrees else 0,
            'max_out_degree': max(out_degrees.values()) if out_degrees else 0,
            'avg_in_degree': sum(in_degrees.values()) / len(in_degrees) if in_degrees else 0,
            'avg_out_degree': sum(out_degrees.values()) / len(out_degrees) if out_degrees else 0,
        })

        # Edge details
        edge_details = []
        for u, v, k in graph.edges(keys=True):
            edge_data = graph[u][v][k]
            edge_details.append({
                'from_state': u,
                'to_state': v,
                'key': k,
                'full_neighborhood': edge_data['full_neighborhood'],
                'neighbor_config': edge_data['neighbor_config'],
                'neighbor_string': edge_data['neighbor_string']
            })

        # Rule table
        rule_table = self.rule_number_to_table(rule_number)
        rule_binary = format(rule_number, '08b')

        return {
            'graph_stats': graph_stats,
            'rule_table': {str(k): v for k, v in rule_table.items()},
            'rule_binary': rule_binary,
            'edge_details': edge_details,
            'gnn_features': {
                'edge_index': features['edge_index'].tolist(),
                'edge_attr': [attr.tolist() for attr in features['edge_attr']],
                'rule_ids': features['rule_ids'].tolist()
            }
        }


def analyze_all_rules():
    """
    Analyze the graph representation of all 256 ECA rules (0-255).
    """
    ca_converter = CAToGraph(num_states=2, radius=1)
    all_rules_data = []

    print("Analyzing graph representations for all ECA rules...")
    print("=" * 60)

    for rule_number in range(256):
        try:
            rule_data = ca_converter.get_graph_representation(rule_number)
            all_rules_data.append(rule_data)

            # Print basic information
            stats = rule_data['graph_stats']
            print(f"Rule {rule_number:3d}: "
                  f"Nodes={stats['num_nodes']}, "
                  f"Edges={stats['num_edges']}, "
                  f"Density={stats['density']:.3f}, "
                  f"StronglyConnected={stats['is_strongly_connected']}")

        except Exception as e:
            print(f"Error processing Rule {rule_number}: {e}")
            continue

    print("=" * 60)
    print(f"Successfully analyzed {len(all_rules_data)} rules.")

    # Save all data to a JSON file
    output_file = "all_eca_rules_graph_representation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_rules_data, f, indent=2, ensure_ascii=False)

    print(f"Graph representations for all rules have been saved to: {output_file}")

    # Generate summary statistics
    generate_summary_statistics(all_rules_data)

    return all_rules_data


def generate_summary_statistics(all_rules_data):
    """
    Generate summary statistics.
    """
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("=" * 60)

    # Extract statistical data
    num_edges_list = [data['graph_stats']['num_edges'] for data in all_rules_data]
    density_list = [data['graph_stats']['density'] for data in all_rules_data]
    strongly_connected_count = sum(1 for data in all_rules_data if data['graph_stats']['is_strongly_connected'])

    # Basic statistics
    print(f"Edge Count Statistics:")
    print(f"  Min Edges: {min(num_edges_list)}")
    print(f"  Max Edges: {max(num_edges_list)}")
    print(f"  Avg Edges: {sum(num_edges_list)/len(num_edges_list):.2f}")

    print(f"\nDensity Statistics:")
    print(f"  Min Density: {min(density_list):.3f}")
    print(f"  Max Density: {max(density_list):.3f}")
    print(f"  Avg Density: {sum(density_list)/len(density_list):.3f}")

    print(f"\nConnectivity:")
    print(f"  Strongly Connected Rules: {strongly_connected_count}/256 ({strongly_connected_count/256*100:.1f}%)")

    # Group by edge count
    edge_count_groups = {}
    for data in all_rules_data:
        edge_count = data['graph_stats']['num_edges']
        if edge_count not in edge_count_groups:
            edge_count_groups[edge_count] = []
        edge_count_groups[edge_count].append(data['graph_stats']['rule_number'])

    print(f"\nGrouped by Edge Count:")
    for edge_count in sorted(edge_count_groups.keys()):
        rules = edge_count_groups[edge_count]
        print(f"  {edge_count} edges: {len(rules)} rules - {rules[:10]}{'...' if len(rules) > 10 else ''}")

    # Save summary statistics to CSV
    summary_data = []
    for data in all_rules_data:
        stats = data['graph_stats']
        summary_data.append({
            'rule_number': stats['rule_number'],
            'rule_binary': data['rule_binary'],
            'num_nodes': stats['num_nodes'],
            'num_edges': stats['num_edges'],
            'density': stats['density'],
            'is_strongly_connected': stats['is_strongly_connected'],
            'max_in_degree': stats['max_in_degree'],
            'max_out_degree': stats['max_out_degree'],
            'avg_in_degree': stats['avg_in_degree'],
            'avg_out_degree': stats['avg_out_degree']
        })

    df = pd.DataFrame(summary_data)
    csv_file = "eca_rules_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nSummary data has been saved to: {csv_file}")


if __name__ == "__main__":
    analyze_all_rules()
