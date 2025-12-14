"""
Rule to Graph Conversion for ECA

Implements three graph construction methods:
1. Truth-Table Graph: 8 nodes representing 3-bit neighborhoods
2. De Bruijn Graph: Transition graph showing rule propagation
3. Evolution Graph: Sampled dynamics from actual CA evolution
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


class ECARule:
    """Elementary Cellular Automaton Rule."""
    
    def __init__(self, rule_number: int):
        if not 0 <= rule_number < 256:
            raise ValueError(f"Rule number must be 0-255, got {rule_number}")
        
        self.rule_number = rule_number
        self.lookup = self._build_lookup_table()
    
    def _build_lookup_table(self) -> Dict[Tuple[int, int, int], int]:
        """Build lookup table from rule number."""
        binary = format(self.rule_number, '08b')[::-1]  # Reverse for correct indexing
        lookup = {}
        for i in range(8):
            neighborhood = ((i >> 2) & 1, (i >> 1) & 1, i & 1)
            lookup[neighborhood] = int(binary[i])
        return lookup
    
    def apply(self, neighborhood: Tuple[int, int, int]) -> int:
        """Apply rule to a 3-bit neighborhood."""
        return self.lookup[neighborhood]
    
    def get_output_pattern(self) -> np.ndarray:
        """Get 8-bit output pattern as array."""
        return np.array([self.lookup[(i>>2 & 1, i>>1 & 1, i & 1)] for i in range(8)])


def truth_table_graph(rule: ECARule) -> Dict:
    """
    Method A: Truth-Table Graph (Baseline)
    
    Creates a graph with 8 nodes, one for each 3-bit neighborhood pattern.
    Nodes are connected in binary sequence order.
    
    Node features: [left_bit, center_bit, right_bit, output_bit]
    
    Args:
        rule: ECARule object
        
    Returns:
        Dictionary with graph data
    """
    # Create 8 nodes for 8 neighborhoods
    nodes = list(range(8))
    
    # Node features: [3-bit input pattern + 1-bit output]
    node_features = np.zeros((8, 4))
    for i in range(8):
        left = (i >> 2) & 1
        center = (i >> 1) & 1
        right = i & 1
        output = rule.apply((left, center, right))
        node_features[i] = [left, center, right, output]
    
    # Create edges: connect in binary order (creates a path graph)
    # Also add reverse edges to make it undirected
    edges = []
    for i in range(7):
        edges.append((i, i + 1))
        edges.append((i + 1, i))
    
    # Add edges connecting nodes that differ by one bit
    for i in range(8):
        for j in range(i + 1, 8):
            hamming = bin(i ^ j).count('1')
            if hamming == 1:  # Adjacent in binary space
                if (i, j) not in edges:
                    edges.append((i, j))
                    edges.append((j, i))
    
    return {
        'nodes': nodes,
        'edges': edges,
        'node_features': node_features,
        'rule_number': rule.rule_number,
        'graph_type': 'truth_table',
        'num_nodes': 8,
        'num_edges': len(edges)
    }


def de_bruijn_graph(rule: ECARule) -> Dict:
    """
    Method B: De Bruijn / Dependency Graph
    
    Models transitions between 2-bit patterns based on 3-bit rule outputs.
    Shows how patterns propagate through the CA.
    
    Nodes: 4 nodes for 2-bit patterns (00, 01, 10, 11)
    Edges: (i, j) if 3-bit pattern [i, *] → j under the rule
    
    Args:
        rule: ECARule object
        
    Returns:
        Dictionary with graph data
    """
    # 4 nodes for 2-bit patterns
    nodes = list(range(4))  # 00, 01, 10, 11
    
    # Node features: [bit0, bit1, out_when_left_0, out_when_left_1]
    node_features = np.zeros((4, 4))
    for i in range(4):
        bit0 = (i >> 1) & 1
        bit1 = i & 1
        
        # Output when this pattern is left part and right bit is 0
        out_0 = rule.apply((bit0, bit1, 0))
        # Output when this pattern is left part and right bit is 1
        out_1 = rule.apply((bit0, bit1, 1))
        
        node_features[i] = [bit0, bit1, out_0, out_1]
    
    # Build edges: transitions based on overlapping bits
    edges = []
    edge_features = []
    
    for i in range(4):  # Source 2-bit pattern
        i_bits = ((i >> 1) & 1, i & 1)
        
        for r in [0, 1]:  # Right bit
            # The 3-bit pattern is (i's bits + r)
            output = rule.apply((i_bits[0], i_bits[1], r))
            
            # Next 2-bit pattern is (second bit of i, r)
            next_pattern = (i_bits[1] << 1) | r
            
            edges.append((i, next_pattern))
            # Edge feature: [right_bit, output]
            edge_features.append([r, output])
    
    edge_features = np.array(edge_features)
    
    return {
        'nodes': nodes,
        'edges': edges,
        'node_features': node_features,
        'edge_features': edge_features,
        'rule_number': rule.rule_number,
        'graph_type': 'dependency',
        'num_nodes': 4,
        'num_edges': len(edges)
    }


def evolution_graph(rule: ECARule, 
                   num_samples: int = 20,
                   time_steps: int = 50,
                   lattice_size: int = 50) -> Dict:
    """
    Method C: Evolution Graph (Sampled Dynamics)
    
    Samples actual CA evolution and builds a graph from observed states.
    Nodes represent unique configurations, edges represent transitions.
    
    Args:
        rule: ECARule object
        num_samples: Number of random initial conditions
        time_steps: Evolution steps per sample
        lattice_size: Size of CA lattice
        
    Returns:
        Dictionary with graph data
    """
    def evolve_ca(initial: np.ndarray, steps: int) -> np.ndarray:
        """Evolve CA for given steps."""
        lattice = initial.copy()
        history = [lattice.copy()]
        
        for _ in range(steps):
            new_lattice = np.zeros_like(lattice)
            for i in range(len(lattice)):
                left = lattice[(i - 1) % len(lattice)]
                center = lattice[i]
                right = lattice[(i + 1) % len(lattice)]
                new_lattice[i] = rule.apply((int(left), int(center), int(right)))
            lattice = new_lattice
            history.append(lattice.copy())
        
        return np.array(history)
    
    # Collect unique states and transitions
    states_map = {}  # hash -> (node_id, state)
    transitions = []
    node_id = 0
    
    for _ in range(num_samples):
        # Random initial condition
        initial = np.random.randint(0, 2, lattice_size)
        history = evolve_ca(initial, time_steps)
        
        # Process transitions
        for t in range(len(history) - 1):
            state_t = history[t]
            state_t1 = history[t + 1]
            
            hash_t = state_t.tobytes()
            hash_t1 = state_t1.tobytes()
            
            # Add states to map
            if hash_t not in states_map:
                states_map[hash_t] = (node_id, state_t)
                node_id += 1
            
            if hash_t1 not in states_map:
                states_map[hash_t1] = (node_id, state_t1)
                node_id += 1
            
            # Add transition
            id_t = states_map[hash_t][0]
            id_t1 = states_map[hash_t1][0]
            transitions.append((id_t, id_t1))
    
    # Build node features: aggregate statistics of each state
    num_nodes = len(states_map)
    node_features = np.zeros((num_nodes, 4))
    
    node_list = sorted(states_map.items(), key=lambda x: x[1][0])
    for _, (node_id, state) in node_list:
        density = np.mean(state)
        entropy = -np.sum([(p := np.mean(state == i)) * np.log2(p + 1e-10) 
                          for i in [0, 1]])
        # Simple pattern features
        transitions_01 = np.sum(np.diff(state) != 0)
        max_run = max([len(list(g)) for k, g in 
                      __import__('itertools').groupby(state)])
        
        node_features[node_id] = [density, entropy, transitions_01, max_run]
    
    # Remove duplicate edges
    edges = list(set(transitions))
    
    return {
        'nodes': list(range(num_nodes)),
        'edges': edges,
        'node_features': node_features,
        'rule_number': rule.rule_number,
        'graph_type': 'evolution',
        'num_nodes': num_nodes,
        'num_edges': len(edges)
    }


def convert_rule(rule_number: int,
                methods: List[str] = ['truth_table'],
                visualize: bool = False,
                verbose: bool = True) -> Dict[str, Dict]:
    """
    Convert ECA rule to graph representation(s).
    
    Args:
        rule_number: Rule number (0-255)
        methods: List of methods to use. Options:
                 'truth_table', 'dependency', 'evolution'
        visualize: Whether to visualize graphs (requires matplotlib)
        verbose: Print progress
        
    Returns:
        Dictionary mapping method names to graph data dictionaries
    """
    rule = ECARule(rule_number)
    results = {}
    
    method_functions = {
        'truth_table': truth_table_graph,
        'dependency': de_bruijn_graph,
        'evolution': evolution_graph
    }
    
    for method in methods:
        if method not in method_functions:
            warnings.warn(f"Unknown method '{method}', skipping")
            continue
        
        if verbose:
            print(f"Building {method} graph for Rule {rule_number}...")
        
        graph_data = method_functions[method](rule)
        results[method] = graph_data
        
        if verbose:
            print(f"  Nodes: {graph_data['num_nodes']}, "
                  f"Edges: {graph_data['num_edges']}")
    
    if visualize:
        try:
            visualize_graphs(results, rule_number)
        except ImportError:
            warnings.warn("Matplotlib not available, skipping visualization")
    
    return results


def visualize_graphs(graphs: Dict[str, Dict], rule_number: int) -> None:
    """
    Visualize graph representations.
    
    Requires matplotlib and networkx (optional dependencies).
    """
    import matplotlib.pyplot as plt
    try:
        import networkx as nx
    except ImportError:
        warnings.warn("NetworkX not available for visualization")
        return
    
    n_graphs = len(graphs)
    fig, axes = plt.subplots(1, n_graphs, figsize=(6 * n_graphs, 5))
    if n_graphs == 1:
        axes = [axes]
    
    for ax, (method, data) in zip(axes, graphs.items()):
        G = nx.DiGraph()
        G.add_nodes_from(data['nodes'])
        G.add_edges_from(data['edges'])
        
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax, with_labels=True, 
               node_color='lightblue', node_size=500,
               edge_color='gray', arrows=True)
        
        ax.set_title(f"Rule {rule_number} - {method.replace('_', ' ').title()}")
    
    plt.tight_layout()
    plt.show()


def build_multi_ic(rule_number: int,
                  num_ics: int = 10,
                  lattice_size: int = 100,
                  time_steps: int = 300) -> Dict:
    """
    Build evolution data for multiple initial conditions.
    
    Used for analyzing IC-sensitivity of controversial rules.
    
    Args:
        rule_number: Rule number
        num_ics: Number of random initial conditions
        lattice_size: Size of CA lattice
        time_steps: Evolution steps
        
    Returns:
        Dictionary with evolution data for each IC
    """
    rule = ECARule(rule_number)
    results = []
    
    for i in range(num_ics):
        initial = np.random.randint(0, 2, lattice_size)
        
        # Evolve
        history = [initial]
        lattice = initial.copy()
        
        for _ in range(time_steps):
            new_lattice = np.zeros_like(lattice)
            for j in range(len(lattice)):
                left = lattice[(j - 1) % len(lattice)]
                center = lattice[j]
                right = lattice[(j + 1) % len(lattice)]
                new_lattice[j] = rule.apply((int(left), int(center), int(right)))
            lattice = new_lattice
            history.append(lattice.copy())
        
        results.append({
            'ic_index': i,
            'initial_condition': initial,
            'evolution': np.array(history),
            'final_state': lattice
        })
    
    return {
        'rule_number': rule_number,
        'num_ics': num_ics,
        'lattice_size': lattice_size,
        'time_steps': time_steps,
        'evolutions': results
    }


# A Quick test
if __name__ == "__main__":
    print("Testing rule2graph module...")
    
    # Test Rule 30 (Class III - Chaotic)
    print("\n=== Rule 30 ===")
    graphs_30 = convert_rule(30, methods=['truth_table', 'dependency'])
    
    # Test Rule 110 (Class IV - Complex)
    print("\n=== Rule 110 ===")
    graphs_110 = convert_rule(110, methods=['truth_table'])
    
    print("\nrule2graph module working!")