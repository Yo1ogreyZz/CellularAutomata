"""
Rule-to-Graph Conversion Methods for Elementary Cellular Automata

This module implements three different approaches to represent ECA rules as graphs:
- Method A: Truth-Table Graph (fixed 8-node structure)
- Method B: Dependency Graph (de Bruijn-style, 4 nodes)
- Method C: Evolution Graph (temporal dynamics)

Author: Hannah
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple, Dict, Optional
import warnings


class ECARule:
    """Elementary Cellular Automaton Rule representation"""
    
    def __init__(self, rule_number: int):
        if not 0 <= rule_number <= 255:
            raise ValueError("Rule number must be between 0 and 255")
        
        self.rule_number = rule_number
        self.rule_binary = format(rule_number, '08b')
        
        # Build lookup table: 3-bit pattern -> output
        self.lookup = {format(i, '03b'): int(self.rule_binary[7-i]) for i in range(8)}
            
    def apply(self, left: int, center: int, right: int) -> int:
        """Apply rule to a 3-cell neighborhood"""
        return self.lookup[f"{left}{center}{right}"]
    
    def evolve(self, initial_state: np.ndarray, steps: int) -> np.ndarray:
        """Evolve an initial state for given number of steps"""
        width = len(initial_state)
        spacetime = np.zeros((steps + 1, width), dtype=int)
        spacetime[0] = initial_state.copy()
        
        current = initial_state.copy()
        for t in range(steps):
            next_state = np.zeros(width, dtype=int)
            for i in range(width):
                left = current[(i-1) % width]
                center = current[i]
                right = current[(i+1) % width]
                next_state[i] = self.apply(left, center, right)
            current = next_state.copy()
            spacetime[t+1] = current
            
        return spacetime
    
    def __repr__(self):
        return f"ECARule({self.rule_number})"


class TruthTableGraph:
    """
    Method A: Truth-Table Graph
    
    Creates a uniform graph structure (same topology for all rules):
    - 8 nodes (one per 3-bit input pattern)
    - Node features: [left_bit, center_bit, right_bit, output_bit]
    - Edges: Sequential + Hamming-distance-1 connections
    """
    
    def __init__(self, rule: ECARule):
        self.rule = rule
        self.num_nodes = 8
        
    def build(self) -> Dict:
        """Build the truth-table graph"""
        # Node features: [left, center, right, output]
        node_features = np.array([
            [int(b) for b in format(i, '03b')] + [self.rule.lookup[format(i, '03b')]]
            for i in range(self.num_nodes)
        ], dtype=float)
        
        # Build edges
        edges = []
        
        # Sequential connections (bidirectional)
        for i in range(self.num_nodes):
            j = (i + 1) % self.num_nodes
            edges.extend([(i, j), (j, i)])
        
        # Hamming-distance-1 connections
        for i in range(self.num_nodes):
            for j in range(i+1, self.num_nodes):
                if bin(i ^ j).count('1') == 1:
                    edges.extend([(i, j), (j, i)])
        
        return {
            'nodes': list(range(self.num_nodes)),
            'node_features': node_features,
            'edges': edges,
            'graph_type': 'truth_table',
            'rule_number': self.rule.rule_number
        }
    
    def visualize(self, save_path: Optional[str] = None, show: bool = False):
        """Visualize the truth-table graph"""
        graph_data = self.build()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        G = nx.Graph()
        G.add_nodes_from(graph_data['nodes'])
        G.add_edges_from(graph_data['edges'])
        
        pos = nx.circular_layout(G)
        node_colors = graph_data['node_features'][:, 3]  # Output bit
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors,
                              cmap='RdYlGn',
                              node_size=800,
                              vmin=0, vmax=1,
                              ax=ax)
        
        # Draw edges
        seq_edges = [(i, (i+1) % 8) for i in range(8)]
        ham_edges = [e for e in graph_data['edges'] 
                     if e not in seq_edges and (e[1], e[0]) not in seq_edges]
        
        nx.draw_networkx_edges(G, pos, seq_edges, 
                              width=2, alpha=0.6, edge_color='black', ax=ax)
        nx.draw_networkx_edges(G, pos, ham_edges,
                              width=1, alpha=0.3, edge_color='blue', 
                              style='dashed', ax=ax)
        
        # Labels
        labels = {i: format(i, '03b') for i in range(8)}
        nx.draw_networkx_labels(G, pos, labels, font_size=12, 
                               font_weight='bold', ax=ax)
        
        ax.set_title(f'Truth-Table Graph | Rule {self.rule.rule_number}', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig


class DependencyGraph:
    """
    Method B: Dependency Graph (de Bruijn-style)
    
    Models transitions between local patterns:
    - 4 nodes (2-bit patterns: 00, 01, 10, 11)
    - Directed edges: AB -> BC with output as edge feature
    - Captures propagation dynamics
    """
    
    def __init__(self, rule: ECARule):
        self.rule = rule
        self.num_nodes = 4
        
    def build(self) -> Dict:
        """Build the dependency graph"""
        # Node features: 2-bit pattern values
        node_features = np.array([
            [int(b) for b in format(i, '02b')]
            for i in range(self.num_nodes)
        ], dtype=float)
        
        # Build directed edges
        edges = []
        edge_features = []
        
        for i in range(8):  # All 3-bit patterns
            pattern = format(i, '03b')
            A, B, C = [int(b) for b in pattern]
            output = self.rule.lookup[pattern]
            
            src = A * 2 + B  # Node for AB
            dst = B * 2 + C  # Node for BC
            
            edges.append((src, dst))
            edge_features.append([output])
        
        return {
            'nodes': list(range(self.num_nodes)),
            'node_features': node_features,
            'edges': edges,
            'edge_features': np.array(edge_features, dtype=float),
            'graph_type': 'dependency',
            'rule_number': self.rule.rule_number
        }
    
    def visualize(self, save_path: Optional[str] = None, show: bool = False):
        """Visualize the dependency graph"""
        graph_data = self.build()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        G = nx.DiGraph()
        G.add_nodes_from(graph_data['nodes'])
        G.add_edges_from(graph_data['edges'])
        
        # Square layout
        pos = {0: (0, 1), 1: (1, 1), 2: (0, 0), 3: (1, 0)}
        
        nx.draw_networkx_nodes(G, pos, node_color='lightgreen',
                              node_size=1500, ax=ax)
        
        edge_colors = [feat[0] for feat in graph_data['edge_features']]
        nx.draw_networkx_edges(G, pos,
                              edge_color=edge_colors,
                              edge_cmap=plt.cm.RdYlGn,
                              width=3,
                              arrows=True,
                              arrowsize=20,
                              connectionstyle='arc3,rad=0.1',
                              edge_vmin=0,
                              edge_vmax=1,
                              ax=ax)
        
        labels = {i: format(i, '02b') for i in range(4)}
        nx.draw_networkx_labels(G, pos, labels, font_size=14, 
                               font_weight='bold', ax=ax)
        
        ax.set_title(f'Dependency Graph | Rule {self.rule.rule_number}',
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.3, 1.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig


class EvolutionGraph:
    """
    Method C: Evolution Graph
    
    Samples short-term evolution:
    - Nodes: Time steps with statistical features
    - Node features: [density, block_density, entropy]
    - Edges: Temporal connections
    """
    
    def __init__(self, rule: ECARule, 
                 width: int = 21, 
                 steps: int = 10,
                 initial_density: float = 0.5,
                 seed: int = 42):
        self.rule = rule
        self.width = width
        self.steps = steps
        self.initial_density = initial_density
        self.seed = seed
        
    def _compute_features(self, state: np.ndarray) -> np.ndarray:
        """Compute statistical features for a state"""
        density = state.mean()
        
        # Count blocks (contiguous 1s)
        blocks = np.sum(np.diff(np.concatenate([[0], state, [0]])) == 1)
        block_density = blocks / len(state)
        
        # Binary entropy
        if 0 < density < 1:
            entropy = -density * np.log2(density) - (1-density) * np.log2(1-density)
        else:
            entropy = 0.0
        
        return np.array([density, block_density, entropy])
    
    def build(self) -> Dict:
        """Build the evolution graph"""
        np.random.seed(self.seed)
        initial = (np.random.random(self.width) < self.initial_density).astype(int)
        
        spacetime = self.rule.evolve(initial, self.steps)
        
        node_features = np.array([
            self._compute_features(spacetime[t]) 
            for t in range(self.steps + 1)
        ])
        
        # Temporal connections (directed)
        edges = [(t, t+1) for t in range(self.steps)]
        
        return {
            'nodes': list(range(self.steps + 1)),
            'node_features': node_features,
            'edges': edges,
            'spacetime': spacetime,
            'graph_type': 'evolution',
            'rule_number': self.rule.rule_number
        }
    
    def visualize(self, save_path: Optional[str] = None, show: bool = False):
        """Visualize the evolution graph"""
        graph_data = self.build()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Spacetime diagram
        spacetime = graph_data['spacetime']
        ax1.imshow(spacetime, cmap='binary', interpolation='nearest', aspect='auto')
        ax1.set_xlabel('Space')
        ax1.set_ylabel('Time')
        ax1.set_title(f'Spacetime Evolution | Rule {self.rule.rule_number}')
        
        # Graph structure
        G = nx.DiGraph()
        G.add_nodes_from(graph_data['nodes'])
        G.add_edges_from(graph_data['edges'])
        
        pos = {i: (i, 0) for i in range(len(graph_data['nodes']))}
        node_colors = graph_data['node_features'][:, 0]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              cmap='viridis', node_size=500, ax=ax2)
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, ax=ax2)
        
        labels = {i: f't={i}' for i in range(len(graph_data['nodes']))}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax2)
        
        ax2.set_title('Evolution Graph')
        ax2.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
        
        return fig


# Utility Functions
def convert_rule(rule_number: int, 
                 methods: List[str] = ['truth_table', 'dependency', 'evolution'],
                 output_dir: Optional[str] = None,
                 visualize: bool = True,
                 verbose: bool = False) -> Dict:
    """
    Convert a rule to graph representations.
    
    Args:
        rule_number: ECA rule number (0-255)
        methods: List of methods to use
        output_dir: Directory to save visualizations (None = no save)
        visualize: Whether to generate visualizations
        verbose: Whether to print progress
        
    Returns:
        Dictionary with graph data for each method
    """
    import os
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    rule = ECARule(rule_number)
    results = {}
    
    if verbose:
        print(f"Processing Rule {rule_number}")
    
    # Method A: Truth-Table Graph
    if 'truth_table' in methods:
        graph_a = TruthTableGraph(rule)
        results['truth_table'] = graph_a.build()
        
        if visualize and output_dir:
            save_path = f"{output_dir}/rule_{rule_number}_truth_table.png"
            graph_a.visualize(save_path=save_path, show=False)
        
        if verbose:
            print(f"  ✓ Truth-Table: {len(results['truth_table']['nodes'])} nodes, "
                  f"{len(results['truth_table']['edges'])} edges")
    
    # Method B: Dependency Graph
    if 'dependency' in methods:
        graph_b = DependencyGraph(rule)
        results['dependency'] = graph_b.build()
        
        if visualize and output_dir:
            save_path = f"{output_dir}/rule_{rule_number}_dependency.png"
            graph_b.visualize(save_path=save_path, show=False)
        
        if verbose:
            print(f"  ✓ Dependency: {len(results['dependency']['nodes'])} nodes, "
                  f"{len(results['dependency']['edges'])} edges")
    
    # Method C: Evolution Graph
    if 'evolution' in methods:
        graph_c = EvolutionGraph(rule, width=21, steps=10)
        results['evolution'] = graph_c.build()
        
        if visualize and output_dir:
            save_path = f"{output_dir}/rule_{rule_number}_evolution.png"
            graph_c.visualize(save_path=save_path, show=False)
        
        if verbose:
            print(f"  ✓ Evolution: {len(results['evolution']['nodes'])} nodes, "
                  f"{len(results['evolution']['edges'])} edges")
    
    return results


def batch_convert(rule_list: List[int], **kwargs) -> Dict:
    """
    Batch convert multiple rules to graphs.
    
    Args:
        rule_list: List of rule numbers
        **kwargs: Arguments passed to convert_rule()
        
    Returns:
        Dictionary mapping rule numbers to their graph representations
    """
    results = {}
    verbose = kwargs.get('verbose', False)
    
    for rule_num in rule_list:
        try:
            results[rule_num] = convert_rule(rule_num, **kwargs)
        except Exception as e:
            if verbose:
                print(f"  ✗ Error processing rule {rule_num}: {e}")
    
    return results


# Main
if __name__ == "__main__":
    # Example usage
    print("ECA Rule-to-Graph Converter")
    print("=" * 50)
    
    # Convert Rule 110 as example
    rule_num = 110
    print(f"\nConverting Rule {rule_num}...")
    
    results = convert_rule(
        rule_num, 
        output_dir='./graphs',
        visualize=True,
        verbose=True
    )
    
    print(f"\n✓ Completed! Visualizations saved to ./graphs/")