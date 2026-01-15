"""
Multi-view graph representations for radius-2 Elementary Cellular Automata.

This module provides 4 primary graph views that capture different structural
and dynamical aspects of CA rules:
- View 0 (Symbol): Input-output mapping abstraction
- View 1 (Lattice): Spatial topology and connectivity
- View 2 (De Bruijn): Local state sequence transitions
- View 3 (Dependency): Spatiotemporal causal dependencies
"""

import numpy as np
import torch
from torch_geometric.data import Data


def get_bits(value, n_bits):
    """Convert integer to bit list (MSB first)."""
    return [(value >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]


class CAGraphRepresentation:
    """
    Unified class for generating multiple graph representations of CA rules.

    Each view captures different aspects:
    - Symbol: Functional mapping (2 nodes, 32 edges)
    - Lattice: Physical connectivity (N nodes, 4N edges)
    - De Bruijn: Symbolic dynamics (32 nodes, 64 edges)
    - Dependency: Causal structure (T*W nodes, variable edges)
    """
    
    def __init__(self, rule_id):
        """
        Initialize graph generator for a specific CA rule.
        
        Args:
            rule_id: Integer in [0, 2^32-1] encoding the CA transition rule
        """
        self.rule_id = int(rule_id)
        self.rule_table = [(self.rule_id >> i) & 1 for i in range(32)]
        
        # Pre-compute rule influence flags (for dependency graph)
        self._compute_influence_flags()
    
    def _compute_influence_flags(self):
        """
        Compute which neighbor positions (left, center, right) influence output.
        Uses boolean derivative / sensitivity analysis.
        """
        # Left influence: exists (c,r) where f(0,c,r) != f(1,c,r)
        # In 5-bit encoding: bit 4 is left-2, bit 3 is left-1
        # We check if flipping left bits changes output
        self.left_influential = any(
            self.rule_table[i] != self.rule_table[i ^ 0b10000]  # flip bit 4 (left-2)
            for i in range(32)
        ) or any(
            self.rule_table[i] != self.rule_table[i ^ 0b01000]  # flip bit 3 (left-1)
            for i in range(32)
        )
        
        # Center influence: exists (l,r) where f(l,0,r) != f(l,1,r)
        # Bit 2 is center
        self.center_influential = any(
            self.rule_table[i] != self.rule_table[i ^ 0b00100]  # flip bit 2 (center)
            for i in range(32)
        )
        
        # Right influence: exists (l,c) where f(l,c,0) != f(l,c,1)
        # Bits 1 and 0 are right-1 and right-2
        self.right_influential = any(
            self.rule_table[i] != self.rule_table[i ^ 0b00010]  # flip bit 1 (right-1)
            for i in range(32)
        ) or any(
            self.rule_table[i] != self.rule_table[i ^ 0b00001]  # flip bit 0 (right-2)
            for i in range(32)
        )
    
    def get_symbol_graph(self):
        """
        View 0: Symbol Graph (2 nodes, 32 edges)
        
        Captures the rule as a state transition operator at the symbol level.
        
        Returns:
            PyG Data with:
                - x: [2, 1] node features (states 0 and 1)
                - edge_index: [2, 32] directed edges
                - edge_attr: [32, 6] neighborhood bits + output
                - y: rule_id
        """
        edge_list, edge_attrs = [], []
        
        for neighborhood in range(32):
            bits = get_bits(neighborhood, 5)
            source = bits[2]  # center bit
            target = self.rule_table[neighborhood]
            
            edge_list.append([source, target])
            edge_attrs.append(bits + [target])
        
        return Data(
            x=torch.tensor([[0.], [1.]], dtype=torch.float),
            edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attrs, dtype=torch.float),
            y=torch.tensor([self.rule_id], dtype=torch.long)
        )
    
    def get_lattice_graph(self, N=20, periodic=True):
        """
        View 1: Cell-Lattice Graph (N nodes, 4N edges for periodic)
        
        Represents the physical spatial structure and connectivity.
        Rule-independent - captures the substrate topology.
        
        Args:
            N: Number of cells (lattice width)
            periodic: If True, use periodic boundary; else open boundary
        
        Returns:
            PyG Data with:
                - x: [N, 1] position encodings
                - edge_index: [2, num_edges] undirected edges
                - y: rule_id
        """
        edge_list = []
        
        for i in range(N):
            # Radius-2: connect to neighbors at distance 1 and 2
            for offset in [-2, -1, 1, 2]:
                j = i + offset
                
                if periodic:
                    j = j % N
                    edge_list.append([i, j])
                else:
                    if 0 <= j < N:
                        edge_list.append([i, j])
        
        # Node features: normalized position
        x = torch.arange(N, dtype=torch.float).unsqueeze(1) / N
        
        return Data(
            x=x,
            edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
            y=torch.tensor([self.rule_id], dtype=torch.long)
        )
    
    def get_debruijn_graph(self):
        """
        View 2: De Bruijn Graph (32 nodes, 64 edges)
        
        Captures local state sequence transitions via overlap structure.
        Fundamental to symbolic dynamics analysis.
        
        Returns:
            PyG Data with:
                - x: [32, 5] 5-bit state representations
                - edge_index: [2, 64] directed edges
                - edge_attr: [64, 1] transition outputs
                - y: rule_id
        """
        edge_list, edge_attrs = [], []
        
        for source in range(32):
            # Two possible transitions by appending 0 or 1
            for next_bit in [0, 1]:
                # Shift left and append new bit, keep 5 bits
                target = ((source << 1) | next_bit) & 0b11111
                
                edge_list.append([source, target])
                edge_attrs.append([float(self.rule_table[source])])
        
        # Node features: bit representation
        x = torch.tensor(
            [get_bits(i, 5) for i in range(32)],
            dtype=torch.float
        )
        
        return Data(
            x=x,
            edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attrs, dtype=torch.float),
            y=torch.tensor([self.rule_id], dtype=torch.long)
        )
    
    def get_dependency_graph(self, T=4, W=7, weighted=True):
        """
        View 3: Dependency Graph (T*W nodes, variable edges)
        
        Represents spatiotemporal causal dependencies.
        Edges indicate which past cells influence future cells.
        
        Args:
            T: Number of time steps
            W: Lattice width
            weighted: If True, edge weights reflect influence strength
        
        Returns:
            PyG Data with:
                - x: [T*W, 2] spatiotemporal coordinates (t, i)
                - edge_index: [2, num_edges] directed edges
                - edge_attr: [num_edges, 1] influence weights (if weighted)
                - y: rule_id
        """
        edge_list = []
        edge_weights = [] if weighted else None
        
        for t in range(T - 1):
            for i in range(W):
                source_idx = t * W + i
                
                # Radius-2: check influences from positions at distance -2,-1,0,1,2
                influences = {
                    -2: self.left_influential,
                    -1: self.left_influential,
                     0: self.center_influential,
                     1: self.right_influential,
                     2: self.right_influential
                }
                
                for offset in [-2, -1, 0, 1, 2]:
                    j = i + offset
                    
                    # Open boundary: only connect if j is in range
                    if 0 <= j < W:
                        target_idx = (t + 1) * W + i
                        edge_list.append([source_idx + offset if offset else source_idx, target_idx])
                        
                        if weighted:
                            # Weight = 1.0 if influential, 0.2 if not
                            weight = 1.0 if influences[offset] else 0.2
                            edge_weights.append([weight])
        
        # Node features: (normalized_time, normalized_position)
        node_features = []
        for t in range(T):
            for i in range(W):
                node_features.append([t / T, i / W])
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        data = Data(
            x=x,
            edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
            y=torch.tensor([self.rule_id], dtype=torch.long)
        )
        
        if weighted:
            data.edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        return data
    
    def get_all_views(self, lattice_N=20, dependency_T=4, dependency_W=7):
        """
        Generate all 4 primary graph views for this rule.
        
        Args:
            lattice_N: Lattice width for View 1
            dependency_T: Time steps for View 3
            dependency_W: Width for View 3
        
        Returns:
            Dictionary mapping view names to PyG Data objects
        """
        return {
            'symbol': self.get_symbol_graph(),
            'lattice': self.get_lattice_graph(N=lattice_N),
            'debruijn': self.get_debruijn_graph(),
            'dependency': self.get_dependency_graph(T=dependency_T, W=dependency_W)
        }


class GraphFactory:
    """
    Factory class for backward compatibility and batch generation.
    """
    
    @staticmethod
    def get_views(rule_id, **kwargs):
        """
        Get all graph views for a single rule.
        
        Args:
            rule_id: CA rule ID
            **kwargs: Parameters passed to individual graph generators
        
        Returns:
            Dictionary of graph views
        """
        generator = CAGraphRepresentation(rule_id)
        return generator.get_all_views(**kwargs)
    
    @staticmethod
    def batch_generate(rule_ids, view_name='all', **kwargs):
        """
        Generate graphs for multiple rules.
        
        Args:
            rule_ids: List or array of rule IDs
            view_name: 'all', 'symbol', 'lattice', 'debruijn', or 'dependency'
            **kwargs: Parameters for graph generation
        
        Returns:
            List of PyG Data objects (or dict if view_name='all')
        """
        results = []
        
        for rule_id in rule_ids:
            generator = CAGraphRepresentation(rule_id)
            
            if view_name == 'all':
                results.append(generator.get_all_views(**kwargs))
            elif view_name == 'symbol':
                results.append(generator.get_symbol_graph())
            elif view_name == 'lattice':
                results.append(generator.get_lattice_graph(**kwargs))
            elif view_name == 'debruijn':
                results.append(generator.get_debruijn_graph())
            elif view_name == 'dependency':
                results.append(generator.get_dependency_graph(**kwargs))
            else:
                raise ValueError(f"Unknown view: {view_name}")
        
        return results


# Convenience functions for backward compatibility
def rule_to_symbol_graph(rule_id):
    return CAGraphRepresentation(rule_id).get_symbol_graph()

def rule_to_lattice_graph(rule_id, N=20):
    return CAGraphRepresentation(rule_id).get_lattice_graph(N=N)

def rule_to_debruijn_graph(rule_id):
    return CAGraphRepresentation(rule_id).get_debruijn_graph()

def rule_to_dependency_graph(rule_id, T=4, W=7):
    return CAGraphRepresentation(rule_id).get_dependency_graph(T=T, W=W)