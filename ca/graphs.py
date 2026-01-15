"""
Multi-view graph representations for radius-2 Elementary Cellular Automata.

This module provides 4 primary graph views that capture different structural
and dynamical aspects of CA rules:
- View 0 (Symbol): Input-output mapping abstraction (2 nodes, 16 edges)
- View 1 (Lattice): Spatial topology and connectivity (8 nodes, 32 edges)
- View 2 (De Bruijn): Local state sequence transitions (16 nodes, 32 edges)
- View 3 (Dependency): Spatiotemporal causal dependencies (28 nodes, ~147 edges)
"""

import torch
from torch_geometric.data import Data


def get_bits(value, n_bits):
    """Convert integer to bit list (MSB first)."""
    return [(value >> (n_bits - 1 - i)) & 1 for i in range(n_bits)]


class CAGraphRepresentation:
    """
    Unified class for generating multiple graph representations of CA rules.

    Each view captures different aspects:
    - Symbol: Functional mapping (2 nodes, 16 edges)
    - Lattice: Physical connectivity (8 nodes, 32 edges)
    - De Bruijn: Symbolic dynamics (16 nodes, 32 edges)
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
        View 0: Symbol Graph (2 nodes, 16 edges)
        
        Captures the rule as a state transition operator at the symbol level.
        Center bit is determined by source node, edges encode 4-bit neighbor configs.
        
        Returns:
            PyG Data with:
                - x: [2, 1] node features (states 0 and 1)
                - edge_index: [2, 16] directed edges
                - edge_attr: [16, 5] [left-2, left-1, right-1, right-2, output]
                - y: rule_id
        """
        edge_list, edge_attrs = [], []
        
        # For center=0, enumerate all 16 neighbor configurations
        for neighbor_config in range(16):  # 2^4 = 16
            # Extract 4 bits: left-2, left-1, right-1, right-2
            left2 = (neighbor_config >> 3) & 1
            left1 = (neighbor_config >> 2) & 1
            right1 = (neighbor_config >> 1) & 1
            right2 = neighbor_config & 1
            
            # Construct full 5-bit neighborhood for rule lookup
            # Order: [left-2, left-1, center, right-1, right-2]
            full_neighborhood = (left2 << 4) | (left1 << 3) | (0 << 2) | (right1 << 1) | right2
            output = self.rule_table[full_neighborhood]
            
            edge_list.append([0, output])  # center=0 -> output
            edge_attrs.append([left2, left1, right1, right2, output])
        
        # For center=1 (note: this gives us 32 total edges if we're not careful)
        # But we want only 16 edges total, so we DON'T duplicate
        # The 16 edges above already cover all cases when source node determines center
        
        return Data(
            x=torch.tensor([[0.], [1.]], dtype=torch.float),
            edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attrs, dtype=torch.float),
            y=torch.tensor([self.rule_id], dtype=torch.long)
        )
    
    def get_lattice_graph(self, N=8, periodic=True):
        """
        View 1: Cell-Lattice Graph (8 nodes, 32 edges for periodic)
        
        Represents the physical spatial structure and connectivity.
        Rule-independent - captures the substrate topology.
        
        Args:
            N: Number of cells (lattice width), default 8
            periodic: If True, use periodic boundary; else open boundary
        
        Returns:
            PyG Data with:
                - x: [N, 1] position encodings
                - edge_index: [2, num_edges] directed edges
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
        View 2: De Bruijn Graph (16 nodes, 32 edges)
        
        Captures local state sequence transitions via sliding window.
        Window: [left-2, left-1, center, right-1] (4 bits)
        
        Returns:
            PyG Data with:
                - x: [16, 4] 4-bit window representations
                - edge_index: [2, 32] directed edges
                - edge_attr: [32, 1] center output for source window
                - y: rule_id
        """
        edge_list, edge_attrs = [], []
        
        # 16 nodes representing 4-bit windows: [L2, L1, C, R1]
        for source_window in range(16):
            # Extract bits
            source_bits = get_bits(source_window, 4)
            left2, left1, center, right1 = source_bits
            
            # For each possible right-2 bit (0 or 1)
            for right2 in [0, 1]:
                # Construct 5-bit neighborhood for rule lookup
                neighborhood = (left2 << 4) | (left1 << 3) | (center << 2) | (right1 << 1) | right2
                output = self.rule_table[neighborhood]
                
                # Next window slides right: [L1, C, R1, R2]
                target_window = ((source_window << 1) | right2) & 0b1111
                
                edge_list.append([source_window, target_window])
                edge_attrs.append([float(output)])
        
        # Node features: 4-bit window representation
        x = torch.tensor(
            [get_bits(i, 4) for i in range(16)],
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
            T: Number of time steps (default 4)
            W: Lattice width (default 7)
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
                        source_node_idx = t * W + j  # Source at position j, time t
                        target_idx = (t + 1) * W + i  # Target at position i, time t+1
                        edge_list.append([source_node_idx, target_idx])
                        
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
        
        if weighted and edge_weights:
            data.edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        return data
    
    def get_all_views(self, lattice_N=8, dependency_T=4, dependency_W=7):
        """
        Generate all 4 primary graph views for this rule.
        
        Args:
            lattice_N: Lattice width for View 1 (default 8)
            dependency_T: Time steps for View 3 (default 4)
            dependency_W: Width for View 3 (default 7)
        
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

def rule_to_lattice_graph(rule_id, N=8):
    return CAGraphRepresentation(rule_id).get_lattice_graph(N=N)

def rule_to_debruijn_graph(rule_id):
    return CAGraphRepresentation(rule_id).get_debruijn_graph()

def rule_to_dependency_graph(rule_id, T=4, W=7):
    return CAGraphRepresentation(rule_id).get_dependency_graph(T=T, W=W)