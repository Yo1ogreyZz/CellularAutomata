"""
Rule to Graph Conversion for ECA
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class ECARule:
    """Elementary Cellular Automaton Rule."""
    
    def __init__(self, rule_number: int):
        if not 0 <= rule_number < 256:
            raise ValueError(f"Rule number must be 0-255, got {rule_number}")
        
        self.rule_number = rule_number
        self.lookup = self._build_lookup_table()
    
    def _build_lookup_table(self) -> Dict[Tuple[int, int, int], int]:
        """Build lookup table from rule number."""
        binary = format(self.rule_number, '08b')[::-1]
        lookup = {}
        for i in range(8):
            neighborhood = ((i >> 2) & 1, (i >> 1) & 1, i & 1)
            lookup[neighborhood] = int(binary[i])
        return lookup
    
    def apply(self, neighborhood: Tuple[int, int, int]) -> int:
        """Apply rule to a 3-bit neighborhood."""
        return self.lookup[neighborhood]
    
    def evolve(self, initial: np.ndarray, steps: int) -> np.ndarray:
        """Evolve CA for given steps."""
        lattice = initial.copy()
        history = [lattice.copy()]
        
        for _ in range(steps):
            new_lattice = np.zeros_like(lattice)
            for i in range(len(lattice)):
                left = lattice[(i - 1) % len(lattice)]
                center = lattice[i]
                right = lattice[(i + 1) % len(lattice)]
                new_lattice[i] = self.apply((int(left), int(center), int(right)))
            lattice = new_lattice
            history.append(lattice.copy())
        
        return np.array(history)


class TruthTableGraph:
    """Truth-Table Graph (8 nodes)"""
    
    def __init__(self, rule: ECARule):
        self.rule = rule
    
    def build(self) -> Dict:
        nodes = list(range(8))
        
        node_features = np.zeros((8, 4))
        for i in range(8):
            left = (i >> 2) & 1
            center = (i >> 1) & 1
            right = i & 1
            output = self.rule.apply((left, center, right))
            node_features[i] = [left, center, right, output]
        
        edges = []
        for i in range(7):
            edges.append((i, i + 1))
            edges.append((i + 1, i))
        
        for i in range(8):
            for j in range(i + 1, 8):
                hamming = bin(i ^ j).count('1')
                if hamming == 1:
                    if (i, j) not in edges:
                        edges.append((i, j))
                        edges.append((j, i))
        
        return {
            'nodes': nodes,
            'edges': edges,
            'node_features': node_features,
            'rule_number': self.rule.rule_number,
            'graph_type': 'truth_table',
            'num_nodes': 8,
            'num_edges': len(edges)
        }


class DependencyGraph:
    """Dependency Graph (4 nodes)"""
    
    def __init__(self, rule: ECARule):
        self.rule = rule
    
    def build(self) -> Dict:
        nodes = list(range(4))
        
        node_features = np.zeros((4, 2))
        for i in range(4):
            bit0 = (i >> 1) & 1
            bit1 = i & 1
            node_features[i] = [bit0, bit1]
        
        edges = []
        edge_features = []
        
        for i in range(4):
            i_bits = ((i >> 1) & 1, i & 1)
            
            for r in [0, 1]:
                output = self.rule.apply((i_bits[0], i_bits[1], r))
                next_pattern = (i_bits[1] << 1) | r
                
                edges.append((i, next_pattern))
                edge_features.append([r, output])
        
        edge_features = np.array(edge_features)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'node_features': node_features,
            'edge_features': edge_features,
            'rule_number': self.rule.rule_number,
            'graph_type': 'dependency',
            'num_nodes': 4,
            'num_edges': len(edges)
        }


class EvolutionGraph:
    """Evolution Graph (dynamic)"""
    
    def __init__(self, rule: ECARule, width: int = 51, steps: int = 20, 
                 initial_density: float = 0.5, seed: int = 42):
        self.rule = rule
        self.width = width
        self.steps = steps
        self.initial_density = initial_density
        self.seed = seed
    
    def build(self) -> Dict:
        np.random.seed(self.seed)
        initial = np.random.binomial(1, self.initial_density, self.width)
        history = self.rule.evolve(initial, self.steps)
        
        states_map = {}
        transitions = []
        node_id = 0
        
        for t in range(len(history) - 1):
            state_t = history[t]
            state_t1 = history[t + 1]
            
            hash_t = state_t.tobytes()
            hash_t1 = state_t1.tobytes()
            
            if hash_t not in states_map:
                states_map[hash_t] = (node_id, state_t)
                node_id += 1
            
            if hash_t1 not in states_map:
                states_map[hash_t1] = (node_id, state_t1)
                node_id += 1
            
            id_t = states_map[hash_t][0]
            id_t1 = states_map[hash_t1][0]
            transitions.append((id_t, id_t1))
        
        num_nodes = len(states_map)
        node_features = np.zeros((num_nodes, 3))
        
        node_list = sorted(states_map.items(), key=lambda x: x[1][0])
        for _, (node_id, state) in node_list:
            density = np.mean(state)
            p0 = np.mean(state == 0)
            p1 = np.mean(state == 1)
            entropy = -p0 * np.log2(p0 + 1e-10) - p1 * np.log2(p1 + 1e-10)
            transitions_01 = np.sum(np.diff(state) != 0)
            
            node_features[node_id] = [density, entropy, transitions_01]
        
        edges = list(set(transitions))
        
        return {
            'nodes': list(range(num_nodes)),
            'edges': edges,
            'node_features': node_features,
            'rule_number': self.rule.rule_number,
            'graph_type': 'evolution',
            'num_nodes': num_nodes,
            'num_edges': len(edges)
        }