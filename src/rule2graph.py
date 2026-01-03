"""Graph construction methods for ECA rules"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class ECARule:
    """Elementary Cellular Automaton Rule"""
    
    def __init__(self, rule_number: int):
        if not 0 <= rule_number < 256:
            raise ValueError(f"Rule number must be 0-255, got {rule_number}")
        
        self.rule_number = rule_number
        self.rule_binary = format(rule_number, '08b')
        
        # Build lookup table: 3-bit pattern -> output
        self.lookup = {}
        for i in range(8):
            pattern = format(i, '03b')
            self.lookup[pattern] = int(self.rule_binary[7-i])
    
    def apply(self, left: int, center: int, right: int) -> int:
        """Apply rule to a 3-cell neighborhood"""
        pattern = f"{left}{center}{right}"
        return self.lookup[pattern]
    
    def evolve(self, initial_state: np.ndarray, steps: int) -> np.ndarray:
        """Evolve initial state for given steps, returns spacetime diagram"""
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
    """Truth-table graph: 8 nodes, one per 3-bit pattern"""
    
    def __init__(self, rule: ECARule):
        self.rule = rule
        self.num_nodes = 8
    
    def build(self) -> Dict:
        node_features = np.zeros((8, 4), dtype=float)
        for i in range(8):
            pattern = format(i, '03b')
            left, center, right = [int(b) for b in pattern]
            output = self.rule.lookup[pattern]
            node_features[i] = [left, center, right, output]
        
        edges = []
        for i in range(8):
            j = (i + 1) % 8
            edges.extend([(i, j), (j, i)])
        
        for i in range(8):
            for j in range(i+1, 8):
                if bin(i ^ j).count('1') == 1:
                    edges.extend([(i, j), (j, i)])
        
        return {
            'nodes': list(range(8)),
            'node_features': node_features,
            'edges': edges,
            'graph_type': 'truth_table',
            'rule_number': self.rule.rule_number,
            'num_nodes': 8,
            'num_edges': len(edges)
        }


class DependencyGraph:
    """Dependency graph: 4 nodes for 2-bit patterns"""
    
    def __init__(self, rule: ECARule):
        self.rule = rule
        self.num_nodes = 4
    
    def build(self) -> Dict:
        node_features = np.zeros((4, 2), dtype=float)
        for i in range(4):
            bit0 = (i >> 1) & 1
            bit1 = i & 1
            node_features[i] = [bit0, bit1]
        
        edges = []
        edge_features = []
        
        for i in range(4):
            i_bits = ((i >> 1) & 1, i & 1)
            for right_bit in [0, 1]:
                left, center = i_bits
                output = self.rule.apply(left, center, right_bit)
                next_pattern = (center << 1) | right_bit
                
                edges.append((i, next_pattern))
                edge_features.append([right_bit, output])
        
        edge_features = np.array(edge_features, dtype=float)
        
        return {
            'nodes': list(range(4)),
            'node_features': node_features,
            'edges': edges,
            'edge_features': edge_features,
            'graph_type': 'dependency',
            'rule_number': self.rule.rule_number,
            'num_nodes': 4,
            'num_edges': len(edges)
        }


class EvolutionGraph:
    """Spatiotemporal evolution graph"""
    
    def __init__(self, rule, width: int = 51, steps: int = 50, 
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
        
        num_nodes = self.width * (self.steps + 1)
        node_features = np.zeros((num_nodes, 3))
        edges = []
        
        def node_id(x, t):
            return t * self.width + x
        
        for t in range(self.steps + 1):
            state = history[t]
            for x in range(self.width):
                nid = node_id(x, t)
                cell_state = state[x]
                t_normalized = t / self.steps if self.steps > 0 else 0.0
                x_normalized = x / (self.width - 1) if self.width > 1 else 0.0
                node_features[nid] = [cell_state, t_normalized, x_normalized]
                
                if t < self.steps:
                    edges.append((nid, node_id(x, t + 1)))
                    edges.append((nid, node_id((x - 1) % self.width, t + 1)))
                    edges.append((nid, node_id((x + 1) % self.width, t + 1)))
                
                if x < self.width - 1:
                    edges.append((nid, node_id(x + 1, t)))
                if x > 0:
                    edges.append((nid, node_id(x - 1, t)))
        
        return {
            'nodes': list(range(num_nodes)),
            'edges': edges,
            'node_features': node_features,
            'rule_number': self.rule.rule_number,
            'graph_type': 'evolution',
            'num_nodes': num_nodes,
            'num_edges': len(edges)
        }


class PatternVocabularyGraph:
    """Pattern vocabulary graph for k-bit patterns"""
    
    def __init__(self, rule, pattern_size: int = 3):
        self.rule = rule
        self.k = pattern_size
        self.n_patterns = 2 ** pattern_size
    
    def build(self) -> Dict:
        nodes = list(range(self.n_patterns))
        node_features = []
        
        for p in range(self.n_patterns):
            bits = [(p >> i) & 1 for i in range(self.k)]
            stability = self._compute_stability(bits)
            node_features.append(bits + [stability])
        
        edges = []
        edge_features = []
        
        for p1 in range(self.n_patterns):
            for p2 in range(self.n_patterns):
                prob = self._transition_probability(p1, p2)
                if prob > 0:
                    edges.append((p1, p2))
                    edge_features.append([prob])
        
        return {
            'nodes': nodes,
            'edges': edges,
            'node_features': np.array(node_features),
            'edge_features': np.array(edge_features) if edge_features else None,
            'rule_number': self.rule.rule_number,
            'graph_type': 'pattern_vocabulary',
            'num_nodes': self.n_patterns,
            'num_edges': len(edges)
        }
    
    def _compute_stability(self, pattern: List[int]) -> float:
        survival_count = 0
        total = 0
        
        for left in [0, 1]:
            for right in [0, 1]:
                new_pattern = []
                for i in range(len(pattern)):
                    l = left if i == 0 else pattern[i-1]
                    c = pattern[i]
                    r = right if i == len(pattern)-1 else pattern[i+1]
                    new_pattern.append(self.rule.apply(l, c, r))
                
                if self._pattern_contains(new_pattern, pattern):
                    survival_count += 1
                total += 1
        
        return survival_count / total if total > 0 else 0.0
    
    def _pattern_contains(self, evolved: List[int], original: List[int]) -> bool:
        if len(evolved) < len(original):
            return False
        for i in range(len(evolved) - len(original) + 1):
            if evolved[i:i+len(original)] == original:
                return True
        return False
    
    def _transition_probability(self, p1: int, p2: int) -> float:
        bits1 = [(p1 >> i) & 1 for i in range(self.k)]
        bits2 = [(p2 >> i) & 1 for i in range(self.k)]
        
        if self.k < 2:
            return 0.0
        
        overlap = bits1[1:]
        prefix = bits2[:-1]
        
        if overlap != prefix:
            return 0.0
        
        if self.k == 2:
            right_bit = bits2[-1]
            for left in [0, 1]:
                evolved = []
                for i in range(self.k):
                    l = left if i == 0 else bits1[i-1]
                    c = bits1[i]
                    r = bits1[i+1] if i < self.k-1 else right_bit
                    evolved.append(self.rule.apply(l, c, r))
                if evolved == bits2:
                    return 1.0
            return 0.0
        
        right_bit = bits2[-1]
        
        evolved_middle = []
        for i in range(1, self.k):
            l = bits1[i-1]
            c = bits1[i]
            r = bits1[i+1] if i < self.k-1 else right_bit
            evolved_middle.append(self.rule.apply(l, c, r))
        
        if evolved_middle == bits2[1:]:
            return 1.0
        else:
            return 0.0


def convert_rule(rule_number: int,
                methods: List[str] = ['truth_table', 'dependency', 'evolution'],
                verbose: bool = False,
                **kwargs) -> Dict[str, Dict]:
    rule = ECARule(rule_number)
    results = {}
    
    if verbose:
        print(f"Converting Rule {rule_number}...")
    
    for method in methods:
        if method == 'truth_table':
            graph = TruthTableGraph(rule)
            results[method] = graph.build()
            if verbose:
                print(f"  Truth-Table: {results[method]['num_nodes']} nodes, {results[method]['num_edges']} edges")
        
        elif method == 'dependency':
            graph = DependencyGraph(rule)
            results[method] = graph.build()
            if verbose:
                print(f"  Dependency: {results[method]['num_nodes']} nodes, {results[method]['num_edges']} edges")
        
        elif method == 'evolution':
            graph = EvolutionGraph(rule, **kwargs)
            results[method] = graph.build()
            if verbose:
                print(f"  Evolution: {results[method]['num_nodes']} nodes, {results[method]['num_edges']} edges")
        
        elif method == 'pattern_vocabulary':
            pattern_size = kwargs.get('pattern_size', 3)
            graph = PatternVocabularyGraph(rule, pattern_size=pattern_size)
            results[method] = graph.build()
            if verbose:
                print(f"  Pattern Vocabulary: {results[method]['num_nodes']} nodes, {results[method]['num_edges']} edges")
        
        else:
            print(f"Warning: Unknown method '{method}', skipping")
    
    return results


def batch_convert(rule_numbers: List[int], 
                 methods: List[str] = ['truth_table', 'dependency', 'evolution'],
                 verbose: bool = False,
                 **kwargs) -> Dict[int, Dict[str, Dict]]:
    results = {}
    
    for rule_num in rule_numbers:
        results[rule_num] = convert_rule(rule_num, methods=methods, verbose=verbose, **kwargs)
    
    return results


# Quick test
if __name__ == "__main__":
    print("Testing rule2graph module...")
    
    # Test basic rule creation
    rule = ECARule(30)
    print(f"\nCreated {rule}")
    
    # Test evolution
    initial = np.array([0, 0, 0, 1, 0, 0, 0])
    spacetime = rule.evolve(initial, 5)
    print(f"Evolution shape: {spacetime.shape}")
    
    # Test all three graph types
    print("\n=== Testing Graph Representations ===")
    
    tt_graph = TruthTableGraph(rule)
    tt_data = tt_graph.build()
    print(f"Truth-Table: {tt_data['num_nodes']} nodes, {tt_data['num_edges']} edges")
    
    dep_graph = DependencyGraph(rule)
    dep_data = dep_graph.build()
    print(f"Dependency: {dep_data['num_nodes']} nodes, {dep_data['num_edges']} edges")
    
    evo_graph = EvolutionGraph(rule, width=21, steps=10)
    evo_data = evo_graph.build()
    print(f"Evolution: {evo_data['num_nodes']} nodes, {evo_data['num_edges']} edges")
    
    # Test convert_rule
    print("\n=== Testing convert_rule ===")
    graphs = convert_rule(110, methods=['truth_table', 'dependency'], verbose=True)
    