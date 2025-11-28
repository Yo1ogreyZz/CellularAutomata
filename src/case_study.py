"""
Case Study Framework for Controversial Rules

Provides tools to analyze why certain rules are hard to classify.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from .rule2graph import ECARule, TruthTableGraph, DependencyGraph, EvolutionGraph
from .utils import get_wolfram_class, get_controversy_info

class RuleCaseStudy:
    """Framework for detailed analysis of a single rule"""
    
    def __init__(self, rule_number: int):
        self.rule_number = rule_number
        self.rule = ECARule(rule_number)
        self.wolfram_class = get_wolfram_class(rule_number)
        self.controversy = get_controversy_info(rule_number)
        
    def extract_all_graphs(self) -> Dict:
        """Extract all three graph representations"""
        return {
            'truth_table': TruthTableGraph(self.rule).build(),
            'dependency': DependencyGraph(self.rule).build(),
            'evolution': EvolutionGraph(self.rule, width=50, steps=100).build_multi_ic(n_samples=4)
        }
    
    def test_ic_sensitivity(self, n_tests: int = 10, steps: int = 200) -> Dict:
        """
        Test how sensitive the rule is to initial conditions.
        
        Returns statistical measures across different ICs.
        """
        np.random.seed(42)
        
        results = {
            'final_densities': [],
            'entropies': [],
            'hamming_distances': [],
            'spacetimes': []
        }
        
        for i in range(n_tests):
            # Random IC
            initial = np.random.binomial(1, 0.5, 100)
            spacetime = self.rule.evolve(initial, steps)
            
            # Compute metrics
            final_density = spacetime[-1].mean()
            
            # Entropy over time
            entropies = []
            for t in range(len(spacetime)):
                p = spacetime[t].mean()
                if 0 < p < 1:
                    ent = -p * np.log2(p) - (1-p) * np.log2(1-p)
                else:
                    ent = 0
                entropies.append(ent)
            
            # Hamming distance between consecutive steps
            hamming = [np.sum(spacetime[t] != spacetime[t-1]) 
                      for t in range(1, len(spacetime))]
            
            results['final_densities'].append(final_density)
            results['entropies'].append(np.mean(entropies))
            results['hamming_distances'].append(np.mean(hamming))
            results['spacetimes'].append(spacetime)
        
        # Compute variability measures
        results['density_std'] = np.std(results['final_densities'])
        results['entropy_std'] = np.std(results['entropies'])
        results['hamming_std'] = np.std(results['hamming_distances'])
        
        return results
    
    def compare_with_neighbors(self, neighbor_rules: List[int]) -> Dict:
        """
        Compare this rule with structurally/dynamically similar rules.
        
        Args:
            neighbor_rules: List of rule numbers to compare with
        """
        comparisons = {}
        
        # Get own graphs
        own_graphs = self.extract_all_graphs()
        
        for neighbor in neighbor_rules:
            neighbor_obj = RuleCaseStudy(neighbor)
            neighbor_graphs = neighbor_obj.extract_all_graphs()
            
            # Compare truth table features
            own_tt = own_graphs['truth_table']['node_features']
            neighbor_tt = neighbor_graphs['truth_table']['node_features']
            tt_distance = np.linalg.norm(own_tt - neighbor_tt)
            
            # Compare dependency graph edge features
            own_dep_edges = own_graphs['dependency']['edge_features']
            neighbor_dep_edges = neighbor_graphs['dependency']['edge_features']
            dep_distance = np.linalg.norm(own_dep_edges - neighbor_dep_edges)
            
            comparisons[neighbor] = {
                'truth_table_distance': tt_distance,
                'dependency_distance': dep_distance,
                'neighbor_class': get_wolfram_class(neighbor)
            }
        
        return comparisons
    
    def generate_report(self, output_path: str = None):
        """Generate a comprehensive case study report"""
        print(f"\n{'='*60}")
        print(f"CASE STUDY: Rule {self.rule_number}")
        print(f"{'='*60}")
        
        print(f"\nWolfram Classification: {self.wolfram_class}")
        
        if self.controversy:
            print(f"\nCONTROVERSIAL RULE")
            print(f"  Disputed as: {self.controversy['disputed_as']}")
            print(f"  Reason: {self.controversy['reason']}")
            print(f"  Priority: {self.controversy['priority']}")
        
        print(f"\n--- IC Sensitivity Analysis ---")
        ic_results = self.test_ic_sensitivity(n_tests=10)
        print(f"  Final density std: {ic_results['density_std']:.4f}")
        print(f"  Entropy std: {ic_results['entropy_std']:.4f}")
        print(f"  Hamming distance std: {ic_results['hamming_std']:.4f}")
        
        if ic_results['density_std'] > 0.1:
            print("HIGH IC SENSITIVITY DETECTED")
        
        print(f"\n{'='*60}\n")

# Convenience function
def analyze_rule_54():
    """Quick analysis for Rule 54"""
    study = RuleCaseStudy(54)
    
    # Compare with other Class IV rules
    neighbors = [41, 106, 110]
    comparisons = study.compare_with_neighbors(neighbors)
    
    print("\n=== Rule 54 vs Other Class IV Rules ===")
    for neighbor, data in comparisons.items():
        print(f"\nRule {neighbor} ({data['neighbor_class']}):")
        print(f"  Truth-table distance: {data['truth_table_distance']:.4f}")
        print(f"  Dependency distance: {data['dependency_distance']:.4f}")
    
    # Generate full report
    study.generate_report()
    
    return study