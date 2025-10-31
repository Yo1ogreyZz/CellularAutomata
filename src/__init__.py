"""
ECA-GNN: Graph Representation Learning for Elementary Cellular Automata
========================================================================

This package converts ECA rules into graph representations for GNN analysis.

Main Components:
---------------
- ECARule: Rule representation and evolution
- TruthTableGraph: Fixed 8-node structure
- DependencyGraph: de Bruijn-style 4-node graph
- EvolutionGraph: Temporal dynamics graph
- convert_rule: Main conversion function
- batch_convert: Batch processing

Quick Start:
-----------
>>> from src import ECARule, TruthTableGraph, convert_rule
>>> 
>>> # Method 1: Direct graph building
>>> rule = ECARule(110)
>>> graph = TruthTableGraph(rule)
>>> data = graph.build()
>>> 
>>> # Method 2: Using convenience function
>>> results = convert_rule(110, methods=['truth_table'], verbose=True)
"""

# Core classes and functions from rule2graph
from .rule2graph import (
    ECARule,
    TruthTableGraph,
    DependencyGraph,
    EvolutionGraph,
    convert_rule,
    batch_convert
)

# Utilities
from .utils import (
    # Wolfram classification
    get_wolfram_class,
    get_wolfram_class_id,
    get_wolfram_class_name,
    get_rules_by_class,
    get_class_distribution,
    WOLFRAM_CLASSES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    CLASS_NAMES,
    
    # Data I/O
    save_graph_data,
    load_graph_data,
    
    # PyTorch Geometric
    to_pyg_data,
    batch_to_pyg_dataset,
    
    # Helpers
    batch_build_graphs,
    compute_graph_statistics,
    batch_compute_statistics,
    get_representative_rules,
    get_test_rules,
)

__version__ = '0.2.0'

__all__ = [
    # Core classes
    'ECARule',
    'TruthTableGraph',
    'DependencyGraph',
    'EvolutionGraph',
    
    # Main functions
    'convert_rule',
    'batch_convert',
    
    # Wolfram classification
    'get_wolfram_class',
    'get_wolfram_class_id',
    'get_wolfram_class_name',
    'get_rules_by_class',
    'get_class_distribution',
    'WOLFRAM_CLASSES',
    'CLASS_TO_ID',
    'ID_TO_CLASS',
    'CLASS_NAMES',
    
    # Data I/O
    'save_graph_data',
    'load_graph_data',
    
    # PyTorch Geometric
    'to_pyg_data',
    'batch_to_pyg_dataset',
    
    # Helpers
    'batch_build_graphs',
    'compute_graph_statistics',
    'batch_compute_statistics',
    'get_representative_rules',
    'get_test_rules',
]