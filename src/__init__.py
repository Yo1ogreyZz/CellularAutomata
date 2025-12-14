"""
ECA-GNN: Graph Representation Learning for Elementary Cellular Automata
"""

from .rule2graph import ECARule, TruthTableGraph, DependencyGraph, EvolutionGraph
from .utils import (
    WOLFRAM_CLASSES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    CLASS_NAMES,
    CONTROVERSIAL_RULES,
    get_wolfram_class,
    get_wolfram_class_id,
    get_wolfram_class_name,
    get_rules_by_class,
    get_class_distribution,
    get_controversial_rules,
    is_controversial,
    get_controversy_info,
    to_pyg_data,
    save_graph_data,
    load_graph_data,
)

__version__ = '0.2.0'

__all__ = [
    'ECARule',
    'TruthTableGraph',
    'DependencyGraph',
    'EvolutionGraph',
    'WOLFRAM_CLASSES',
    'CLASS_TO_ID',
    'ID_TO_CLASS',
    'CLASS_NAMES',
    'CONTROVERSIAL_RULES',
    'get_wolfram_class',
    'get_wolfram_class_id',
    'get_wolfram_class_name',
    'get_rules_by_class',
    'get_class_distribution',
    'get_controversial_rules',
    'is_controversial',
    'get_controversy_info',
    'to_pyg_data',
    'save_graph_data',
    'load_graph_data',
]