# Core classes and functions from rule2graph
from .rule2graph import (
    ECARule,
    TruthTableGraph,
    DependencyGraph,
    EvolutionGraph,
    convert_rule,
    batch_convert,
    PatternVocabularyGraph
)

# Utilities
from .utils import (
    # Wolfram classification
    get_wolfram_class,
    get_wolfram_class_id,
    get_wolfram_class_name,
    get_rules_by_class,
    get_class_distribution,
    get_canonical_rule,
    get_equivalent_rules,
    get_canonical_rules,
    build_equivalence_classes,
    WOLFRAM_CLASSES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    CLASS_NAMES,
    
    # Controversial rules
    CONTROVERSIAL_RULES,
    get_controversial_rules,
    is_controversial,
    get_controversy_info,
    
    # Data I/O
    save_graph_data,
    load_graph_data,
    
    # PyTorch Geometric (optional)
    to_pyg_data,
    batch_to_pyg_dataset,
    
    # Helpers
    get_representative_rules,
    get_test_rules,
    compute_graph_statistics,
)

__version__ = '1.0.0'

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
    
    # Controversial rules
    'CONTROVERSIAL_RULES',
    'get_controversial_rules',
    'is_controversial',
    'get_controversy_info',
    
    # Data I/O
    'save_graph_data',
    'load_graph_data',
    
    # PyTorch Geometric
    'to_pyg_data',
    'batch_to_pyg_dataset',
    
    # Helpers
    'get_representative_rules',
    'get_test_rules',
    'compute_graph_statistics',

    'get_canonical_rule',
    'get_equivalent_rules', 
    'get_canonical_rules',
    'build_equivalence_classes',
]