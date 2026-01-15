# CA (Cellular Automata) module

from . import evolve
from . import init
from . import rules

# Explicitly export init functions for better IDE support
from .init import random_init

# Optional imports (may require torch_geometric)
try:
    from . import debruijn
except ImportError:
    pass

try:
    from . import simple01
except ImportError:
    pass

__all__ = ['evolve', 'init', 'rules', 'random_init']