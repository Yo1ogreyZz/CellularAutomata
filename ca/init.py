import numpy as np

def random_init(width, p=0.5):
    """
    Random Bernoulli initial condition.
    """
    return (np.random.rand(width) < p).astype(np.uint8)