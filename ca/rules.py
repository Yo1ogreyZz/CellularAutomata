import numpy as np

def rule_id_to_table(rule_id):
    """
    Convert a 32-bit rule_id into a lookup table of size 32.
    Index corresponds to 5-bit neighborhood interpreted as an integer.
    """
    # Ensure Python int
    rule_id = int(rule_id)

    # Binary representation, LSB = neighborhood 0
    bits = [(rule_id >> i) & 1 for i in range(32)]
    table = np.array(bits, dtype=np.uint8)
    return table


def neighborhood_to_index(neigh):
    """
    neigh: iterable of 5 bits (0/1)
    order: [x-2, x-1, x, x+1, x+2]
    """
    idx = 0
    for b in neigh:
        idx = (idx << 1) | int(b)
    return idx
