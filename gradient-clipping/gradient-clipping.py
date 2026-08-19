import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    arr_g = np.asarray(g, dtype=float).copy()
    # ab_g = np.sqrt(np.sum(g*g))
    norm = np.linalg.norm(arr_g)

    if max_norm <= 0:
        return arr_g

    if norm <= max_norm:
        return arr_g
    if norm > max_norm and norm > 0:
        clipped_g = arr_g * (max_norm / norm)
        return clipped_g

    return arr_g