import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Write code here
    x = np.asarray(x, dtype = float)

    vec = 1 + np.vectorize(math.erf)(x / np.sqrt(2))
    return np.asarray((x * vec) / 2)