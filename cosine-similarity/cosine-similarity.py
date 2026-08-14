import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a = np.asarray(a)
    b = np.asarray(b)

    n_a = np.linalg.norm(a)
    n_b = np.linalg.norm(b)

    if n_a != 0 and n_b != 0:
        return float(np.dot(a , b) / (n_a * n_b))
    else:
        return 0.0
        