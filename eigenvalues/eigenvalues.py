import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    try:
        matrix = np.asarray(matrix)
    except (ValueError, TypeError):
        return None
    if matrix.ndim != 2:
        return None
    N, D = matrix.shape
    if N != D:
        return None
    if N == 0:
        return None

    eigenvalues = np.linalg.eigvals(matrix)

    real = eigenvalues.real
    imag = eigenvalues.imag

    idx = np.lexsort((imag, real))

    return eigenvalues[idx]