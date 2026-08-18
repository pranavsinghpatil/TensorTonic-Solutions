import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A = np.asarray(A)
    if A.ndim != 2:
        return None
    r,c = A.shape
    if r != c:
        return None

    det = np.linalg.det(A)
    if abs(det) < 1e-10:
        return None
    A_inv = np.linalg.inv(A)

    return A_inv
    