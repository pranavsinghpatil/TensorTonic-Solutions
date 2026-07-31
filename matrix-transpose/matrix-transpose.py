import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    n = len(A)
    m = len(A[0])

    ar = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            ar[i][j] = A[j][i]
    return ar
    
