import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 :
        return None
        
    N = X.shape[0]

    mean = np.mean(X, axis = 0)
    x_bar = X - mean

    c_mat = (x_bar.T @ x_bar) / (N - 1)

    return c_mat