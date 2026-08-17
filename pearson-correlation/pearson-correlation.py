import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    if X.ndim != 2:
        return None
    N, D = X.shape
    if N < 2:
        return None
    
    X_bar = np.mean(X, axis=0)
    n = X.shape[0]
    xc = X - X_bar

    cov = (xc.T @ xc ) / (n-1)

    std_devs = np.std(X, axis=0, ddof=1)
    denominator = np.outer(std_devs, std_devs)
    correlation = cov / denominator

    zero_var = std_devs == 0

    zero_mask = zero_var[:, None] | zero_var[None, :]
    correlation[zero_mask] = np.nan
    # np.fill_diagonal(correlation, 1.0)

    return correlation

    