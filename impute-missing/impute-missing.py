import numpy as np

def impute_missing(X: list, strategy: str = "mean") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as X.
    """
    # Write code here
    X = np.asarray(X, dtype=float).copy()

    if X.ndim == 1:
        mask = np.isnan(X)

        observed = ~mask

        if X[observed].size == 0:
            fill_value = 0.0
        elif strategy == "mean":
            fill_value = np.mean(X[observed])
        else:
            fill_value = np.median(X[observed])

        X[mask] = fill_value

        return X

    # 2D case
    rows, cols = X.shape

    for j in range(cols):

        mask = np.isnan(X[:, j])

        observed = ~mask

        if X[observed, j].size == 0:
            fill_value = 0.0
        elif strategy == "mean":
            fill_value = np.mean(X[observed, j])
        else:
            fill_value = np.median(X[observed, j])

        X[mask, j] = fill_value

    return X