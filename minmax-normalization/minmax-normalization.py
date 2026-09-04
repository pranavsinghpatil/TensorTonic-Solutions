import numpy as np

def minmax_scale(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns a floating-point NumPy array matching the shape of X.
    """
    # Write code here
    X = np.asarray(X)

    xmin = np.min(X, axis=axis, keepdims=True)
    xmax = np.max(X, axis=axis, keepdims=True)

    data_range = xmax - xmin

    denominator = np.where(data_range > eps, data_range, 1.0)

    xn = (X - xmin) / denominator

    return xn