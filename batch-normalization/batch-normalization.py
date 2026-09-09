import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)
    if x.ndim == 2:
        # For 2D input (N, D)
        axis = 0
        gamma = gamma.reshape(1, -1)
        beta = beta.reshape(1, -1)
    elif x.ndim == 4:
        # For 4D input (N, C, H, W)
        axis = (0, 2, 3)
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)
    else:
        raise ValueError("Input must be 2D or 4D")

    # Compute mean and variance
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)

    # Normalize
    x_normalized = (x - mean) / np.sqrt(var + eps)

    # Scale and shift
    out = gamma * x_normalized + beta

    return out