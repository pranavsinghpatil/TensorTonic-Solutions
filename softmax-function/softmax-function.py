import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x)
    if x.ndim == 1:
        mx = np.max(x)
        so = np.exp(x - mx) / (np.sum(np.exp(x - mx)))
        return so
    else:
        x_ro = x - np.max(x, axis=1, keepdims=True)
        ex = np.exp(x_ro)
        sm = np.sum(ex, axis=1, keepdims=True)
        return ex / sm

    return None
        