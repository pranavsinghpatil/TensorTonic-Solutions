import numpy as np

def swish(x: list) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x = np.asarray(x, dtype=float)

    sig = np.exp(-np.logaddexp(0.0, -x)) 

    return x * sig