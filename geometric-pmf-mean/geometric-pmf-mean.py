import numpy as np

def geometric_pmf_mean(k: list, p: float) -> dict:
    """
    Returns a dictionary with pmf and mean.
    """
    # Write code here
    k = np.asarray(k)
    r = ((1 - p) ** (k-1)) * p 
    return {
        "pmf" : r,
        "mean" : float(1 / p)
    }