import numpy as np

def percentiles(x: list, q: list) -> np.ndarray:
    """
    Returns a NumPy array of percentiles.
    """
    # Write code here
    x = np.sort(np.array(x))
    q = np.asarray(q)
    n = x.size
    
    positions = (q / 100.0 )* (n - 1)
    
    l = np.floor(positions).astype(int)
    u = np.ceil(positions).astype(int)
    
    w = positions - l
    
    result = (1 - w) * x[l] + w * x[u]
    
    return result