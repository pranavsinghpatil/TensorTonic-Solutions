import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.asarray(x)
    p = np.asarray(p)
    
    if x.shape != p.shape: return

    if not np.allclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError()

    return float(np.sum(x*p))