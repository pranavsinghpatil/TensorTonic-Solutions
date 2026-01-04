import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    n = len(x)
    xn = np.array(x)
    m = np.mean(xn)
    v = np.sum((xn - m)**2 / (n -1))
    r = np.sqrt(v)

    return (v , r)
    