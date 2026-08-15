import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    x = np.asarray(x)
    x_bar = np.mean(x)
    n  = x.shape[0]
    sm = np.sum((x-x_bar)**2)
    vari = sm / (n-1)
    std = np.sqrt(vari)

    return vari, std