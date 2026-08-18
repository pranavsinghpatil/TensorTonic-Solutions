import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.asarray(x)
    
    nu = np.exp(x) - np.exp(-x)
    de = np.exp(x) + np.exp(-x)

    return nu / de