import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    # Write code here
    y = np.asarray(y)
    N = y.shape[0]
    
    if num_classes is None:
        k = np.max(y)  + 1
    else:
        k = num_classes
    Y = np.zeros((N, k))
    Y[np.arange(N), y] = 1
    return Y