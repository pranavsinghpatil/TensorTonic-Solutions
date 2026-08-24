import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    
    _, counts = np.unique(y, return_counts=True)
    
    probabilities = counts / len(y)
    
    entropy_value = -np.sum(probabilities * np.log2(probabilities, where=(probabilities > 0)))
    
    return float(entropy_value)