import numpy as np

def info_nce_loss(Z1: list, Z2: list, temperature: float = 0.1) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    Z1 = np.array(Z1, dtype=np.float64)
    Z2 = np.array(Z2, dtype=np.float64)
    
    # Compute pairwise similarity logits
    logits = Z1 @ Z2.T / temperature
    
    # Stabilize: subtract row-wise maximum before exponentiation
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    
    # Exponentiate all entries
    exp_logits = np.exp(logits_shifted)
    
    # Sum of exponentials in each row (denominator)
    sum_exp = np.sum(exp_logits, axis=1)
    
    # Exponential of diagonal entries (numerator, positive pairs)
    positive_exp = np.exp(np.diag(logits_shifted))
    
    # Row-wise InfoNCE loss
    row_loss = -np.log(positive_exp / sum_exp)
    
    # Return mean loss as a Python float
    return float(np.mean(row_loss))