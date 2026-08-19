import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Write code here
    p = np.asarray(p)
    q = np.asarray(q)

    mask = p > 0    
    q_stable = q + eps

    return np.sum(p[mask] * np.log(p[mask] / q_stable[mask]))
