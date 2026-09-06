import numpy as np

def wasserstein_critic_loss(real_scores: list, fake_scores: list) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    real_scores = np.array(real_scores, dtype=np.float64)
    fake_scores = np.array(fake_scores, dtype=np.float64)
    
    # Handle empty case (for safety)
    if real_scores.size == 0:
        mean_real = 0.0
    else:
        mean_real = np.mean(real_scores)
        
    if fake_scores.size == 0:
        mean_fake = 0.0
    else:
        mean_fake = np.mean(fake_scores)
    
    # L = mean(fake) - mean(real)
    loss = mean_fake - mean_real
    
    return float(loss)