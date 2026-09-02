import math

def label_smoothing_loss(predictions: list, target: int, epsilon: float) -> float:
    """
    Returns cross-entropy loss for the smoothed target distribution.
    """
    # Write code here
    K = len(predictions)
    cl = []
    sm = 0
    
    for i in range(K):
        if i == target:
            cl.append(epsilon/K + (1-epsilon))
            sm += cl[i] * math.log(predictions[i])
        else:
            cl.append(epsilon/K )
            sm += cl[i] * math.log(predictions[i])
    
    return -sm