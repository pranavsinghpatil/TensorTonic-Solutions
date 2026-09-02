import math

def binary_focal_loss(predictions: list, targets: list, alpha: float, gamma: float) -> float:
    """
    Returns the mean binary focal loss as a float.
    """
    # Write code here
    total = 0

    for p, y in zip(predictions, targets):
        if y == 1 :
            pt = p
        if y == 0 :
            pt = 1 - p

        FL = -alpha *((1 - pt)**gamma) * math.log(pt)
        total += FL

    return total / len(predictions)