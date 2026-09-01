import numpy as np

def dice_loss(p: list, y: list, eps: float = 1e-8) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    p = np.asarray(p)
    y = np.asarray(y)

    intersection = np.sum(p * y)
    sum_p = np.sum(p)
    sum_y = np.sum(y)

    dice = (2.0 * intersection + eps) / (sum_p + sum_y + eps)

    ld = 1.0 - dice

    return float(ld)