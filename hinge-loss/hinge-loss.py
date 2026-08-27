import numpy as np

def hinge_loss(y_true: list, y_score: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    yt = np.asarray(y_true)
    ys = np.asarray(y_score)

    loss = np.maximum(0, margin - yt * ys)

    if reduction == "mean":
        return float(np.mean(loss))

    elif reduction == "sum":
        return float(np.sum(loss))

    else:
        return loss