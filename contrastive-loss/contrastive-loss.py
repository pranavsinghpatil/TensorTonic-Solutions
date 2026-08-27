import numpy as np

def contrastive_loss(a: list, b: list, y: list, margin: float = 1.0, reduction: str = "mean") -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y)

    if a.ndim == 1:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)

    diff = abs(a - b)
    distance = np.linalg.norm(diff, axis=1)

    positive = y * distance ** 2
    negative = (1 - y) * np.maximum(0, margin - distance) ** 2
    total = positive + negative

    if reduction == 'mean' :
        return float(np.mean(total))
    else :
        return float(np.sum(total))
    