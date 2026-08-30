import numpy as np

def focal_loss(p: list, y: list, gamma: float = 2.0) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    p = np.asarray(p, dtype=float)
    y = np.asarray(y)

    p = np.clip(p, 1e-15, 1.0 - 1e-15)

    loss = (
        -(1 - p) ** gamma * y * np.log(p)
        - p ** gamma * (1 - y) * np.log1p(-p)
    )

    return float(np.mean(loss))