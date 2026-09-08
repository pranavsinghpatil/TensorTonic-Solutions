import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    """
    Returns the coefficient of determination as a Python float.
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    y_mean = np.mean(y_true)

    num = np.sum(np.square(y_true - y_pred))
    den = np.sum(np.square(y_true - y_mean))

    if den == 0:
        return 1.0 if num == 0 else 0.0

    r2 = 1 - (num / den)
    return float(r2)