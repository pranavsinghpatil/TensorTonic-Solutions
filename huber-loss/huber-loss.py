import numpy as np

def huber_loss(y_true: list, y_pred: list, delta: float = 1.0) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    err = y_true - y_pred
    abs_err = np.abs(err)
    loss = np.where(abs_err <= delta, 
                    (err**2)/2, 
                    delta * (abs_err - (delta/2)))

    return float(np.mean(loss))