import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    N = y_pred.shape[0]
    se = np.sum((y_pred-y_true)**2)
    mse = se / N
    return mse
