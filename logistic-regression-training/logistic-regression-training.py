import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    w = np.zeros(X.shape[1])
    b = 0.0
    
    for _ in range(steps):
        linear = X @ w + b
        prediction = _sigmoid(linear)
        error = prediction - y
        dw = X.T @ error / X.shape[0]
        db = np.mean(error)
        w = w - lr * dw
        b = b - lr * db
        
    # Write code here
    return (w,b)