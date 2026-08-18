def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)
    d = X.shape[1]
    I = np.eye(d)
    w = np.linalg.inv(X.T @ X + lam*I) @ (X.T @ y) 
    return w