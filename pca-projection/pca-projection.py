import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here
    X = np.asarray(X)
    X_bar = np.mean(X, axis=0)
    n = X.shape[0]
    xc = X - X_bar

    cov = (xc.T @ xc ) / (n-1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sorted_lambdas = eigenvalues[::-1]
    sorted_vectors = eigenvectors[:, ::-1]
    
    best_arrows = sorted_vectors[:, 0:k]
    final = xc @ np.asarray(best_arrows)

    return final