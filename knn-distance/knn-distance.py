import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    # Write code here
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
        
    # Pairwise differences
    diff = X_test[:, None, :] - X_train[None, :, :]

    # Euclidean distances
    distances = np.sqrt(np.sum(diff ** 2, axis=2))

    # Indices sorted by distance
    sorted_indices = np.argsort(distances, axis=1)

    # Number of real neighbors available
    n_neighbors = min(k, X_train.shape[0])

    # Take nearest neighbors
    neighbors = sorted_indices[:, :n_neighbors]

    # Pad with -1 if k > number of training samples
    if k > n_neighbors:
        padding = np.full(
            (X_test.shape[0], k - n_neighbors),
            -1,
            dtype=int
        )

        neighbors = np.concatenate(
            [neighbors, padding],
            axis=1
        )

    return neighbors.astype(int)