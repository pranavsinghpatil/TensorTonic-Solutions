import numpy as np

def gini_impurity(y_left: list, y_right: list) -> float:
    """
    Returns the impurity as a float.
    """
    # Write code here
    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)

    # Total samples
    N_left = len(y_left)
    N_right = len(y_right)
    N = N_left + N_right
    if N == 0:
        return 0.0

    def gini(y):
        # Empty child
        if len(y) == 0:
            return 0.0

        # Count each class
        _, counts = np.unique(y, return_counts=True)

        # Class probabilities
        probabilities = counts / len(y)

        # Gini impurity
        return 1.0 - np.sum(probabilities ** 2)

    # Compute impurity of each child
    gini_left = gini(y_left)
    gini_right = gini(y_right)

    # Weighted impurity
    weighted_gini = (
        (N_left / N) * gini_left
        + (N_right / N) * gini_right
    )

    return float(weighted_gini)    