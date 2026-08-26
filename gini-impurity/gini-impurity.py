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
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)

        probabilities = counts / len(y)

        return 1.0 - np.sum(probabilities ** 2)

    gini_left = gini(y_left)
    gini_right = gini(y_right)

    weighted_gini = (
        (N_left / N) * gini_left
        + (N_right / N) * gini_right
    )

    return float(weighted_gini)    