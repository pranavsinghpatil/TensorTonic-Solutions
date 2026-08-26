import numpy as np

def information_gain(y: list, split_mask: list) -> float:
    """
    Returns the information gain as a float.
    """
    # Write code here
    y = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)
    y_left = y[split_mask]
    y_right = y[~split_mask]

    N = len(y)
    N_left = len(y_left)
    N_right = len(y_right)

    if N_left == 0 or N_right == 0:
        return 0.0

    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)

        probabilities = counts / len(labels)
        return -np.sum(probabilities * np.log2(probabilities))

    parent_entropy = entropy(y)

    left_entropy = entropy(y_left)
    right_entropy = entropy(y_right)

    weighted_child_entropy = (
        (N_left / N) * left_entropy
        + (N_right / N) * right_entropy
    )

    information_gain = parent_entropy - weighted_child_entropy

    return float(information_gain)