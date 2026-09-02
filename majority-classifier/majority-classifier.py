import numpy as np

def majority_classifier(y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a one-dimensional NumPy array.
    """
    # Write code here
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)

    labels, first_indices, counts = np.unique(y_train, return_index=True, return_counts=True)

    max_count = np.max(counts)
    candidates = labels[counts == max_count]

    if len(candidates) > 1:
        candidate_first_indices = first_indices[counts == max_count]
        majority_label = candidates[np.argmin(candidate_first_indices)]
    else:
        majority_label = candidates[0]

    predictions = np.full(X_test.shape[0], majority_label, dtype=int)

    return predictions

    