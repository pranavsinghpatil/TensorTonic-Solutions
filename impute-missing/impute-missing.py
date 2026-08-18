import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    X = np.asarray(X, dtype=float).copy()

    if strategy not in ('mean', 'median'):
        return None

    mask = np.isnan(X)
    mask_valid = np.logical_not(mask)

    if X.ndim == 1:
        valid_values = X[mask_valid]

        if valid_values.size == 0:
            fill_value = 0

        elif strategy == 'mean':
            fill_value = np.mean(valid_values)

        elif strategy == 'median':
            fill_value = np.median(valid_values)

        X[mask] = fill_value

    elif X.ndim == 2:

        for j in range(X.shape[1]):
            valid_values = X[:, j][mask_valid[:, j]]

            if valid_values.size == 0:
                fill_value = 0

            elif strategy == 'mean':
                fill_value = np.mean(valid_values)

            elif strategy == 'median':
                fill_value = np.median(valid_values)

            X[mask[:, j], j] = fill_value

    else:
        return None

    return X