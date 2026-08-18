import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    # Write code here
    indices = np.arange(N)
    if shuffle:
        if rng is not None:
            indices = rng.permutation(indices)
        else:
            np.random.shuffle(indices)

    folds = np.array_split(indices, k)
    results = []
    for i in range(k):
        val = folds[i]
        train = np.concatenate(folds[:i] + folds[i+1:])
        results.append((train, val))

    return results