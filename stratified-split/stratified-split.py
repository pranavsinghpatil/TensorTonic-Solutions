import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)
    N = y.shape[0]
    classes = np.unique(y)
    train_indices = []
    test_indices = []

    for cls in classes:
        class_idx = np.where(y == cls)[0]
        
        if rng is not None:
            rng.shuffle(class_idx)
        else:
            np.random.shuffle(class_idx)

        n_test = min(round(len(class_idx) * test_size),len(class_idx) - 1)
                
        test_idx = class_idx[:n_test]
        train_idx = class_idx[n_test:]
        
        train_indices.append(train_idx)
        test_indices.append(test_idx)

    train_indices = np.sort(np.concatenate(train_indices))
    test_indices = np.sort(np.concatenate(test_indices))

    X_train = X[train_indices]
    X_test = X[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
    