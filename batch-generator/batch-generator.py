import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)

    N = X.shape[0]

    indices = np.arange(N)

    if rng is not None: rng.shuffle(indices)
        
    else: np.random.shuffle(indices) 
    
    for start in range(0, N, batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]

        if len(batch_indices) < batch_size and drop_last:
            break

        yield X[batch_indices], y[batch_indices]
