import numpy as np

def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    # Write code here
    X = np.asarray(X)
    y = np.asarray(y)

    def gini(labels):
        classes, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return 1 - np.sum(probabilities ** 2)

    parent_gini = gini(y)
    best_gain = -np.inf
    best_feature = None
    best_threshold = None
        
    for f in range(X.shape[1]):
        values = np.unique(X[:, f])
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i+1]) / 2
                
            left_mask = X[:, f] <= threshold
            right_mask = X[:, f] > threshold
                
            y_left = y[left_mask]
            y_right = y[right_mask]
                
            left_gini = gini(y_left)
            right_gini = gini(y_right)

            weighted_gini = (
                len(y_left) / len(y) * left_gini
                + len(y_right) / len(y) * right_gini
            )

            gain = parent_gini - weighted_gini
                
            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_threshold = threshold

    return [best_feature, best_threshold]