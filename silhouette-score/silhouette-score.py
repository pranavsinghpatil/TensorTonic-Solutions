import numpy as np

def silhouette_score(X: list, labels: list[int]) -> float:
    """
    Returns the mean Silhouette Score as a Python float.
    """
    # Write code here
    X = np.array(X)
    labels = np.array(labels)

    diff = X[:, None, :] - X[None, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))

    a = np.zeros(len(X))
    b = np.zeros(len(X))

    for i in range(len(X)):
        same_cluster = (labels == labels[i]) & (np.arange(len(X)) != i)
        a[i] = np.mean(distances[i, same_cluster])

    for i in range(len(X)):
        other_clusters = labels != labels[i]
        if np.any(other_clusters):
            other_cluster_labels = labels[other_clusters]
            unique_other_clusters = np.unique(other_cluster_labels)

            avg_distances = []
            for c in unique_other_clusters:
                cluster_mask = other_cluster_labels == c
                avg_distances.append(np.mean(distances[i, other_clusters][cluster_mask]))

            b[i] = np.min(avg_distances)

    s = (b - a) / np.maximum(a, b)

    return float(np.mean(s))