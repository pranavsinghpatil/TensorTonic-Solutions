def k_means_centroid_update(points: list, assignments: list, k: int) -> list:
    """
    Returns one updated centroid for each cluster.
    """
    # Write code here
    d = len(points[0])

    # Sum of coordinates for each cluster
    sums = [[0.0] * d for _ in range(k)]

    # Number of points in each cluster
    counts = [0] * k

    # Add every point to its assigned cluster
    for point, cluster in zip(points, assignments):

        for j in range(d):
            sums[cluster][j] += point[j]

        counts[cluster] += 1

    # Compute centroids
    centroids = []

    for cluster in range(k):

        # Empty cluster
        if counts[cluster] == 0:
            centroid = [0.0] * d

        else:
            centroid = []

            for j in range(d):
                mean = sums[cluster][j] / counts[cluster]
                centroid.append(mean)

        centroids.append(centroid)

    return centroids