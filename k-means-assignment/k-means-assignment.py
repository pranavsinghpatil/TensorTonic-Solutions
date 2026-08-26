def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    # Write code here
    assignments = []

    for point in points:
        best_index = 0
        best_distance = float("inf")

        for i, centroid in enumerate(centroids):

            distance = 0

            for p, c in zip(point, centroid):
                distance += (p - c) ** 2

            if distance < best_distance:
                best_distance = distance
                best_index = i

        assignments.append(best_index)

    return assignments