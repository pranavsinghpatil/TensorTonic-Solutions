import numpy as np
# probe
def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    single_point = np.array(points).ndim == 1
    points = np.atleast_2d(points)
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))
    transformed = (T @ points_h.T).T

    result = transformed[:, :3]

    if single_point:
        return result[0]

    return result
    