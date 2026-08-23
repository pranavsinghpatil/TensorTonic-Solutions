import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Return the Euclidean distance between x and y.
    """
    # Write code here
    x = np.asarray(x)
    y = np.asarray(y)

    dis = np.sqrt(np.sum((x - y)**2))

    return float(dis)