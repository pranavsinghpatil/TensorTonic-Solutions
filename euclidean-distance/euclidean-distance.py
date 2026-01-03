import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # s = 0
    xn = np.array(x)
    yn = np.array(y)
    # for i in range(len(x)):
    s =np.sum(np.square(xn-yn))
    return np.sqrt(s)

    # Write code here
    # pass