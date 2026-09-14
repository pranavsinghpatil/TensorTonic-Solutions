import math

def he_initialization(W: list, fan_in: int) -> list:
    """
    Returns the weights mapped to the He uniform range.
    """
    # Write code here
    l = math.sqrt(6 / fan_in)
    wd = []

    for i in range(len(W)):
        cl = []
        for j in range(len(W[0])):
            t = float(W[i][j] * 2*l - l)
            cl.append(t)
        wd.append(cl)
    return wd
