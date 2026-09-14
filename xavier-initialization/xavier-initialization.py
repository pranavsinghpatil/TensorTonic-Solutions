import math

def xavier_initialization(W: list, fan_in: int, fan_out: int) -> list:
    """
    Returns the weights mapped to the Xavier uniform range.
    """
    # Write code here
    l = math.sqrt(6 / (fan_in + fan_out))
    wd = []

    for i in range(len(W)):
        cl = []
        for j in range(len(W[0])):
            t = float(W[i][j] * 2*l - l)
            cl.append(t)
        
        wd.append(cl)

    return wd