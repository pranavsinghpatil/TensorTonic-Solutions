import math

def elu(x: list, alpha: float = 1.0) -> list:
    """
    Returns ELU applied elementwise to the input values.
    """
    # Write code here
    r = []
    for i in x:
        if i > 0:
           r.append(i)
        else :
            elu = alpha * (math.exp(i) - 1)
            r.append(elu)
    return r
            