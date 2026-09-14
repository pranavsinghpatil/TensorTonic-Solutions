import math

def selu(x: list) -> list:
    """
    Returns SELU values rounded to four decimal places.
    """
    # Write code here
    alpha = 1.67326
    lambda_ = 1.05070

    r = []
    for i in x:
        if i > 0:
           r.append(lambda_ * i)
        else :
            elu = lambda_ * alpha * (math.exp(i) - 1)
            r.append(elu)
    return r
            