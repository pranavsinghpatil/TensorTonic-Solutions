import math

def log_transform(values: list) -> list:
    """
    Returns the log1p-transformed values rounded to four decimals.
    """
    # Write code here
    y = []
    for val in values:
        r = math.log1p(val)
        y.append(round(r, 4))
    return y