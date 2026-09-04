def binning(values: list, num_bins: int) -> list:
    """
    Returns the equal-width bin index of every value.
    """
    # Write code here
    minimum = min(values)
    maximum = max(values)
    result = []

    range = maximum - minimum
    if range == 0:
        for value in values:
            result.append(range)
        return result
    width = (maximum - minimum) / num_bins
        
    for value in values:
        idx = int((value - minimum) / width)
        idx = min(idx, num_bins - 1)
        result.append(idx)

    return result