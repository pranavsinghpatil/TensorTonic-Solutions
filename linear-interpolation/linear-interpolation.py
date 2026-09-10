def linear_interpolation(values: list) -> list:
    """
    Returns a copy with every missing value interpolated.
    """
    # Write code here
    result = values.copy()
    
    i = 0
    while i < len(values):

        if values[i] is not None:
            i += 1
            continue

        left = i - 1

        right = i + 1
        while values[right] is None:
            right += 1

        j = i
        while values[j] is None:

            fraction = (j - left) / (right - left)

            result[j] = values[left] + fraction * (values[right] - values[left])

            j += 1

        i = right

    return result