def robust_scaling(values: list) -> list:
    """
    Returns values centered by the median and scaled by the interquartile range.
    """
    # Write code here
    sorted_values = sorted(values)
    n = len(sorted_values)

    def median(sorted_list):
        m = len(sorted_list)
        if m % 2 == 1:
            return sorted_list[m // 2]
        else:
            return (sorted_list[m // 2 - 1] + sorted_list[m // 2]) / 2

    q2 = median(sorted_values)
    if n == 1:
        return [0.0]

    lower = sorted_values[:n // 2]
    upper = sorted_values[n // 2 + (1 if n % 2 != 0 else 0):]

    q1 = median(lower)
    q3 = median(upper)

    iqr = q3 - q1

    if iqr == 0:
        return [0.0 for _ in values]  # Avoid division by zero

    return [(x - q2) / iqr for x in values]