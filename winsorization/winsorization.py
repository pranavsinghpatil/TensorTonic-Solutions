def winsorize(values: list, lower_pct: float, upper_pct: float) -> list:
    """
    Returns values clipped to the interpolated percentile bounds.
    """
    # Write code here
    sorted_values = sorted(values)
    
    def percentile(sorted_values, pct):
        n = len(sorted_values)

        k = (n - 1) * pct / 100

        lower_index = int(k)
        fraction = k - lower_index        

        if lower_index == n - 1:
            return sorted_values[lower_index]
        else:
            upper_index = lower_index + 1
            return sorted_values[lower_index] + fraction * (
    sorted_values[upper_index] - sorted_values[lower_index]
)

    lower_bound = percentile(sorted_values, lower_pct)
    upper_bound = percentile(sorted_values, upper_pct)

    winsorized = [
        max(min(x, upper_bound), lower_bound) for x in values
    ]

    return winsorized