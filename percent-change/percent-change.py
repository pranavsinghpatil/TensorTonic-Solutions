def percent_change(series: list) -> list:
    """
    Returns the fractional change between consecutive values.
    """
    # Write code here
    r = []
    for i in range(1,len(series)):
        if series[i-1] == 0:
            t = 0
        else:
            t = (series[i] - series[i-1]) / series[i-1]
        r.append(t)
    return r