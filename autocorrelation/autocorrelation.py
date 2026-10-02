def autocorrelation(series: list, max_lag: int) -> list:
    """
    Returns normalized autocorrelation from lag zero through max_lag.
    """
    # Write code here
    n = len(series)
    mean = sum(series) / n

    diff = [x - mean for x in series]

    gamma_0 = sum(d * d for d in diff)
    if gamma_0 == 0:
        return [1.0] + [0.0] * max_lag

    autocorr = []
    for k in range(max_lag + 1):
        if k == 0:
            autocorr.append(1.0)
            continue

        cov_k = sum(diff[t] * diff[t + k] for t in range(n - k))
        autocorr.append(round(cov_k / gamma_0, 6))

    return autocorr