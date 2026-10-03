def lag_features(series: list, lags: list) -> list:
    """
    Returns the lag feature matrix.
    """
    # Write code here
    max_lag = max(lags)
    n = len(series)
    
    return [
        [series[t - lag] for lag in lags]
        for t in range(max_lag, n)
    ]