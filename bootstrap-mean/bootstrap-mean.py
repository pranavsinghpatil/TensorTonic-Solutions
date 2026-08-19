import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    boot_means = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        indices = rng.integers(0, N, size=N)
        sample = x[indices]
        mean = np.mean(sample)
        boot_means[b] = mean

    alpha = (1 - ci) / 2
    lower = np.quantile(boot_means, alpha)
    upper = np.quantile(boot_means, 1 - alpha)

    return boot_means, lower, upper
