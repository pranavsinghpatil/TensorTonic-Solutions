import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    # Write code here
    x = np.asarray(x)
    n = x.shape[0]
    x_bar = np.mean(x)

    s = np.sqrt(np.sum((x-x_bar)**2) / (n-1))
    se = s / np.sqrt(n)
    t = (x_bar -mu0) / se

    return t