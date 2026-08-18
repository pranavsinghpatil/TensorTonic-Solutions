import numpy as np

def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    logkf = np.sum(np.log(np.arange(1, k + 1)))
    log_pmf = -lam + k* np.log(lam) - logkf 
    pmf = np.exp(log_pmf)
    
    cdf = 0
    for i in range(k + 1):
        log_pmf_i = -lam + i * np.log(lam) - np.sum(np.log(np.arange(1, i + 1)))
        pmf_i = np.exp(log_pmf_i)
        cdf += pmf_i

    return pmf,cdf