import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    i = np.arange(k+1)
    pmf = comb(n, k) * (p**k) * ((1-p)**(n-k))
    terms = comb(n, i) * (p**i) * ((1-p)**(n-i)) 
    cdf = np.sum(terms)

    return pmf, cdf

    