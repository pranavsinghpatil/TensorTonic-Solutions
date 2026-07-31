import numpy as np

def finite_difference_derivative(coefficients, x, h):
    """
    Returns: the polynomial value at x, the value at x plus h, and the forward-difference slope
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    
    def f(q):
        acc = 0
        for coeff in reversed(coefficients):
            acc = acc * q + coeff
        return acc  

    fx = f(x)
    fxh = f(x+h)
    
    r  = (fxh-fx) / h
    return (fx,fxh,r)
