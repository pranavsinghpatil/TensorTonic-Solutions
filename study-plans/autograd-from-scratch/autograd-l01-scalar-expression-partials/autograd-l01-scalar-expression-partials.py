import numpy as np

def scalar_expression_partials(a, b, c, h):
    """
    Returns: the expression value and its three numerical partial derivatives
    """
    def de(a,b,c):
        return a*b + c

    d = de(a,b,c)
    da = de((a+h), b, c)
    db = de(a, (b+h), c)
    dc = de(a, b, (c+h))
    
    dap=(da-d)/h
    dbp=(db-d)/h
    dcp=(dc-d)/h

    return (d, dap, dbp, dcp)
