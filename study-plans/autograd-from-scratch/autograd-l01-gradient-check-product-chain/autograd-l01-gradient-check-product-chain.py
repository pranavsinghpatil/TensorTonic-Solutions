import numpy as np

def gradient_check_product_chain(a, b, c, f, h):
    """
    Returns: the loss, analytic gradients, numerical gradients, and maximum absolute disagreement
    """
    d = a * b
    e = d + c
    L = e * f
    analytic = [ b * f, a * f, f, e]

    def loss(a, b, c, f):
        return ((a * b) + c) * f

    grad_a = (loss(a + h, b, c, f)- loss(a, b, c, f)) / h
    grad_b = (loss(a, b + h, c, f)- loss(a, b, c, f)) / h
    grad_c = (loss(a, b, c + h, f)- loss(a, b, c, f)) / h
    grad_f = (loss(a, b, c, f + h)- loss(a, b, c, f)) / h

    numerical = [grad_a, grad_b, grad_c, grad_f]

    errors = [abs(a - n) for a, n in zip(analytic, numerical)]
    max_error = max(errors)

    return (L, analytic, numerical, max_error)
