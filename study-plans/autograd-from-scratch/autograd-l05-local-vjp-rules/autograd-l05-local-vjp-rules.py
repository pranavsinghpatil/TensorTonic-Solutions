import numpy as np

def local_vjp(operation, inputs, output, upstream_gradient):
    """
    Returns: one gradient contribution per input in input order.
    """
    inputs = np.asarray(inputs, dtype=np.float64)
    g = np.float64(upstream_gradient)
    output = np.float64(output)
    
    if operation == "add":
        return [g,g]
    elif operation == "mul":
        return [g*inputs[1], g*inputs[0]]
    elif operation == "tanh":
        return [g * (1 - output**2)]
