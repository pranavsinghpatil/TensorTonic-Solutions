import numpy as np

def rmsprop_step(
    w: list,
    g: list,
    s: list,
    lr: float = 0.001,
    beta: float = 0.9,
    eps: float = 1e-8,
) -> tuple[list, list]:
    """
    Perform one RMSProp update step.
    """
    # Write code here
    w = np.asarray(w)
    g = np.asarray(g)
    s = np.asarray(s)
    
    st = beta * s + (1 - beta) * (g ** 2)
    wt = w - lr * g / (np.sqrt(st) + eps)


    return (wt, st)