import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    # Write code here
    w = np.asarray(w)
    v = np.asarray(v)
    g = np.asarray(grad)

    v_new = (momentum * v) + (lr * g )#* (wl))
    w_new = w-v_new

    return (w_new, v_new) 
    

    
    