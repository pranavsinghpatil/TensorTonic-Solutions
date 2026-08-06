import torch

def tanh_forward_backward(x, upstream_gradient):
    """
    Returns: tanh output and its upstream-scaled input gradient
    """
    y = torch.tanh(x)

    loc_grad =(1 - y**2)

    in_grad = loc_grad * upstream_gradient

    return y, in_grad

