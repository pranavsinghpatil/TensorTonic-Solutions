import torch

def neuron_backward(inputs, weights, bias, upstream_gradient):
    """
    Returns: output, input gradients, weight gradients, and bias gradient
    """
    # inputs = inputs.to(dtype)
    # weights = weights.to(dtype)
    # bias = bias.to(dtype)
    # upstream_gradient = upstream_gradient.to(dtype)
    
    preactivation = torch.sum(inputs * weights) + bias
    y = torch.tanh(preactivation)
    de = (1- (y**2)) * upstream_gradient

    # inputs = inputs.to(preactivation.dtype)
    dei = de * weights.to(preactivation.dtype)
    dew = de * inputs.to(preactivation.dtype)
    # dei = dei.to(dtype=torch.float32)
    # dew = dew.to(dtype=torch.float32)

    return y, dei, dew, de

