import torch

def neuron_forward(inputs, weights, bias):
    """
    Returns: scalar preactivation and tanh output
    """
    preactivation = torch.sum(inputs * weights) + bias
    out = torch.tanh(preactivation)
    return preactivation, out
