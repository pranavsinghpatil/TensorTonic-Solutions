import torch

def neuron_gradient_check(inputs, weights, bias, h):
    """
    Returns: analytic and numerical parameter gradients with their maximum error
    """
    inputs = inputs.to(torch.float64)
    weights = weights.to(torch.float64)
    bias = bias.to(torch.float64)

    preactivation = torch.sum(inputs * weights) + bias
    y = torch.tanh(preactivation)

    analytic_bias = (1 - y**2)
    analytic_weight = inputs * analytic_bias

    def forward(inputs, weights, bias):
        a = torch.sum(inputs * weights) + bias
        return torch.tanh(a)

    numerical_weight = torch.zeros_like(weights, dtype=torch.float64)
    for i in range(len(weights)):
        perturbed = weights.clone()
        perturbed[i] += h
        numerical_weight[i] = (forward(inputs, perturbed, bias) - forward(inputs, weights, bias)) / h

    numerical_bias = (forward(inputs, weights, bias + h) - forward(inputs, weights, bias)) / h

    weight_diff = torch.abs(analytic_weight - numerical_weight)   
    bias_diff = torch.abs(analytic_bias - numerical_bias)         

    # Fix: Safely handle zero-element tensors
    if weight_diff.numel() == 0:
        max_error = bias_diff
    else:
        weight_error = weight_diff.max()
        max_error = torch.maximum(weight_error, bias_diff)

    return analytic_weight, numerical_weight, analytic_bias, numerical_bias, max_error