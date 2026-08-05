import numpy as np

def gradient_descent_step(values, gradients, learning_rate):
    """
    Returns: updated values and the predicted first-order objective change
    """
    values = np.asarray(values, dtype=np.float64)
    gradients = np.asarray(gradients, dtype=np.float64)

    # for value, grad in zip(values,gradients):
    update = -learning_rate * gradients
    new_values = values + update

    predicted_change = gradients @ update

    # return (new_values, predicted_change)
    return (new_values.tolist(), float(predicted_change))