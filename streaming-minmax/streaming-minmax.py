import numpy as np

def streaming_minmax(D: int, batches: list, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with normalized_batches, min, and max.
    """
    # Write code here
    running_min = np.full(D, np.inf)
    running_max = np.full(D, -np.inf)

    normalized_batches = []

    for batch in batches:

        batch_min = np.min(batch, axis=0)
        batch_max = np.max(batch, axis=0)

        running_min = np.minimum(running_min, batch_min)
        running_max = np.maximum(running_max, batch_max)

        denominator = np.maximum(running_max - running_min, eps)

        normalized = (batch - running_min) / denominator

        normalized_batches.append(normalized)

    return {
        "normalized_batches": normalized_batches,
        "min": running_min,
        "max": running_max
    }