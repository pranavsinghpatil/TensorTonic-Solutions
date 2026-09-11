import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Write code here
    x = np.asarray(x)

    if rng is not None:
        random_values = rng.random(x.shape)
    else:
        random_values = np.random.random(x.shape)

    mask = random_values < (1-p)

    scale = 1 / (1 - p)

    dropout_pattern = np.where(mask, scale, 0.0)

    output = x * dropout_pattern

    return output, dropout_pattern