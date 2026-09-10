import math

def cosine_embedding_loss(x1: list, x2: list, label: int, margin: float) -> float:
    """
    Returns the cosine embedding loss as a float.
    """
    # Write code here
    dot = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0

    for a, b in zip(x1, x2):
        dot += a * b
        norm1_sq += a * a
        norm2_sq += b * b

    norm1 = math.sqrt(norm1_sq)
    norm2 = math.sqrt(norm2_sq)

    cosine = dot / (norm1 * norm2)

    if label == 1:
        loss = 1 - cosine
    else:
        loss = max(0, cosine - margin)

    return float(loss)