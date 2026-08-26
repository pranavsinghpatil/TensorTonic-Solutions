def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    # Write code here
    import numpy as np

def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Compute precision and recall at cutoff k.
    Returns [precision, recall] as floats.
    """
    if k == 0:
        return [0.0, 0.0]

    rel_set = set(relevant)
    recommended_at_k = recommended[:k]

    hits = sum(item in rel_set for item in recommended_at_k)

    precision = hits / k

    recall = hits / len(rel_set) if rel_set else 0.0

    return [precision, recall]