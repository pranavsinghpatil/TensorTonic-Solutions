import numpy as np

def auc(fpr: list, tpr: list) -> float:
    """
    Returns the area as a float.
    """
    # Write code here
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    widths = np.diff(fpr)
    heights = 0.5 * (tpr[:-1] + tpr[1:])
    auc = np.sum(widths * heights)
    return float(auc)