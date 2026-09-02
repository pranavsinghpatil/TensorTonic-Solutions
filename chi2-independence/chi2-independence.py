import numpy as np

def chi2_independence(C: list) -> dict:
    """
    Returns a dictionary with chi2 and expected.
    """
    # Write code here
    C = np.asarray(C, dtype=float)

    row_totals = np.sum(C, axis=1)
    column_totals = np.sum(C, axis=0)
    total = np.sum(C)

    expected = np.outer(row_totals, column_totals) / total

    chi2 = np.sum((C - expected)**2 / expected)

    return {"chi2": float(chi2), "expected": expected}
    