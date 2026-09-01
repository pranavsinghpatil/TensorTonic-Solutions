def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    # Write code here
    TP = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    mismatches = float(sum(1 for true, pred in zip(y_true, y_pred) if true != pred))
    FN = mismatches
    FP = mismatches
    denominator = 2 * TP + FP + FN
    if denominator == 0:
        return 0.0  
    micro_f1 = (2 * TP) / denominator
    
    return float(round(micro_f1, 4))
