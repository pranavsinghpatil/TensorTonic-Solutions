import numpy as np

def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    """
    Returns a dictionary containing accuracy, precision, recall, and f1 rounded to six decimals.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))

    TP = np.zeros(len(classes), dtype=int)
    FP = np.zeros(len(classes), dtype=int)
    FN = np.zeros(len(classes), dtype=int)

    for i, c in enumerate(classes):
        TP[i] = np.sum((y_pred == c) & (y_true == c))
        FP[i] = np.sum((y_pred == c) & (y_true != c))
        FN[i] = np.sum((y_true == c) & (y_pred != c))

    accuracy = np.sum(y_true == y_pred) / len(y_true)

    if average == "micro":
        total_TP = np.sum(TP)
        total_FP = np.sum(FP)
        total_FN = np.sum(FN)

        precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
        recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    elif average == "macro":
        precision_per_class = TP / (TP + FP)
        precision_per_class[np.isnan(precision_per_class)] = 0.0
        precision = np.mean(precision_per_class)

        recall_per_class = TP / (TP + FN)
        recall_per_class[np.isnan(recall_per_class)] = 0.0
        recall = np.mean(recall_per_class)

        f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class)
        f1_per_class[np.isnan(f1_per_class)] = 0.0
        f1 = np.mean(f1_per_class)

    elif average == "weighted":
        support = np.sum(y_true == classes[:, None], axis=1)

        precision_per_class = TP / (TP + FP)
        precision_per_class[np.isnan(precision_per_class)] = 0.0
        precision = np.sum(precision_per_class * support) / np.sum(support)

        recall_per_class = TP / (TP + FN)
        recall_per_class[np.isnan(recall_per_class)] = 0.0
        recall = np.sum(recall_per_class * support) / np.sum(support)

        f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class)
        f1_per_class[np.isnan(f1_per_class)] = 0.0
        f1 = np.sum(f1_per_class * support) / np.sum(support)

    elif average == "binary":
        idx = np.where(classes == pos_label)[0]
        if len(idx) == 0:
            precision, recall, f1 = 0.0, 0.0, 0.0
        else:
            idx = idx[0]
            precision = TP[idx] / (TP[idx] + FP[idx]) if (TP[idx] + FP[idx]) > 0 else 0.0
            recall = TP[idx] / (TP[idx] + FN[idx]) if (TP[idx] + FN[idx]) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Round to six decimals
    return {
        "accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6)
    }
