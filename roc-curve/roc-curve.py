import numpy as np

def roc_curve(y_true: list, y_score: list) -> dict:
    """
    Returns a dictionary with fpr, tpr, and thresholds.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    order = np.argsort(-y_score)

    sorted_scores = y_score[order]
    sorted_labels = y_true[order]

    # Total positives and negatives
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    # Start of ROC curve
    fpr = [0.0]
    tpr = [0.0]
    thresholds = [np.inf]

    tp = 0
    fp = 0

    i = 0

    while i < len(sorted_scores):

        current_score = sorted_scores[i]

        # Process every sample with the same score
        while i < len(sorted_scores) and sorted_scores[i] == current_score:

            if sorted_labels[i] == 1:
                tp += 1
            else:
                fp += 1

            i += 1

        # ROC point after processing this score group
        tpr.append(tp / P)
        fpr.append(fp / N)
        thresholds.append(current_score)

    return {
        "fpr": np.asarray(fpr),
        "tpr": np.asarray(tpr),
        "thresholds": np.asarray(thresholds)
    }