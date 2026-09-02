def expected_calibration_error(y_true: list, y_pred: list, n_bins: int) -> float:
    """
    Returns the calibration error as a float.
    """
    # Write code here
    # accuracy = mean(targets)
    bin_targets = [0] * n_bins
    bin_predictions = [0] * n_bins
    bin_counts = [0] * n_bins
    
    total = 0
    for y, p in zip(y_true, y_pred):

        bin_idx = min(int(p * n_bins), n_bins - 1)

        bin_targets[bin_idx] += y
        bin_predictions[bin_idx] += p
        bin_counts[bin_idx] += 1

    for m in range(n_bins):
        if bin_counts[m] > 0:
            acc = bin_targets[m] / bin_counts[m]
            conf = bin_predictions[m] / bin_counts[m]

            total += (bin_counts[m] / len(y_true)) * abs(acc - conf)

    return float(total)