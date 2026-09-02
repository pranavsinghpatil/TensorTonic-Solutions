import numpy as np

def confusion_matrix_norm(y_true: list, y_pred: list, num_classes: int | None = None, normalize: str = "none") -> np.ndarray:
    """
    Returns the confusion matrix as a NumPy array.
    """
    # Write code here
    # Step 1: Determine the number of classes
    if num_classes is None:
        if len(y_true) == 0 or len(y_pred) == 0:
            num_classes = 0
        else:
            num_classes = max(max(y_true), max(y_pred)) + 1

    # Handle empty inputs
    if len(y_true) == 0 or len(y_pred) == 0:
        if normalize == "none":
            return np.zeros((num_classes, num_classes), dtype=int)
        else:
            return np.zeros((num_classes, num_classes), dtype=float)

    # Step 2: Build the confusion matrix
    flattened = np.array(y_true) * num_classes + np.array(y_pred)
    counts = np.bincount(flattened, minlength=num_classes**2)
    confusion = counts.reshape((num_classes, num_classes))

    # Step 3: Normalize the matrix
    if normalize == "none":
        return confusion.astype(int)
    else:
        confusion = confusion.astype(float)

        if normalize == "true":
            # Normalize rows to sum to 1
            row_sums = confusion.sum(axis=1, keepdims=True)
            np.divide(confusion, row_sums, out=confusion, where=row_sums != 0)
        elif normalize == "pred":
            # Normalize columns to sum to 1
            col_sums = confusion.sum(axis=0, keepdims=True)
            np.divide(confusion, col_sums, out=confusion, where=col_sums != 0)
        elif normalize == "all":
            # Normalize the entire matrix to sum to 1
            total = confusion.sum()
            np.divide(confusion, total, out=confusion, where=total != 0)

        return confusion