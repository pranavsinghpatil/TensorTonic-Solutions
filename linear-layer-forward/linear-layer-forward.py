def linear_layer_forward(X: list, W: list, b: list) -> list:
    """
    Returns the affine transformation for every input row.
    """
    # Write code here
    result = []

    for i in range(len(X)):
        row = []

        for j in range(len(W[0])):
            total = 0

            for k in range(len(W)):
                total += X[i][k] * W[k][j]

            total += b[j]
            row.append(total)

        result.append(row)

    return result