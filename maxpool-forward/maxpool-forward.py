def maxpool_forward(X: list, pool_size: int, stride: int) -> list:
    """
    Returns the maximum value from every pooling window.
    """
    # Write code here
    H = len(X)
    W = len(X[0])

    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    out = []

    for i in range(H_out):
        row = []

        for j in range(W_out):

            row_start = i * stride
            col_start = j * stride


            values = []

            for r in range(row_start, row_start + pool_size):
                for c in range(col_start, col_start + pool_size):
                    values.append(X[r][c])

            result = max(values)
            row.append(result)

        out.append(row)

    return out