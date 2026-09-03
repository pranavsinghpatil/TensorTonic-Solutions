def min_max_scaling(data: list) -> list:
    """
    Returns each data column scaled to the range from 0 through 1.
    """
    # Write code here
    rows = len(data)
    cols = len(data[0])

    result = [[0.0] * cols for _ in range(rows)]

    for j in range(cols):

        column = [row[j] for row in data]
        min_val = min(column)
        max_val = max(column)

        for i in range(rows):

            if min_val == max_val:
                result[i][j] = 0.0
            else:
                result[i][j] = (data[i][j] - min_val) / (max_val - min_val)

    return result