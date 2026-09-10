def mean_rating_imputation(ratings_matrix: list, mode: str) -> list:
    """
    Returns a copy with missing ratings replaced by user or item means.
    """
    # Write code here
    result = [row.copy() for row in ratings_matrix]

    rows = len(ratings_matrix)
    cols = len(ratings_matrix[0])

    if mode == "user":

        means = []

        for i in range(rows):
            total = 0
            count = 0

            for j in range(cols):
                if ratings_matrix[i][j] != 0:
                    total += ratings_matrix[i][j] 
                    count += 1

            if count > 0:
                mean = total / count
            else:
                mean = 0.0

            means.append(mean)

        for i in range(rows):
            for j in range(cols):
                if ratings_matrix[i][j] == 0:
                    result[i][j] = means[i]

    else:  

        means = []

        for j in range(cols):
            total = 0
            count = 0

            for i in range(rows):
                if ratings_matrix[i][j] != 0:
                    total += ratings_matrix[i][j] 
                    count += 1

            if count > 0:
                mean = total / count
            else:
                mean = 0.0

            means.append(mean)

        for i in range(rows):
            for j in range(cols):
                if ratings_matrix[i][j] == 0:
                    result[i][j] = means[j]

    return result