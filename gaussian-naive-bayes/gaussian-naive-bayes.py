def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    # Write code here
    # Get unique classes
    classes = []
    for label in y_train:
        if label not in classes:
            classes.append(label)

    n_features = len(X_train[0])
    n_total = len(X_train)

    # Store statistics for each class
    class_stats = {}
    for c in classes:
        X_class = []
        for i in range(n_total):
            if y_train[i] == c:
                X_class.append(X_train[i])

        n_c = len(X_class)

        # Prior probability
        prior = n_c / n_total

        # Mean of each feature
        means = []
        for j in range(n_features):
            total = 0
            for row in X_class:
                total += row[j]

            means.append(total / n_c)
        # Population variance of each feature
        variances = []

        for j in range(n_features):
            total = 0
            for row in X_class:
                total += (row[j] - means[j]) ** 2
            variance = total / n_c
            # Prevent division by zero
            variance += 1e-9
            variances.append(variance)

        class_stats[c] = (prior, means, variances)

    # Predict
    predictions = []
    for x in X_test:
        best_class = None
        best_score = None
        for c in classes:
            prior, means, variances = class_stats[c]

            # log(prior)
            # Using ** approximation is avoided; log is implemented below.
            score = ln(prior)
            for j in range(n_features):
                var = variances[j]
                mean = means[j]

                score += (
                    -0.5 * ln(2 * PI * var)
                    - ((x[j] - mean) ** 2) / (2 * var)
                )

            if best_score is None or score > best_score:
                best_score = score
                best_class = c

        predictions.append(best_class)

    return predictions
    
# Constants and natural log approximation
PI = 3.141592653589793
def ln(x):
    """
    Natural logarithm approximation without importing math.
    """
    if x <= 0:
        return float("-inf")
    # Normalize x near 1
    k = 0
    while x > 2:
        x /= 2
        k += 1

    while x < 0.5:
        x *= 2
        k -= 1

    y = (x - 1) / (x + 1)

    term = y
    result = 0
    n = 1
    for _ in range(100):
        result += term / n
        term *= y * y
        n += 2
    LN2 = 0.6931471805599453
    return 2 * result + k * LN2