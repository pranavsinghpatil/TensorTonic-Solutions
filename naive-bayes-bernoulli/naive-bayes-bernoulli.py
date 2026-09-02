import numpy as np

def naive_bayes_bernoulli(X_train: list, y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a NumPy array of log posteriors.
    """
    # Write code here
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    
    classes, class_counts = np.unique(y_train, return_counts=True)
    priors = np.log(class_counts / len(y_train))

    n_classes = len(classes)
    n_features = X_train.shape[1]
    theta = np.zeros((n_classes, n_features))

    for i, c in enumerate(classes):
        X_c = X_train[y_train == c]
        N_c = len(X_c)
        N_jc = X_c.sum(axis=0)
        theta[i, :] = (N_jc + 1) / (N_c + 2)

    log_theta = np.log(theta)
    log_1_minus_theta = np.log(1 - theta)

    present_contrib = X_test @ log_theta.T
    absent_contrib = (1 - X_test) @ log_1_minus_theta.T

    log_likelihood = present_contrib + absent_contrib

    log_likelihood += priors

    log_likelihood = np.round(log_likelihood, 4)

    return log_likelihood