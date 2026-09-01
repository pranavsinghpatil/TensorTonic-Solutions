def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """
    # Write code here
    stats = {}

    for category, target in zip(categories, targets):

        if category not in stats:
            stats[category] = {
                "sum": target,
                "count": 1
            }
        else:
            stats[category]["sum"] += target
            stats[category]["count"] += 1

    encoded = []

    for category in categories:
        mean = stats[category]["sum"] / stats[category]["count"]
        encoded.append(float(mean))

    return encoded
        