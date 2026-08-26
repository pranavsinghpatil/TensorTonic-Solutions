def random_forest_vote(predictions: list) -> list:
    """
    Returns the majority-vote label for every sample.
    """
    # Write code here
    n_samples = len(predictions[0])
    result = []

    for i in range(n_samples):
        votes = {}

        for tree in predictions:
            label = tree[i]

            if label in votes:
                votes[label] += 1
            else:
                votes[label] = 1

        max_votes = max(votes.values())

        winners = []

        for label in votes:
            if votes[label] == max_votes:
                winners.append(label)

        result.append(min(winners))

    return result