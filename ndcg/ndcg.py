import math

def ndcg(relevance_scores: list, k: int) -> float:
    """
    Returns NDCG as a float.
    """
    # Write code here
    ideal = sorted(relevance_scores, reverse=True)

    def dcg(scores, k):
        total = 0
        for i, r in enumerate(scores[:k], start=1):
            gain = (2 ** r) - 1
            discount = math.log2(i + 1)
            total += gain / discount
        return total

    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    else:
        return dcg(relevance_scores, k) / ideal_dcg