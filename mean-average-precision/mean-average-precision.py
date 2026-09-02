import numpy as np

def mean_average_precision(y_true_list: list, y_score_list: list, k: int | None = None) -> dict:
    """
    Returns a dictionary with map_value and ap_per_query.
    """
    # Write code here
    ap_per_query = []
    map_value = 0.0

    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.array(y_true)
        y_score = np.array(y_score)

        # Step 1: Rank items by descending score
        ranked_indices = np.argsort(-y_score, kind="stable")
        ranked_relevance = y_true[ranked_indices]

        # Step 2: Compute cumulative relevance and precision
        cumulative_relevance = np.cumsum(ranked_relevance)
        precision_at_rank = cumulative_relevance / np.arange(1, len(ranked_relevance) + 1)

        # Step 3: Mask precision by relevance
        masked_precision = precision_at_rank * ranked_relevance

        # Step 4: Compute AP for the query
        total_relevant = np.sum(y_true)
        if total_relevant == 0:
            ap = 0.0
        else:
            if k is not None:
                masked_precision = masked_precision[:k]
            ap = np.sum(masked_precision) / total_relevant

        # Round to 6 decimal places and convert to float
        ap_rounded = float(f"{ap:.6f}")
        ap_per_query.append(ap_rounded)

    # Step 5: Compute mAP
    map_value = float(f"{np.mean(ap_per_query):.6f}")

    return {"map_value": map_value, "ap_per_query": ap_per_query}