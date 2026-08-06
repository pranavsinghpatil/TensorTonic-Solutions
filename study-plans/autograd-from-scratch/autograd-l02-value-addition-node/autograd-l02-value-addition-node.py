import numpy as np

def value_addition_node(left, right, output_id):
    """
    Returns: an addition node that retains the two supplied leaf records as ordered parents
    """
    left_value = np.float64(left["data"])
    right_value = np.float64(right["data"])

    data = left_value + right_value
    node = {

    "id": output_id,
    "data": data,
    "grad": 0.0,
    "op": "+",
    "parents": [left, right]

}

    return node
