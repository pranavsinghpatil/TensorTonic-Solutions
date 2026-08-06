import numpy as np

def build_expression_graph(leaves, operations):
    """
    Returns: node records in creation order and the final node ID
    """
    lookup = {}
    nodes = []

    for leaf in leaves:
        node = {
            "id": leaf["id"],
            "data": np.float64(leaf["data"]),
            "grad": 0.0,
            "op": "",
            "parents": []
        }
        nodes.append(node)
        lookup[node["id"]] = node

    for op in operations:
        left_node = lookup[op["left"]]
        right_node = lookup[op["right"]]
        if op["op"] == "+":
            data = left_node["data"] + right_node["data"]
            node = {
                "id": op["id"],
                "data": data,
                "grad": 0.0,
                "op": "+",
                "parents": [left_node["id"], right_node["id"]]
            }
            nodes.append(node)
            lookup[node["id"]] = node
           
        else:
            data = left_node["data"] * right_node["data"]
            node = {
                "id": op["id"],
                "data": data,
                "grad": 0.0,
                "op": "*",
                "parents": [left_node["id"], right_node["id"]]
            }
            nodes.append(node)
            lookup[node["id"]] = node

    return (nodes, nodes[-1]["id"])
