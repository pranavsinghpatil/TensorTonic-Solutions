import numpy as np

def trace_reachable_graph(nodes, output_id):
    """
    Returns: reachable node IDs and parent-to-child edges in deterministic order
    """
    lookup = {}
    for node in nodes:
        lookup[node["id"]] = node
    edges = []
    visited = set()


    def dfs(node_id):
        if node_id in visited:
            return
        visited.add(node_id)
        for parent in lookup[node_id]["parents"]:
            dfs(parent)

    dfs(output_id)

    reachable_ids = []
    edges = []

    for node in nodes:
        if node["id"] in visited:
            reachable_ids.append(node["id"])
            for parent in node["parents"]:
                if parent in visited:
                    edges.append([parent, node["id"]])

    return (reachable_ids, edges)