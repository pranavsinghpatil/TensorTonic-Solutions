import numpy as np

def topological_sort(nodes, output_id):
    """
    Returns: reachable node IDs in deterministic topological order
    using input order to break valid ordering ties.
    """

    lookup = {node["id"]: node for node in nodes}
    position = {node["id"]: i for i, node in enumerate(nodes)}

    # --------------------------------------------------
    # 1. Find reachable nodes
    # --------------------------------------------------
    reachable = set()

    def visit(node_id):
        if node_id in reachable:
            return
        reachable.add(node_id)
        for parent_id in lookup[node_id]["parents"]:
            visit(parent_id)

    visit(output_id)

    # --------------------------------------------------
    # 2. Build children + indegree
    # --------------------------------------------------
    children = {node_id: [] for node_id in reachable}
    indegree = {node_id: 0 for node_id in reachable}

    for node in nodes:
        node_id = node["id"]
        if node_id not in reachable:
            continue
        for parent_id in node["parents"]:
            children[parent_id].append(node_id)
            indegree[node_id] += 1

    # --------------------------------------------------
    # 3. Initially ready nodes
    # --------------------------------------------------
    ready = [nid for nid in reachable if indegree[nid] == 0]
    ready.sort(key=lambda nid: position[nid])

    # --------------------------------------------------
    # 4. Kahn's algorithm
    # --------------------------------------------------
    order = []
    while ready:
        current = ready.pop(0)
        order.append(current)  # <-- append ID, not full node
        for child_id in children[current]:
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                ready.append(child_id)
        ready.sort(key=lambda nid: position[nid])

    return order
