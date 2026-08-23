import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """Return the cosine similarity of a and b."""
    # Write code here
    a = np.asarray(a)
    b = np.asarray(b)

    num = np.dot(a, b)
    if num == 0: return 0.0

    deno = (np.linalg.norm(a) * np.linalg.norm(b))

    dis = num / deno

    return float(dis)