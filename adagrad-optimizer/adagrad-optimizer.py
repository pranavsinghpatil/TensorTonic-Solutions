import numpy as np

def adagrad_step(w: list, g: list, G: list, lr: float = 0.01, eps: float = 1e-8) -> dict:
    """
    Returns a dictionary with new_w and new_G.
    """
    # Write code
    w = np.asarray(w)
    g = np.asarray(g)
    G = np.asarray(G)

    new_G = G + g ** 2
    wn = w - (lr / np.sqrt(new_G + eps)) * g

    return {"new_w": wn , "new_G": new_G}
    