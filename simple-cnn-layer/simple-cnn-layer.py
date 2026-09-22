import numpy as np

def conv2d(x: list, W: list, b: list) -> np.ndarray:
    """
    Returns the convolved batch as a floating-point NumPy array.
    """
    # Write code here
    # x: (N, Cin, H, W)
    # W: (Cout, Cin, KH, KW)
    # b: (Cout,)

    N, Cin, H, W_in = x.shape
    Cout, _, KH, KW = W.shape

    H_out = H - KH + 1
    W_out = W_in - KW + 1

    y = np.zeros((N, Cout, H_out, W_out), dtype=float)

    for n in range(N):
        for c in range(Cout):
            for i in range(H_out):
                for j in range(W_out):

                    total = 0.0

                    for d in range(Cin):
                        for u in range(KH):
                            for v in range(KW):

                                total += x[n, d, i+u, j+v] * W[c, d, u, v]

                    y[n, c, i, j] = total +  b[c]

    return y