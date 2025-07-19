import numpy as np
from scipy.sparse.linalg import LinearOperator

def igs_gmres(A, b, restart=100, maxiter=500, tol=1e-6, M=None, callback=None):
    """
    GMRES with Iterated Gram-Schmidt orthogonalization and right preconditioning.
    A: matrix or LinearOperator
    b: right-hand side vector
    M: preconditioner (as LinearOperator or matrix-like), applied as A(Mx)
    """
    n = A.shape[0]
    x = np.zeros(n)
    r = b - A @ x
    beta = np.linalg.norm(r)

    if beta == 0:
        return x, 0

    # validate M
    if M is None:
        Mfunc = lambda v: v
    elif hasattr(M, "__matmul__"):
        Mfunc = lambda v: M @ v
    else:
        Mfunc = M

    total_iters = 0
    for outer in range(maxiter // restart):
        # initialize Krylov and Hessenberg
        V = np.zeros((n, restart + 1), dtype=A.dtype)
        H = np.zeros((restart + 1, restart), dtype=A.dtype)

        r = b - A @ x
        beta = np.linalg.norm(r)
        if beta < tol:
            return x, 0
        V[:, 0] = r / beta
        g = np.zeros(restart + 1)
        g[0] = beta

        for j in range(restart):
            w = A @ Mfunc(V[:, j])

            # igs
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w -= H[i, j] * V[:, i]
            for i in range(j + 1):
                h_add = np.dot(V[:, i], w)
                H[i, j] += h_add
                w -= h_add * V[:, i]

            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] != 0:
                V[:, j + 1] = w / H[j + 1, j]
            else:
                y, *_ = np.linalg.lstsq(H[:j+2, :j+1], g[:j+2], rcond=None)
                x += Mfunc(V[:, :j+1] @ y)
                if callback:
                    callback(x)
                return x, 0

            # least squares
            y, *_ = np.linalg.lstsq(H[:j+2, :j+1], g[:j+2], rcond=None)
            x_new = x + Mfunc(V[:, :j+1] @ y)
            res = np.linalg.norm(b - A @ x_new)

            if callback:
                callback(x_new)

            total_iters += 1
            if res < tol:
                return x_new, 0

        x = x_new  # restart

    return x, total_iters