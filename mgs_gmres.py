import numpy as np

def mgs_gmres(A, b, restart=100, tol=1e-6, maxiter=None, M=None):
    n = len(b)
    if maxiter is None:
        maxiter = n

    x = np.zeros_like(b)
    r = b - A @ x if M is None else M(A @ x) - b
    beta = np.linalg.norm(r)
    if beta < tol:
        return x, 0

    V = np.zeros((n, restart + 1))
    H = np.zeros((restart + 1, restart))
    V[:, 0] = r / beta

    for iter_outer in range(0, maxiter, restart):
        g = np.zeros(restart + 1)
        g[0] = beta

        for j in range(restart):
            w = A @ V[:, j]
            if M is not None:
                w = M(w)

            # mgs
            for i in range(j + 1):
                H[i, j] = np.dot(V[:, i], w)
                w -= H[i, j] * V[:, i]
            H[j + 1, j] = np.linalg.norm(w)

            if H[j + 1, j] != 0 and j + 1 < restart:
                V[:, j + 1] = w / H[j + 1, j]

            # least squares
            y, _, _, _ = np.linalg.lstsq(H[:j+2, :j+1], g[:j+2], rcond=None)
            x_approx = x + V[:, :j+1] @ y
            res = np.linalg.norm(b - A @ x_approx)

            if res < tol:
                return x_approx, iter_outer + j + 1

        # restart
        x += V[:, :restart] @ y
        r = b - A @ x
        beta = np.linalg.norm(r)
        if beta < tol:
            return x, iter_outer + restart

        V[:, 0] = r / beta

    return x, maxiter