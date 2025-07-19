from scipy.sparse.linalg import spilu, LinearOperator

def jacobi(A):
    """Return a LinearOperator that applies Jacobi (diagonal) preconditioning."""
    M_inv_diag = 1.0 / A.diagonal()

    def apply(x):
        return M_inv_diag * x

    return LinearOperator(A.shape, matvec=apply, dtype=A.dtype)

def ilu0(A):
    """Returns a LinearOperator using ILU(0) as the preconditioner."""
    A = A.tocsc()
    ilu = spilu(A)

    def apply(x):
        return ilu.solve(x)

    return LinearOperator(A.shape, matvec=apply, dtype=A.dtype)