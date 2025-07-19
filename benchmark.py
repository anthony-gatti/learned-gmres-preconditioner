import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.sparse import kron, eye, diags, coo_matrix
from scipy.sparse.linalg import LinearOperator, gmres
from igs_gmres import igs_gmres
from preconditioners import jacobi, ilu0
from model import DiagonalPreconditioner

N = 256
SCALE = 1
BLOCK_SIZE = 4
MODEL_PATH = "diag_synth.pt"
PLOT_PATH = "plot_diag.png"

# generate matrix
def generate_poisson_2d(n):
    e = np.ones(n)
    T = diags([e, -4*e, e], [-1, 0, 1], shape=(n, n))
    I = eye(n)
    return (kron(I, T) + kron(diags([e], [0], shape=(n, n)), I)).tocsr()

def make_non_normal(A, scale=1e-2, density=1e-4):
    n = A.shape[0]
    nnz = int(n * n * density)
    rng = np.random.default_rng()
    row_idx = rng.integers(low=0, high=n, size=nnz)
    col_idx = rng.integers(low=0, high=n, size=nnz)
    vals = rng.uniform(low=0.01, high=1.0, size=nnz)
    mask = row_idx <= col_idx
    row_idx = row_idx[mask]
    col_idx = col_idx[mask]
    vals = vals[mask]
    R = coo_matrix((vals, (row_idx, col_idx)), shape=A.shape).tocsr()
    return A + scale * R

# get features from mtx
def extract_diag_features(A):
    diag = A.diagonal().reshape(-1, 1)
    row_data = A.tocsr()
    row_norms = np.sqrt(row_data.multiply(row_data).sum(axis=1)).A
    row_sums = np.abs(row_data).sum(axis=1).A
    row_max = np.abs(row_data).max(axis=1).toarray()
    dominance = np.abs(diag) / (row_sums + 1e-8)
    return np.hstack([diag, row_norms, row_sums, row_max, dominance])

# begin benchmark
def run_benchmark():
    print("Generating synthetic test matrix...")
    A = generate_poisson_2d(N)
    A = make_non_normal(A, SCALE)
    A = A.tocsr()
    x_true = np.ones(A.shape[0])
    b = A @ x_true

    bnrm2 = np.linalg.norm(b)
    adjusted_tol = 1e-5 / bnrm2

    # no pc
    print("Running baseline GMRES...")
    residuals = []
    _, info = gmres(A, b, restart=50, tol=adjusted_tol, maxiter=300,
                    callback=lambda rk: residuals.append(rk * bnrm2))

    # jacobi
    print("Running Jacobi preconditioned GMRES...")
    residuals_jacobi = []
    M_jacobi = jacobi(A)
    _, info = gmres(A, b, restart=50, tol=adjusted_tol, maxiter=300, M=M_jacobi,
                    callback=lambda rk: residuals_jacobi.append(rk * bnrm2))

    # ilu(0)
    print("Running ILU(0) preconditioned GMRES...")
    residuals_ilu = []
    M_ilu = ilu0(A)
    _, info = gmres(A, b, restart=50, tol=adjusted_tol, maxiter=300, M=M_ilu,
                    callback=lambda rk: residuals_ilu.append(rk * bnrm2))

    # learned diag
    print("Running learned diagonal GMRES...")
    residuals_learned = []
    model = DiagonalPreconditioner(input_dim=5, hidden_dim=32)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    features = extract_diag_features(A)

    with torch.no_grad():
        s = model(torch.tensor(features, dtype=torch.float32)).numpy()
        diag_scaling = np.exp(s.flatten())

    M_learned = LinearOperator(A.shape, matvec=lambda x: x / diag_scaling, dtype=np.float64)

    _, info = gmres(A, b, restart=50, tol=adjusted_tol, maxiter=300, M=M_learned,
                    callback=lambda rk: residuals_learned.append(rk * bnrm2))

    if info == 0:
        print("Learned GMRES converged!")
    else:
        print(f"Learned GMRES did not converge (info = {info}).")

    # print results
    print("\nFinal Residuals:")
    print(f"No Preconditioner   : {residuals[-1]:.2e} in {len(residuals)} iters")
    print(f"Jacobi              : {residuals_jacobi[-1]:.2e} in {len(residuals_jacobi)} iters")
    print(f"ILU(0)              : {residuals_ilu[-1]:.2e} in {len(residuals_ilu)} iters")
    print(f"Learned Diagonal    : {residuals_learned[-1]:.2e} in {len(residuals_learned)} iters")

    # plot results
    plt.figure(figsize=(8, 5))
    plt.semilogy(residuals, label="No Preconditioner", marker='o', markevery=10)
    plt.semilogy(residuals_jacobi, label="Jacobi", marker='x', markevery=10)
    plt.semilogy(residuals_ilu, label="ILU(0)", marker='s', markevery=10)
    plt.semilogy(residuals_learned, label="Learned Diagonal", marker='^', markevery=10)
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.title(f"GMRES Convergence on Synthetic Non-Normal Matrix (Scale = {SCALE})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.show()

if __name__ == "__main__":
    run_benchmark()