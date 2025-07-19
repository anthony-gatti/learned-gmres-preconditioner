import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.sparse import kron, eye, diags, coo_matrix
from scipy.sparse.linalg import LinearOperator, gmres
from preconditioners import jacobi, ilu0
from model import DiagonalPreconditioner, SharedMLPBlockPreconditioner

N = 256
SCALE = 1
BLOCK_SIZE = 4
DIAG_MODEL_PATH = "diag_synth.pt"
BLOCK_MODEL_PATH = "mgs_block_model.pt"
PLOT_PATH = "plot_block.png"

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
    R = coo_matrix((vals[mask], (row_idx[mask], col_idx[mask])), shape=A.shape).tocsr()
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

def extract_block_features(A, block_size):
    n = A.shape[0]
    features = []
    A_csr = A.tocsr()
    for i in range(0, n, block_size):
        rows = A_csr[i:i+block_size]
        diag = rows.diagonal().mean()
        norm = np.sqrt(rows.multiply(rows).sum(axis=1)).mean()
        rowsum = np.abs(rows).sum(axis=1).mean()
        maxval = np.abs(rows).max(axis=1).mean()
        dom = np.mean(np.abs(rows.diagonal()) / (np.abs(rows).sum(axis=1).A.flatten() + 1e-8))
        features.append([diag, norm, rowsum, maxval, dom])
    return np.array(features, dtype=np.float32)

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
    residuals_diag = []
    diag_model = DiagonalPreconditioner(input_dim=5, hidden_dim=32)
    diag_model.load_state_dict(torch.load(DIAG_MODEL_PATH))
    diag_model.eval()
    diag_features = extract_diag_features(A)
    with torch.no_grad():
        s = diag_model(torch.tensor(diag_features, dtype=torch.float32)).numpy()
        diag_scaling = np.exp(s.flatten())
    M_diag = LinearOperator(A.shape, matvec=lambda x: x / diag_scaling, dtype=np.float64)
    _, info = gmres(A, b, restart=50, tol=adjusted_tol, maxiter=300, M=M_diag,
                    callback=lambda rk: residuals_diag.append(rk * bnrm2))

    # learned block
    print("Running learned block GMRES...")
    residuals_block = []
    block_model = SharedMLPBlockPreconditioner(input_dim=5, block_size=BLOCK_SIZE, hidden_dim=32)
    block_model.load_state_dict(torch.load(BLOCK_MODEL_PATH))
    block_model.eval()
    block_features = extract_block_features(A, BLOCK_SIZE)
    with torch.no_grad():
        ft = torch.tensor(block_features, dtype=torch.float32)
        L = block_model(ft)
        blocks = torch.bmm(L, L.transpose(1, 2)) + 1e-3 * torch.eye(BLOCK_SIZE).unsqueeze(0)
        blocks = blocks.numpy()

    def block_prec(x):
        if len(x) != len(blocks) * BLOCK_SIZE:
            return x
        y = np.zeros_like(x)
        for i, B in enumerate(blocks):
            s = i * BLOCK_SIZE
            e = s + BLOCK_SIZE
            try:
                y[s:e] = np.linalg.solve(B, x[s:e])
            except np.linalg.LinAlgError:
                y[s:e] = x[s:e]
        return y

    M_block = LinearOperator(A.shape, matvec=block_prec, dtype=np.float64)
    _, info = gmres(A, b, restart=50, tol=adjusted_tol, maxiter=300, M=M_block,
                    callback=lambda rk: residuals_block.append(rk * bnrm2))

    # print results
    print("\nFinal Residuals:")
    print(f"No Preconditioner   : {residuals[-1]:.2e} in {len(residuals)} iters")
    print(f"Jacobi              : {residuals_jacobi[-1]:.2e} in {len(residuals_jacobi)} iters")
    print(f"ILU(0)              : {residuals_ilu[-1]:.2e} in {len(residuals_ilu)} iters")
    print(f"Learned Diagonal    : {residuals_diag[-1]:.2e} in {len(residuals_diag)} iters")
    print(f"Learned Block       : {residuals_block[-1]:.2e} in {len(residuals_block)} iters")

    # plot results
    plt.figure(figsize=(10, 6))
    plt.semilogy(residuals, label="No Preconditioner", marker='o', markevery=10)
    plt.semilogy(residuals_jacobi, label="Jacobi", marker='x', markevery=10)
    plt.semilogy(residuals_ilu, label="ILU(0)", marker='s', markevery=10)
    plt.semilogy(residuals_diag, label="Learned Diagonal", marker='^', markevery=10)
    plt.semilogy(residuals_block, label="Learned Block", marker='D', markevery=10)
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