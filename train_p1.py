import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
from igs_gmres import igs_gmres
from scipy.sparse import diags, eye, kron, triu, random as sparse_random
from scipy.sparse.linalg import LinearOperator

# init model
class DiagonalPreconditioner(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

# create matrix
def generate_poisson_2d(n):
    e = np.ones(n)
    T = diags([e, -4 * e, e], [-1, 0, 1], shape=(n, n))
    I = eye(n)
    return (kron(I, T) + kron(T, I)).tocsr()

def make_non_normal(A, scale):
    R = triu(sparse_random(A.shape[0], A.shape[1], density=1e-4, format='csr'))
    return A + scale * R

# extract mtx features
def extract_diag_features(A):
    diag = A.diagonal().reshape(-1, 1)
    row_data = A.tocsr()
    row_norms = np.sqrt(row_data.multiply(row_data).sum(axis=1)).A
    row_sums = np.abs(row_data).sum(axis=1).A
    row_max = np.abs(row_data).max(axis=1).toarray()
    dominance = np.abs(diag) / (row_sums + 1e-8)
    return np.hstack([diag, row_norms, row_sums, row_max, dominance])

def run_gmres_residual(A, b, M=None):
    x, info = igs_gmres(A, b, restart=100, maxiter=500, tol=1e-6, M=M)
    r = b - A @ x
    return np.linalg.norm(r)

# training step
def train(model, optimizer, A, b, features, baseline_res):
    model.train()
    features_tensor = torch.tensor(features, dtype=torch.float32)
    diag_logits = model(features_tensor)
    dist = torch.distributions.Normal(diag_logits, 0.1)
    s_sampled = dist.rsample()
    log_prob = dist.log_prob(s_sampled).sum()

    base_diag = A.diagonal().astype(np.float32)
    diag_vals = torch.exp(s_sampled.detach()) * torch.tensor(base_diag)

    def preconditioner_fn(x):
        return x / diag_vals.numpy()

    M_op = LinearOperator(A.shape, matvec=preconditioner_fn, dtype=A.dtype)
    final_res = run_gmres_residual(A, b, M=M_op)
    reward = np.clip(np.log(baseline_res / (final_res + 1e-12)), -2.0, 2.0)

    loss = -log_prob * reward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return reward, final_res

# training loop
def main():
    model = DiagonalPreconditioner()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # check if model already exists
    if os.path.exists("learned_model.pt") and os.path.exists("optimizer.pt"):
        model.load_state_dict(torch.load("learned_model.pt"))
        optimizer.load_state_dict(torch.load("optimizer.pt"))
        print("Resumed from checkpoint.")

    total_epochs = 500
    decay = 0.9
    baseline = 0.0

    for epoch in trange(total_epochs):
        n = np.random.choice([64, 128, 256])  # randomly pick mtx size
        N = n * n 
        scale = 10 ** np.random.uniform(-2.5, -1.0)
        A = generate_poisson_2d(n)
        A = make_non_normal(A, scale)
        x_true = np.ones(N)
        b = A @ x_true
        features = extract_diag_features(A)
        baseline_res = run_gmres_residual(A, b)

        reward, final_res = train(model, optimizer, A, b, features, baseline_res)
        advantage = reward - baseline
        baseline = decay * baseline + (1 - decay) * reward

        print(f"Epoch {epoch:04d} | n={n} | scale={scale:.1e} | Reward: {reward:.4f} | Residual: {final_res:.2e}")

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), "learned_model.pt")
            torch.save(optimizer.state_dict(), "optimizer.pt")
            print("Checkpoint saved.")

    torch.save(model.state_dict(), "learned_model.pt")
    torch.save(optimizer.state_dict(), "optimizer.pt")
    print("Final model saved as learned_model.pt")

if __name__ == "__main__":
    main()