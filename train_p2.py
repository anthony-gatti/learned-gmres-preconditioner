import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import trange
from scipy.sparse import random as sparse_random, eye, coo_matrix
from scipy.sparse.linalg import LinearOperator, gmres

# init model
class SharedMLPBlockPreconditioner(nn.Module):
    def __init__(self, input_dim=5, block_size=4, hidden_dim=32):
        super().__init__()
        self.block_size = block_size
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, block_size * block_size)
        )

    def forward(self, x):
        return self.mlp(x).view(-1, self.block_size, self.block_size)

# generate matrix
def generate_non_normal_matrix(n, density=0.01, scale=0.05):
    A = sparse_random(n, n, density=density, format='csr', dtype=np.float32)
    A = A + A.T
    R = sparse_random(n, n, density=1e-4, format='csr', dtype=np.float32)
    A = A + scale * R  # add non-normality
    A = A + 0.1 * eye(n, format='csr')
    return A

# extract features from mtx
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

def build_block_preconditioner(blocks, block_size, n):
    def apply_prec(x):
        if len(x) != n:
            return x
        y = np.zeros_like(x)
        for i, B in enumerate(blocks):
            s = i * block_size
            e = s + block_size
            try:
                y[s:e] = np.linalg.solve(B, x[s:e])
            except np.linalg.LinAlgError:
                y[s:e] = x[s:e]
        return y
    return LinearOperator((n, n), matvec=apply_prec, dtype=np.float32)

# training step
def train_once(model, optimizer, A, b, features, baseline):
    model.train()
    features_tensor = torch.tensor(features, dtype=torch.float32)
    L = model(features_tensor)
    dist = torch.distributions.Normal(L, 0.2)
    sampled = dist.rsample()
    log_prob = dist.log_prob(sampled).sum()

    blocks = torch.bmm(sampled, sampled.transpose(1, 2)) + 1e-3 * torch.eye(model.block_size).unsqueeze(0)
    blocks = blocks.detach().numpy()

    M = build_block_preconditioner(blocks, model.block_size, A.shape[0])
    residuals = []
    bnrm = np.linalg.norm(b)
    callback = lambda rk: residuals.append(rk * bnrm)

    x, info = gmres(A, b, restart=50, tol=1e-5, maxiter=300, M=M, callback=callback)

    if not residuals:
        return 0.0, 0.0

    r0 = np.linalg.norm(b)
    rk = residuals[-1]
    reward = np.log((r0 + 1e-8) / (rk + 1e-8))
    reward = np.clip(reward, -5.0, 5.0)

    loss = -log_prob * reward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return reward, rk

# training loop
def main():
    BLOCK_SIZE = 4
    INPUT_DIM = 5
    HIDDEN_DIM = 32
    TOTAL_EPOCHS = 500
    DECAY = 0.9

    model = SharedMLPBlockPreconditioner(INPUT_DIM, BLOCK_SIZE, HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    baseline = 0.0
    for epoch in trange(TOTAL_EPOCHS):
        # n = np.random.choice([64, 128])
        n = 64
        A = generate_non_normal_matrix(n * n, density=0.015, scale=0.05)
        x_true = np.ones(A.shape[0])
        b = A @ x_true
        features = extract_block_features(A, BLOCK_SIZE)

        reward, final_res = train_once(model, optimizer, A, b, features, baseline)
        baseline = DECAY * baseline + (1 - DECAY) * reward

        print(f"Epoch {epoch:03d} | n={n} | Reward: {reward:.4f} | Final Residual: {final_res:.2e}")

        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f"block_model_epoch{epoch+1}.pt")

if __name__ == "__main__":
    main()