import torch
import torch.nn as nn

class DiagonalPreconditioner(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(DiagonalPreconditioner, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, features):
        return self.mlp(features).squeeze(-1)

class SharedMLPBlockPreconditioner(nn.Module):
    def __init__(self, input_dim, block_size, hidden_dim=64):
        super().__init__()
        self.block_size = block_size
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, block_size * block_size)  # output flattened block
        )

    def forward(self, features):
        out = self.mlp(features)
        return out.view(-1, self.block_size, self.block_size)