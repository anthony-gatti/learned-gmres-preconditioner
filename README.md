# Learned Preconditioning to Accelerate GMRES Convergence

This project explores the use of neural networks to learn effective preconditioners for accelerating GMRES convergence, specifically for non-normal sparse matrices. Traditional preconditioners like Jacobi and ILU(0) often struggle with non-normality, which is common in directional systems such as advection-diffusion problems. This work investigates whether learned diagonal and block preconditioners can offer a lightweight, adaptive alternative.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Implementations](#implementations)
  - [Diagonal Preconditioner (Phase 1)](#diagonal-preconditioner-phase-1)
  - [Block Preconditioner (Phase 2)](#block-preconditioner-phase-2)
- [Benchmarking](#benchmarking)
- [Results](#results)
- [How to Run](#how-to-run)
- [Future Work](#future-work)

---

## Overview

The project is divided into two phases:

1. **Phase 1:** A lightweight diagonal preconditioner is learned by training a neural network on row-wise features of synthetic non-normal matrices. These features include the diagonal value, row norm, max absolute value, and diagonal dominance.

2. **Phase 2:** A more powerful block-based preconditioner is learned, where the model predicts 4×4 blocks using a shared MLP. The blocks are regularized to ensure invertibility.

Both models are trained to minimize the GMRES residual norm using a log-scaled reward function.

---

## Project Files
- benchmark.py           : Benchmarking diagonal preconditioners
- benchmark_block.py     : Benchmarking block preconditioners
- train_p1.py          : Train diagonal preconditioner
- train_p2.py          : Train block preconditioner
- igs_gmres.py         : GMRES with reorthogonalization
- mgs_gmres.py         : Standard GMRES implementation
- model.py             : Neural network model definitions
- preconditioners.py   : Apply and evaluate learned preconditioners
- load_mtx.py          : Load matrix data
- load_vector.py       : Load RHS vector
- diag_synth.pt        : Trained diagonal model
- mgs_block_model.pt   : Trained block model (MGS)

---

## Implementations

### Diagonal Preconditioner (Phase 1)

- Trained MLP model maps row-wise features to log-scaled diagonal values.
- Lightweight and cheap to compute, requiring only simple statistics of each row.
- Trained via reinforcement learning using the GMRES residual norm as reward.

### Block Preconditioner (Phase 2)

- Divides the matrix into 4×4 blocks.
- Each block is passed through a shared MLP to predict a preconditioner matrix.
- Outputs are regularized (`M = B·Bᵗ + εI`) to ensure invertibility.
- Applied in a chunk-wise fashion to transform the system for GMRES.

---

## Benchmarking

- Run `python3 benchmark.py` for diagonal benchmarking.
- Run `python3 benchmark_block.py` for block benchmarking.
- Models are evaluated against:
  - No preconditioner
  - Jacobi
  - ILU(0)
- Synthetic Poisson matrices are used with increasing non-normality (via added sparse upper-triangular noise).
- GMRES solves until the residual norm drops below 1e-5 or max 300 iterations.

Plots summarizing performance are located in:
- `plot_diag.png` – Diagonal model results
- `plot_block.png` – Block model results

---

## How to Run

1. **Train the Models:**
```bash
python3 train_p1.py  # Diagonal
python3 train_p2.py  # Block
```

2. **Benchmark:**
```bash
python3 benchmark.py
python3 benchmark_block.py
```

3. **Plotting:**
```bash
python3 plot.py
```

## Results
- **Diagonal Model:**
    - Competitive with Jacobi on both mild and strong non-normal matrices.
	- Lightweight and parallelizable.
- **Block Model:**
	- Underperformed due to training time constraints.
	- Potential remains high with more compute and larger blocks.