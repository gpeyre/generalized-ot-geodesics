# Generalized OT Geodesics

PyTorch implementation of Lagrangian geodesic approximation for generalized Wasserstein-type energies.

## Repository Layout

- `ot_geodesics/`: main Python library.
- `example/`: Jupyter notebook demonstrating optimization in 2D.
- `paper/`: short mathematical summary in LaTeX.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Usage

```python
import torch
from ot_geodesics.solver import GeneralizedOTGeodesicSolver

d, n, t_steps = 2, 64, 10
x0 = torch.randn(d, n)
x1 = torch.randn(d, n) + torch.tensor([[2.0], [1.5]])

solver = GeneralizedOTGeodesicSolver(
    gamma=25.0,
    phi_mode="constant",
    optimizer_mode="lbfgs",  # or "gd"
)
result = solver.optimize(x0, x1, t_steps=t_steps, n_steps=20, snapshot_every=5)
print(result.losses[-1])
```

## Energy Minimized

The solver optimizes a discrete path of point clouds
$$
X_0, X_1, \dots, X_{T-1}, \quad X_t \in \mathbb{R}^{d \times n},
$$
with fixed initial condition $X_0 = x_0$, by minimizing
$$
\mathcal{E}(X_{1:T-1}) =
W_2^2(X_{T-1}, x_1) +
\frac{\gamma}{n(T-1)}
\sum_{t=1}^{T-1}\sum_{i=1}^n
\phi(X_t, X_t(:,i))\,\|X_t(:,i)-X_{t-1}(:,i)\|_2^2,
$$
where $\phi$ is the chosen metric modulation (`constant` or `transformer`), and $W_2^2$ is the terminal quadratic OT loss (computed with POT).

## Example

Notebook demo:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gpeyre/generalized-ot-geodesics/blob/main/example/generalized_ot_2d_demo.ipynb)

## Notes

- The objective uses only the terminal OT term $W_2^2(X_{T-1}, X_1)$ plus the kinetic regularization term.
- The initial slice is imposed exactly ($X_0 = x0$): only slices $t=1,\dots,T-1$ are optimized.
- The OT endpoint term is computed with POT and used in a differentiable surrogate objective by freezing the transport plan during each closure call.
- Two built-in `phi` models are provided:
  - `constant` (classical OT-like kinetic term)
  - `transformer` with $\phi(X_t,\cdot)=\sum_{i,j}\exp(-\|x_i-x_j\|^2)$
