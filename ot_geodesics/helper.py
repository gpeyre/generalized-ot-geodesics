"""Helper functions for generalized OT geodesic optimization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Tuple
import numpy as np
import ot
import torch


PhiMode = Literal["constant", "transformer"]


@dataclass
class WassersteinPlan:
    """Container for an OT plan and its differentiable objective value."""

    loss: torch.Tensor
    plan: np.ndarray


def squared_euclidean_cost(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return pairwise squared Euclidean costs between columns of x and y.

    Args:
        x: Tensor with shape (d, n).
        y: Tensor with shape (d, m).
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Expected x and y to be 2D tensors of shape (d, n).")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same ambient dimension d.")

    x_col = x.T.unsqueeze(1)  # (n, 1, d)
    y_col = y.T.unsqueeze(0)  # (1, m, d)
    return ((x_col - y_col) ** 2).sum(dim=2)  # (n, m)


def wasserstein_sq_with_pot(x: torch.Tensor, y: torch.Tensor) -> WassersteinPlan:
    """Compute W2^2 with POT and return a differentiable surrogate loss.

    The OT plan is computed with POT from detached NumPy arrays.
    The loss is then computed in PyTorch as <pi, C(x, y)> to allow gradients
    w.r.t. x and y while keeping the plan fixed during the current closure.
    """
    n = x.shape[1]
    m = y.shape[1]
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)

    x_np = x.detach().cpu().T.numpy()
    y_np = y.detach().cpu().T.numpy()
    cost_np = ot.dist(x_np, y_np, metric="sqeuclidean")
    plan = ot.emd(a, b, cost_np)

    plan_t = torch.as_tensor(plan, dtype=x.dtype, device=x.device)
    cost_t = squared_euclidean_cost(x, y)
    loss = (plan_t * cost_t).sum()
    return WassersteinPlan(loss=loss, plan=plan)


def phi_values(x_t: torch.Tensor, mode: PhiMode = "constant", beta: float = 1.0) -> torch.Tensor:
    """Return phi(X_t, X_t(i)) values for i=1..n as a tensor shape (n,)."""
    n = x_t.shape[1]
    if mode == "constant":
        return torch.ones(n, dtype=x_t.dtype, device=x_t.device)
    if mode == "transformer":
        # phi is independent of i according to the problem statement.
        if beta <= 0:
            raise ValueError("beta must be > 0 for transformer phi.")
        cost = squared_euclidean_cost(x_t, x_t)
        scalar = torch.exp(-cost / (2.0 * beta * beta)).sum()
        return scalar * torch.ones(n, dtype=x_t.dtype, device=x_t.device)
    raise ValueError(f"Unknown phi mode: {mode}")


def linear_time_interpolation(x0: torch.Tensor, x1: torch.Tensor, t_steps: int) -> torch.Tensor:
    """Initialize X by linear interpolation between x0 and x1.

    Returns:
        Tensor with shape (d, n, t_steps), including both endpoints.
    """
    if t_steps < 2:
        raise ValueError("t_steps must be >= 2.")
    if x0.shape != x1.shape:
        raise ValueError("x0 and x1 must share the same shape (d, n).")

    alphas = torch.linspace(0.0, 1.0, t_steps, device=x0.device, dtype=x0.dtype)
    frames = [(1.0 - a) * x0 + a * x1 for a in alphas]
    return torch.stack(frames, dim=2)


def sample_points_from_shape_image(
    image_path: str | Path,
    n_samples: int,
    threshold: float = 0.5,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Sample 2D points from a black-foreground shape on white background."""
    import matplotlib.pyplot as plt

    image_path = Path(image_path)
    img = plt.imread(str(image_path))
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    img = img.astype(float)

    lo, hi = float(img.min()), float(img.max())
    thr = lo + threshold * (hi - lo)
    mask = img <= thr
    rows, cols = np.where(mask)
    if rows.size == 0:
        raise RuntimeError(f"No foreground pixels found in {image_path}")

    idx = np.random.choice(rows.size, size=n_samples, replace=(rows.size < n_samples))
    r = rows[idx].astype(float)
    c = cols[idx].astype(float)

    h, w = img.shape
    scale = float(max(h, w))
    x = (c - 0.5 * (w - 1)) / scale
    y = -(r - 0.5 * (h - 1)) / scale
    pts = np.stack([x, y], axis=0)
    return torch.tensor(pts, dtype=dtype)


def normalize_unit_global_variance(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Center cloud and scale it to unit global variance."""
    x = x - x.mean(dim=1, keepdim=True)
    global_std = torch.sqrt((x**2).mean())
    return x / (global_std + eps)


def make_separated_shape_clouds(
    resources_dir: str | Path,
    source_name: str,
    target_name: str,
    n_samples: int,
    offset: float = 2.8,
    threshold: float = 0.5,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build two normalized shape clouds and translate them apart on x-axis."""
    resources_dir = Path(resources_dir)
    x0 = sample_points_from_shape_image(
        resources_dir / source_name, n_samples=n_samples, threshold=threshold, dtype=dtype
    )
    x1 = sample_points_from_shape_image(
        resources_dir / target_name, n_samples=n_samples, threshold=threshold, dtype=dtype
    )
    x0 = normalize_unit_global_variance(x0)
    x1 = normalize_unit_global_variance(x1)
    x0 = x0 + torch.tensor([[-offset], [0.0]], dtype=dtype)
    x1 = x1 + torch.tensor([[offset], [0.0]], dtype=dtype)
    return x0, x1


def draw_trajectories(
    path: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    title: str = "",
    ax: Any | None = None,
    show: bool = True,
    show_legend: bool = True,
    line_width: float = 1.0,
    line_alpha: float = 0.35,
    intermediate_dot_size: float = 8.0,
    show_intermediate_dots: bool = False,
    endpoint_size: float = 18.0,
    n_max: int = 30,
    rng: np.random.Generator | None = None,
) -> Any:
    """Plot trajectories in 2D with optional intermediate-time dots.

    If the number of particles is larger than n_max, randomly subsample n_max
    trajectories for display.
    """
    import matplotlib.pyplot as plt

    if path.ndim != 3:
        raise ValueError("path must have shape (d, n, T).")
    if x0.ndim != 2 or x1.ndim != 2:
        raise ValueError("x0 and x1 must have shape (d, n).")
    if path.shape[0] < 2 or x0.shape[0] < 2 or x1.shape[0] < 2:
        raise ValueError("draw_trajectories expects at least 2D points.")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))

    path_np = path.detach().cpu().numpy()
    x0_np = x0.detach().cpu().numpy()
    x1_np = x1.detach().cpu().numpy()

    n = path_np.shape[1]
    if n_max > 0 and n > n_max:
        generator = rng if rng is not None else np.random.default_rng()
        idx = np.sort(generator.choice(n, size=n_max, replace=False))
    else:
        idx = np.arange(n)

    ax.scatter(x0_np[0, idx], x0_np[1, idx], s=endpoint_size, c="tab:blue", alpha=0.8, label="X0")
    ax.scatter(x1_np[0, idx], x1_np[1, idx], s=endpoint_size, c="tab:red", alpha=0.8, label="X1")

    for i in idx:
        traj_x = path_np[0, i, :]
        traj_y = path_np[1, i, :]
        ax.plot(traj_x, traj_y, color="0.45", alpha=line_alpha, lw=line_width)
        if show_intermediate_dots and path_np.shape[2] > 2:
            ax.scatter(
                traj_x[1:-1],
                traj_y[1:-1],
                s=intermediate_dot_size,
                color="0.35",
                alpha=0.55,
                zorder=3,
            )

    if title:
        ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    if show_legend:
        ax.legend()
    if show:
        plt.show()
    return ax


def draw_intermediate_pointclouds(
    path: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    num_clouds: int = 4,
    title_prefix: str = "Intermediate cloud",
    show: bool = True,
) -> Any:
    """Display intermediate point-cloud snapshots with blue-to-red colors.

    No subsampling is applied in this display.
    """
    import matplotlib.pyplot as plt

    if path.ndim != 3:
        raise ValueError("path must have shape (d, n, T).")
    if path.shape[0] < 2:
        raise ValueError("draw_intermediate_pointclouds expects at least 2D points.")
    if num_clouds < 1:
        raise ValueError("num_clouds must be >= 1.")

    path_np = path.detach().cpu().numpy()
    x0_np = x0.detach().cpu().numpy()
    x1_np = x1.detach().cpu().numpy()
    t_steps = path_np.shape[2]
    if t_steps < 3:
        raise ValueError("Need at least 3 time steps to show intermediate clouds.")

    t_idx = np.linspace(1, t_steps - 2, num_clouds, dtype=int)
    colors = plt.cm.coolwarm(np.linspace(0.15, 0.85, num_clouds))

    fig, axes = plt.subplots(1, num_clouds, figsize=(4.2 * num_clouds, 4), squeeze=False)
    for k, t in enumerate(t_idx):
        ax = axes[0, k]
        cloud = path_np[:, :, t]
        ax.scatter(cloud[0], cloud[1], s=12, color=colors[k], alpha=0.8)
        ax.scatter(x0_np[0], x0_np[1], s=8, color="tab:blue", alpha=0.25)
        ax.scatter(x1_np[0], x1_np[1], s=8, color="tab:red", alpha=0.25)
        ax.set_title(f"{title_prefix} t={t}")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    if show:
        plt.show()
    return axes


def compute_global_bbox(path: torch.Tensor, pad_ratio: float = 0.05) -> Tuple[float, float, float, float]:
    """Compute a global 2D bounding box over all particles and all times."""
    if path.ndim != 3 or path.shape[0] < 2:
        raise ValueError("path must have shape (d>=2, n, T).")
    path_np = path.detach().cpu().numpy()
    x_min = float(path_np[0].min())
    x_max = float(path_np[0].max())
    y_min = float(path_np[1].min())
    y_max = float(path_np[1].max())

    dx = x_max - x_min
    dy = y_max - y_min
    pad_x = pad_ratio * (dx if dx > 0 else 1.0)
    pad_y = pad_ratio * (dy if dy > 0 else 1.0)
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


def plot_interpolation_time_slice(
    path: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor,
    t: int,
    bbox: Tuple[float, float, float, float] | None = None,
    ax: Any | None = None,
    show: bool = True,
) -> Any:
    """Plot one interpolation slice with color evolving from blue (t=0) to red (t=T-1)."""
    import matplotlib.pyplot as plt

    if path.ndim != 3 or path.shape[0] < 2:
        raise ValueError("path must have shape (d>=2, n, T).")
    t_steps = path.shape[2]
    if not (0 <= t < t_steps):
        raise ValueError(f"t must be in [0, {t_steps - 1}].")

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6.5, 5.5))

    path_np = path.detach().cpu().numpy()
    x0_np = x0.detach().cpu().numpy()
    x1_np = x1.detach().cpu().numpy()

    alpha = 0.0 if t_steps <= 1 else t / (t_steps - 1)
    # Blue -> red interpolation.
    color_t = (alpha, 0.0, 1.0 - alpha)

    ax.scatter(x0_np[0], x0_np[1], s=14, c="tab:blue", alpha=0.20, label="X0 (ref)")
    ax.scatter(x1_np[0], x1_np[1], s=14, c="tab:red", alpha=0.20, label="X1 (ref)")
    ax.scatter(path_np[0, :, t], path_np[1, :, t], s=20, color=color_t, alpha=0.9, label=f"X(t={t})")

    if bbox is not None:
        x_min, x_max, y_min, y_max = bbox
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    ax.set_title("Interactive interpolation")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    if show:
        plt.show()
    return ax
