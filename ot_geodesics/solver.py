"""Main API for generalized OT geodesic optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
import torch

from .helper import linear_time_interpolation, phi_values, wasserstein_sq_with_pot


PhiMode = Literal["constant", "transformer"]
OptimizerMode = Literal["lbfgs", "gd"]


@dataclass
class OptimizationResult:
    """Outputs produced by the optimizer."""

    path: torch.Tensor
    losses: List[float]
    snapshots: List[torch.Tensor]
    endpoint_plans: Dict[str, np.ndarray]


def generalized_ot_energy(
    path: torch.Tensor,
    x1: torch.Tensor,
    gamma: float,
    phi_mode: PhiMode = "constant",
    beta: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, np.ndarray]]:
    """Compute generalized OT energy for a discrete path.

    Args:
        path: Tensor (d, n, T), the unknown geodesic samples.
        x1: Target cloud (d, n).
        gamma: Weight for kinetic term.
        phi_mode: Either "constant" or "transformer".
        beta: Temperature for transformer phi.
    """
    if path.ndim != 3:
        raise ValueError("path must have shape (d, n, T).")
    if path.shape[0] != x1.shape[0]:
        raise ValueError("path and x1 must have same ambient dimension.")
    if path.shape[1] != x1.shape[1]:
        raise ValueError("path and x1 must have same number of points.")
    if path.shape[2] < 2:
        raise ValueError("path must include at least two time steps.")

    w1 = wasserstein_sq_with_pot(path[:, :, -1], x1)
    n_particles = path.shape[1]
    n_transitions = path.shape[2] - 1

    kinetic = path.new_tensor(0.0)
    for t in range(1, path.shape[2]):
        x_t = path[:, :, t]
        x_prev = path[:, :, t - 1]
        phi_t = phi_values(x_t, mode=phi_mode, beta=beta)
        velocity_sq = ((x_t - x_prev) ** 2).sum(dim=0)
        kinetic = kinetic + (phi_t * velocity_sq).sum()

    # POT with uniform weights a_i=b_j=1/n already returns a mass-weighted
    # transport cost (average-per-particle scaling), so no extra /n is applied.
    w1_norm = w1.loss
    kinetic_norm = kinetic / (n_particles * n_transitions)
    total = w1_norm + gamma * kinetic_norm
    plans = {"last_to_target": w1.plan}
    return total, plans


class GeneralizedOTGeodesicSolver:
    """Solver for generalized OT geodesics (L-BFGS or vanilla GD)."""

    def __init__(
        self,
        gamma: float = 10.0,
        phi_mode: PhiMode = "constant",
        beta: float = 1.0,
        optimizer_mode: OptimizerMode = "lbfgs",
        lbfgs_lr: float = 1.0,
        lbfgs_max_iter: int = 20,
        gd_lr: float = 1e-2,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.gamma = gamma
        self.phi_mode = phi_mode
        self.beta = beta
        self.optimizer_mode = optimizer_mode
        self.lbfgs_lr = lbfgs_lr
        self.lbfgs_max_iter = lbfgs_max_iter
        self.gd_lr = gd_lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

    def initialize_path(self, x0: torch.Tensor, x1: torch.Tensor, t_steps: int) -> torch.Tensor:
        """Create linear interpolation initialization."""
        return linear_time_interpolation(x0, x1, t_steps=t_steps)

    def optimize(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t_steps: int = 8,
        n_steps: int = 30,
        snapshot_every: int = 5,
        optimizer_mode: Optional[OptimizerMode] = None,
        initial_path: Optional[torch.Tensor] = None,
        tolerance: float = 0.0,
        max_iter: Optional[int] = None,
    ) -> OptimizationResult:
        """Run optimization (L-BFGS or vanilla GD) and collect history."""
        x0 = x0.to(device=self.device, dtype=self.dtype)
        x1 = x1.to(device=self.device, dtype=self.dtype)
        if initial_path is None:
            path = self.initialize_path(x0, x1, t_steps=t_steps).clone().detach()
        else:
            path = initial_path.to(device=self.device, dtype=self.dtype).clone().detach()
            if path.shape != (x0.shape[0], x0.shape[1], t_steps):
                raise ValueError(
                    f"initial_path must have shape {(x0.shape[0], x0.shape[1], t_steps)}, "
                    f"got {tuple(path.shape)}."
                )
        # Enforce path[:, :, 0] == x0 exactly by optimizing only tail time-slices.
        path_tail = path[:, :, 1:].clone().detach()
        path_tail.requires_grad_(True)
        mode = optimizer_mode or self.optimizer_mode
        n_outer = max_iter if max_iter is not None else n_steps

        losses: List[float] = []
        snapshots: List[torch.Tensor] = [torch.cat([x0.unsqueeze(2), path_tail], dim=2).detach().cpu().clone()]
        endpoint_plans: Dict[str, np.ndarray] = {}

        if mode == "lbfgs":
            optimizer = torch.optim.LBFGS(
                [path_tail],
                lr=self.lbfgs_lr,
                max_iter=self.lbfgs_max_iter,
                line_search_fn="strong_wolfe",
            )

            for step in range(n_outer):
                current_plans: Dict[str, np.ndarray] = {}

                def closure() -> torch.Tensor:
                    optimizer.zero_grad()
                    full_path = torch.cat([x0.unsqueeze(2), path_tail], dim=2)
                    loss, plans = generalized_ot_energy(
                        path=full_path,
                        x1=x1,
                        gamma=self.gamma,
                        phi_mode=self.phi_mode,
                        beta=self.beta,
                    )
                    loss.backward()
                    current_plans.clear()
                    current_plans.update(plans)
                    return loss

                loss = optimizer.step(closure)
                loss_value = float(loss.detach().cpu().item())
                losses.append(loss_value)
                endpoint_plans = dict(current_plans)

                if (step + 1) % snapshot_every == 0 or step == n_outer - 1:
                    full_path = torch.cat([x0.unsqueeze(2), path_tail], dim=2)
                    snapshots.append(full_path.detach().cpu().clone())
                if tolerance > 0.0 and len(losses) >= 2:
                    if abs(losses[-2] - losses[-1]) <= tolerance:
                        break
        elif mode == "gd":
            optimizer = torch.optim.SGD([path_tail], lr=self.gd_lr, momentum=0.0)
            for step in range(n_outer):
                optimizer.zero_grad()
                full_path = torch.cat([x0.unsqueeze(2), path_tail], dim=2)
                loss, plans = generalized_ot_energy(
                    path=full_path,
                    x1=x1,
                    gamma=self.gamma,
                    phi_mode=self.phi_mode,
                    beta=self.beta,
                )
                loss.backward()
                optimizer.step()

                loss_value = float(loss.detach().cpu().item())
                losses.append(loss_value)
                endpoint_plans = dict(plans)
                if (step + 1) % snapshot_every == 0 or step == n_outer - 1:
                    full_path = torch.cat([x0.unsqueeze(2), path_tail], dim=2)
                    snapshots.append(full_path.detach().cpu().clone())
                if tolerance > 0.0 and len(losses) >= 2:
                    if abs(losses[-2] - losses[-1]) <= tolerance:
                        break
        else:
            raise ValueError(f"Unknown optimizer mode: {mode}")

        return OptimizationResult(
            path=torch.cat([x0.unsqueeze(2), path_tail], dim=2).detach().cpu(),
            losses=losses,
            snapshots=snapshots,
            endpoint_plans=endpoint_plans,
        )
