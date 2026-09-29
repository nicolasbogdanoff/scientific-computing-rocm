"""Compact, reproducible PINN for the 2D transient heat equation.

The implementation uses only PyTorch and the Python standard library. It is
designed for ROCm-enabled PyTorch, while remaining runnable on CPU for tests.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class Metrics:
    experiment: str
    device: str
    device_name: str
    torch_version: str
    hip_version: str | None
    alpha: float
    seed: int
    width: int
    hidden_layers: int
    evaluation_resolution: int
    steps: int
    interior_points: int
    elapsed_seconds: float
    rmse: float
    mae: float
    max_abs_error: float


class PINN(nn.Module):
    def __init__(self, width: int = 64, hidden_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.network(points)


def exact_solution(points: torch.Tensor, alpha: float) -> torch.Tensor:
    x, y, t = points[:, 0:1], points[:, 1:2], points[:, 2:3]
    return torch.exp(-2 * math.pi**2 * alpha * t) * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def set_seed(seed: int) -> None:
    """Set the random seeds used by CPU and accelerator sampling."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derivatives(model: nn.Module, points: torch.Tensor, alpha: float) -> torch.Tensor:
    points = points.detach().clone().requires_grad_(True)
    prediction = model(points)
    gradient = torch.autograd.grad(prediction, points, torch.ones_like(prediction), create_graph=True)[0]
    u_x, u_y, u_t = gradient[:, 0:1], gradient[:, 1:2], gradient[:, 2:3]
    u_xx = torch.autograd.grad(u_x, points, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, points, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    return u_t - alpha * (u_xx + u_yy)


def sample_interior(n: int, device: torch.device) -> torch.Tensor:
    return torch.rand(n, 3, device=device)


def sample_boundary(n: int, device: torch.device) -> torch.Tensor:
    points = torch.rand(n, 3, device=device)
    side = torch.randint(0, 4, (n,), device=device)
    points[side == 0, 0] = 0.0
    points[side == 1, 0] = 1.0
    points[side == 2, 1] = 0.0
    points[side == 3, 1] = 1.0
    return points


def train(
    model: PINN,
    device: torch.device,
    steps: int,
    n_interior: int,
    alpha: float,
    learning_rate: float,
) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    started = time.perf_counter()
    for _ in range(steps):
        interior = sample_interior(n_interior, device)
        boundary = sample_boundary(max(1024, n_interior // 4), device)
        initial = torch.rand(max(1024, n_interior // 4), 3, device=device)
        initial[:, 2] = 0.0

        residual = derivatives(model, interior, alpha)
        boundary_loss = model(boundary).square().mean()
        initial_loss = (model(initial) - exact_solution(initial, alpha)).square().mean()
        loss = residual.square().mean() + 10.0 * boundary_loss + 10.0 * initial_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - started


@torch.no_grad()
def evaluate(
    model: PINN,
    device: torch.device,
    alpha: float,
    resolution: int = 32,
) -> tuple[float, float, float]:
    axis = torch.linspace(0.0, 1.0, resolution, device=device)
    grid_x, grid_y, grid_t = torch.meshgrid(axis, axis, axis, indexing="ij")
    points = torch.stack((grid_x.flatten(), grid_y.flatten(), grid_t.flatten()), dim=1)
    error = (model(points) - exact_solution(points, alpha)).abs()
    return float(torch.sqrt((error.square()).mean())), float(error.mean()), float(error.max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--interior-points", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("pinn_2d_results.json"))
    args = parser.parse_args()

    if args.steps < 1 or args.interior_points < 1 or args.resolution < 2:
        parser.error("steps, interior-points y resolution deben ser positivos; resolution >= 2")
    if args.width < 1 or args.hidden_layers < 1 or args.learning_rate <= 0:
        parser.error("width, hidden-layers y learning-rate deben ser positivos")
    if args.alpha <= 0:
        parser.error("alpha debe ser positiva")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    model = PINN(width=args.width, hidden_layers=args.hidden_layers).to(device)
    elapsed = train(
        model,
        device,
        args.steps,
        args.interior_points,
        args.alpha,
        args.learning_rate,
    )
    rmse, mae, max_abs_error = evaluate(model, device, args.alpha, args.resolution)

    metrics = Metrics(
        experiment="2d_transient_heat_equation_pinn",
        device=str(device),
        device_name=device_name,
        torch_version=torch.__version__,
        hip_version=getattr(torch.version, "hip", None),
        alpha=args.alpha,
        seed=args.seed,
        width=args.width,
        hidden_layers=args.hidden_layers,
        evaluation_resolution=args.resolution,
        steps=args.steps,
        interior_points=args.interior_points,
        elapsed_seconds=elapsed,
        rmse=rmse,
        mae=mae,
        max_abs_error=max_abs_error,
    )
    args.output.write_text(json.dumps(asdict(metrics), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(metrics), indent=2))


if __name__ == "__main__":
    main()


