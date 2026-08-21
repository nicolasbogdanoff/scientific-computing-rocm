# GPU-Accelerated 2D Heat Equation PINN on AMD ROCm

This experiment trains a compact physics-informed neural network (PINN) with PyTorch automatic differentiation to solve the two-dimensional transient heat equation on an AMD Radeon 890M through AMD University Program Learning Cloud.

## Problem

\[
u_t = \alpha (u_{xx} + u_{yy}), \qquad (x,y)\in[0,1]^2,\quad t\in[0,0.5]
\]

with `alpha = 0.10`, zero Dirichlet boundary conditions, and

\[
u(x,y,0) = \sin(\pi x)\sin(\pi y).
\]

The analytical reference is:

\[
u(x,y,t) = e^{-2\pi^2\alpha t}\sin(\pi x)\sin(\pi y).
\]

The loss combines the PDE residual, boundary conditions, and initial condition. The script is intentionally explicit so it can be adapted as an advanced numerical-methods or scientific-machine-learning laboratory.

## Reproduction

Use a ROCm-enabled PyTorch installation appropriate for the host. ROCm PyTorch exposes the accelerator through the portable `cuda` device API.

```bash
python pinn_2d_heat_equation.py --steps 3000 --interior-points 8000
```

The script reports the selected device and writes `pinn_2d_results.json`. On a CPU-only host it remains runnable, but the experiment is intended for an AMD GPU.

## Reported AMD run

| Metric | Refined result |
|---|---:|
| Device | AMD Radeon 890M Graphics |
| PyTorch | 2.9.1+rocm7.13.0 |
| HIP | 7.13.99004-3309c6114a |
| Initial RMSE | 0.0187208839 |
| Refined RMSE | **0.0034329589** |
| Refined MAE | 0.0026375460 |
| Refined max absolute error | 0.0205512028 |
| Total optimization time | 92.89 s |

The refinement reduced RMSE by approximately 81.7%. A separate residual benchmark measured 1,000, 4,000, and 8,000 interior points to illustrate higher-order automatic-differentiation scaling.

## Scope and limitations

This is a controlled synthetic problem with a known analytical solution. It demonstrates ROCm/PyTorch execution, automatic differentiation, validation, and reproducible reporting; it is not a production thermal model or a claim of multi-GPU performance. The original run used a fixed seed and saved metrics as JSON.

## Citation and provenance

Project author: Nicolás Mauricio Bogdanoff, Universidad Paraguayo Alemana (UPA).

The broader ROCm diagnostics repository is available at the repository root. AMD references used during validation:

- [AMD University Program](https://www.amd.com/en/corporate/university-program.html)
- [AMD PyTorch/ROCm Playbook](https://developer.amd.com/playbooks/pytorch-rocm-llms/)


