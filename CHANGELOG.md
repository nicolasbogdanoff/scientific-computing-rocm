# Changelog

## 0.2.0 — 2026-09-29

- Added Python and optional PyTorch version metadata to backend reports.
- Added a reproducible 2D transient heat-equation PINN dossier with analytical validation and recorded ROCm metadata.
- Added configurable PINN seeds, architecture, learning rate, diffusion coefficient, evaluation resolution, and argument validation.
- Added CI syntax validation for the PINN experiment and removed a duplicated nested dossier path.

## 0.1.0 — 2026-08-18

- Added conservative CPU/CUDA/ROCm backend diagnostics.
- Added device selection that falls back to CPU when acceleration is unavailable.
- Added tests, an environment-report example, CI, and citation metadata.
