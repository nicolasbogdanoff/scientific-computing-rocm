"""Safe diagnostics for optional CPU, CUDA, and ROCm scientific backends."""

from .diagnostics import backend_report, choose_device

__all__ = ["backend_report", "choose_device"]
