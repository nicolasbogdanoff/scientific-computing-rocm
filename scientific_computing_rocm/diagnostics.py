"""Backend detection that never claims acceleration without evidence."""

from __future__ import annotations

import importlib.util
import platform


def backend_report() -> dict[str, object]:
    """Return a serializable report of the locally discoverable backend."""
    if importlib.util.find_spec("torch") is None:
        return {
            "python_version": platform.python_version(),
            "torch_installed": False,
            "backend": "cpu",
            "accelerator": None,
            "device_name": None,
        }
    import torch

    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    return {
        "python_version": platform.python_version(),
        "torch_installed": True,
        "torch_version": torch.__version__,
        "backend": "rocm" if hip_version else ("cuda" if cuda_available else "cpu"),
        "accelerator": hip_version or ("cuda" if cuda_available else None),
        "device_count": device_count,
        "device_name": torch.cuda.get_device_name(0) if device_count else None,
    }


def choose_device(prefer_accelerator: bool = True, require_accelerator: bool = False) -> str:
    """Choose a usable device, optionally failing instead of falling back to CPU."""
    if not prefer_accelerator or importlib.util.find_spec("torch") is None:
        if require_accelerator:
            raise RuntimeError("an accelerator was required but is not enabled")
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if require_accelerator:
        raise RuntimeError("an accelerator was required but torch reports none available")
    return "cpu"
