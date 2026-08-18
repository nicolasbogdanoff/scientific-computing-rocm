"""Backend detection that never claims acceleration without evidence."""

from __future__ import annotations

import importlib.util


def backend_report() -> dict[str, object]:
    """Return a serializable report of the locally discoverable backend."""
    if importlib.util.find_spec("torch") is None:
        return {"torch_installed": False, "backend": "cpu", "accelerator": None}
    import torch

    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    cuda_available = bool(torch.cuda.is_available())
    return {
        "torch_installed": True,
        "backend": "rocm" if hip_version else ("cuda" if cuda_available else "cpu"),
        "accelerator": hip_version or ("cuda" if cuda_available else None),
        "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }


def choose_device(prefer_accelerator: bool = True) -> str:
    """Choose ``cuda`` only when the installed runtime reports it usable."""
    if not prefer_accelerator or importlib.util.find_spec("torch") is None:
        return "cpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
