"""Hardware detection and device selection."""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict

import torch


class GPUUnavailableError(RuntimeError):
    """Raised when a GPU is required but not available."""


def runtime_report() -> Dict[str, Any]:
    """Collect versions and GPU information (safe to call without a GPU)."""
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpus": [],
    }
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpu = {
                "index": idx,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "bf16_supported": torch.cuda.is_bf16_supported() if idx == 0 else None,
            }
            try:
                free, total = torch.cuda.mem_get_info(idx)
                gpu["free_memory_gb"] = round(free / 1024**3, 2)
            except RuntimeError:  # pragma: no cover - driver specific
                gpu["free_memory_gb"] = None
            info["gpus"].append(gpu)
    return info


def format_runtime_report(info: Dict[str, Any]) -> str:
    lines = [
        f"Python        : {info['python']}",
        f"PyTorch       : {info['torch']} (CUDA build: {info['cuda_version']})",
        f"CUDA available: {info['cuda_available']}",
    ]
    if info["gpus"]:
        for gpu in info["gpus"]:
            lines.append(
                f"GPU {gpu['index']}         : {gpu['name']} | total {gpu['total_memory_gb']} GB"
                f" | free {gpu['free_memory_gb']} GB | CC {gpu['compute_capability']}"
            )
    else:
        lines.append("GPU           : none (CPU only - dry-run and preprocessing checks still work)")
    return "\n".join(lines)


def select_device(preference: str = "auto") -> torch.device:
    """Map a config string to a torch.device.

    ``cuda`` raises a clear error if no GPU is present instead of failing deep inside a model.
    """
    preference = preference.lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise GPUUnavailableError(
                "Config requests device 'cuda' but no GPU is visible. In Colab choose "
                "Runtime > Change runtime type > GPU, or set experiment.device=auto/cpu for a dry run."
            )
        return torch.device("cuda")
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device preference '{preference}'")


def require_gpu(task: str) -> None:
    """Stop a GPU-only step with an explanation."""
    if not torch.cuda.is_available():
        raise GPUUnavailableError(
            f"'{task}' needs a CUDA GPU, but none is available. Switch the Colab runtime to GPU "
            "or skip this step in dry-run mode."
        )
