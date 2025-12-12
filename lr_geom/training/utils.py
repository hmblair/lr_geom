"""Training utilities for lr_geom.

Provides common utilities for training:
- Seeding for reproducibility
- Device selection
"""
from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)

    Also configures PyTorch for deterministic operations where possible.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Enable deterministic algorithms where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(
    preference: Literal["auto", "cpu", "cuda", "mps"] = "auto",
) -> torch.device:
    """Get the appropriate torch device.

    Args:
        preference: Device preference. Options:
            - "auto": Automatically select best available (CUDA > MPS > CPU)
            - "cpu": Force CPU
            - "cuda": Force CUDA (raises if unavailable)
            - "mps": Force MPS (raises if unavailable)

    Returns:
        torch.device for the selected device.

    Raises:
        RuntimeError: If requested device is not available.
    """
    if preference == "cpu":
        return torch.device("cpu")

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")

    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")

    # Auto-select best available
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move all tensors in a batch dictionary to the specified device.

    Args:
        batch: Dictionary mapping keys to tensors.
        device: Target device.

    Returns:
        New dictionary with tensors on the target device.
    """
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
