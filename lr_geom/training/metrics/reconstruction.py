"""Reconstruction metrics.

Metrics for evaluating reconstruction quality.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lr_geom.training.metrics.base import Metric


class RMSD(Metric):
    """Root Mean Square Deviation metric.

    Computes RMSD between predicted and target coordinates.

    RMSD = sqrt(mean(||pred - target||^2))

    Expected keys in batch/outputs:
    - outputs["reconstruction"] or outputs: Predicted coordinates
    - batch["coords"] or batch["target"]: Target coordinates
    """

    def __init__(
        self,
        output_key: str = "reconstruction",
        target_key: str = "coords",
    ) -> None:
        """Initialize RMSD metric.

        Args:
            output_key: Key for predictions in outputs dict.
            target_key: Key for targets in batch dict.
        """
        self._output_key = output_key
        self._target_key = target_key
        self._sum_squared_error: float = 0.0
        self._count: int = 0

    @property
    def name(self) -> str:
        """Return metric name."""
        return "rmsd"

    def reset(self) -> None:
        """Reset metric state."""
        self._sum_squared_error = 0.0
        self._count = 0

    def update(
        self,
        batch: dict[str, Any],
        outputs: dict[str, Tensor] | Tensor,
    ) -> None:
        """Update with batch results.

        Args:
            batch: Input batch with target coordinates.
            outputs: Model outputs with predictions.
        """
        # Get predictions
        if isinstance(outputs, dict):
            pred = outputs.get(self._output_key)
            if pred is None:
                return
        else:
            pred = outputs

        # Get targets
        target = batch.get(self._target_key)
        if target is None:
            target = batch.get("target")
        if target is None:
            return

        # Compute squared errors
        # Handle different shapes: (B, N, 3) or (N, 3)
        diff = pred - target
        squared_dist = (diff ** 2).sum(dim=-1)  # Sum over xyz

        # Accumulate
        self._sum_squared_error += squared_dist.sum().item()
        self._count += squared_dist.numel()

    def compute(self) -> float:
        """Compute final RMSD.

        Returns:
            RMSD value.
        """
        if self._count == 0:
            return 0.0
        mse = self._sum_squared_error / self._count
        return mse ** 0.5


class MSE(Metric):
    """Mean Squared Error metric.

    Computes MSE between predicted and target tensors.
    """

    def __init__(
        self,
        output_key: str = "reconstruction",
        target_key: str = "coords",
    ) -> None:
        """Initialize MSE metric.

        Args:
            output_key: Key for predictions in outputs dict.
            target_key: Key for targets in batch dict.
        """
        self._output_key = output_key
        self._target_key = target_key
        self._sum_squared_error: float = 0.0
        self._count: int = 0

    @property
    def name(self) -> str:
        """Return metric name."""
        return "mse"

    def reset(self) -> None:
        """Reset metric state."""
        self._sum_squared_error = 0.0
        self._count = 0

    def update(
        self,
        batch: dict[str, Any],
        outputs: dict[str, Tensor] | Tensor,
    ) -> None:
        """Update with batch results."""
        if isinstance(outputs, dict):
            pred = outputs.get(self._output_key)
            if pred is None:
                return
        else:
            pred = outputs

        target = batch.get(self._target_key)
        if target is None:
            target = batch.get("target")
        if target is None:
            return

        diff = pred - target
        squared_error = (diff ** 2).sum().item()

        self._sum_squared_error += squared_error
        self._count += diff.numel()

    def compute(self) -> float:
        """Compute final MSE."""
        if self._count == 0:
            return 0.0
        return self._sum_squared_error / self._count
