"""Base metric classes for training system.

Provides the Metric abstract base class and MetricCollection for managing metrics.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor


class Metric(ABC):
    """Abstract base class for metrics.

    Metrics accumulate values over batches and compute final values at epoch end.

    Example:
        metric = MyMetric()
        metric.reset()
        for batch, outputs in data:
            metric.update(batch, outputs)
        value = metric.compute()
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name.

        Used as the key in metric dictionaries.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset metric state.

        Called at the start of each epoch.
        """
        ...

    @abstractmethod
    def update(
        self,
        batch: dict[str, Any],
        outputs: dict[str, Tensor] | Tensor,
    ) -> None:
        """Update metric with batch results.

        Args:
            batch: Input batch dictionary.
            outputs: Model outputs.
        """
        ...

    @abstractmethod
    def compute(self) -> float:
        """Compute final metric value.

        Called at end of epoch after all updates.

        Returns:
            Final metric value as a float.
        """
        ...


class MetricCollection:
    """Container for managing multiple metrics.

    Provides convenient methods to reset, update, and compute all metrics.
    """

    def __init__(self, metrics: list[Metric] | None = None) -> None:
        """Initialize metric collection.

        Args:
            metrics: Optional list of metrics to track.
        """
        self.metrics: list[Metric] = metrics or []

    def append(self, metric: Metric) -> None:
        """Add a metric to the collection.

        Args:
            metric: Metric to add.
        """
        self.metrics.append(metric)

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics:
            metric.reset()

    def update(
        self,
        batch: dict[str, Any],
        outputs: dict[str, Tensor] | Tensor,
    ) -> None:
        """Update all metrics with batch results.

        Args:
            batch: Input batch dictionary.
            outputs: Model outputs.
        """
        for metric in self.metrics:
            metric.update(batch, outputs)

    def compute(self) -> dict[str, float]:
        """Compute all metric values.

        Returns:
            Dictionary mapping metric names to values.
        """
        return {metric.name: metric.compute() for metric in self.metrics}

    def __len__(self) -> int:
        """Return number of metrics."""
        return len(self.metrics)

    def __iter__(self):
        """Iterate over metrics."""
        return iter(self.metrics)


class AverageMeter:
    """Tracks running average of a value.

    Useful for tracking loss over batches.
    """

    def __init__(self) -> None:
        """Initialize meter."""
        self.reset()

    def reset(self) -> None:
        """Reset meter state."""
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        """Update with new value.

        Args:
            value: Value to add (pre-multiplied by n if batch average).
            n: Number of samples this value represents.
        """
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        """Return running average."""
        if self.count == 0:
            return 0.0
        return self.sum / self.count
