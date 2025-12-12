"""Early stopping callback.

Stops training when a monitored metric stops improving.
"""
from __future__ import annotations

from typing import Literal

from lr_geom.training.callbacks.base import Callback


class EarlyStopping(Callback):
    """Stop training when metric stops improving.

    Example:
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=0.001,
        )
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: Literal["min", "max"] = "min",
        restore_best_weights: bool = False,
    ) -> None:
        """Initialize early stopping.

        Args:
            monitor: Metric to monitor.
            patience: Number of epochs with no improvement to wait.
            min_delta: Minimum change to qualify as improvement.
            mode: "min" for metrics where lower is better, "max" for higher.
            restore_best_weights: Whether to restore best weights when stopping.
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        # State
        self.best_score: float | None = None
        self.best_weights: dict | None = None
        self.wait: int = 0

    def on_train_begin(self) -> None:
        """Reset state at start of training."""
        self.best_score = None
        self.best_weights = None
        self.wait = 0

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Check if training should stop."""
        current = metrics.get(self.monitor)
        if current is None:
            return

        # Check for improvement
        improved = False
        if self.best_score is None:
            improved = True
        elif self.mode == "min":
            improved = current < self.best_score - self.min_delta
        else:  # max
            improved = current > self.best_score + self.min_delta

        if improved:
            self.best_score = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in self.trainer.model.state_dict().items()
                }
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.trainer.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.trainer.model.load_state_dict(self.best_weights)
