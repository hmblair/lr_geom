"""Model checkpointing callback.

Saves model checkpoints during training based on monitored metrics.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from lr_geom.training.callbacks.base import Callback


class ModelCheckpoint(Callback):
    """Save model checkpoints based on monitored metric.

    Example:
        checkpoint = ModelCheckpoint(
            dirpath="checkpoints",
            filename="model-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
        )
    """

    def __init__(
        self,
        dirpath: str | Path,
        filename: str = "model-{epoch:02d}",
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        save_top_k: int = 1,
        save_last: bool = True,
    ) -> None:
        """Initialize checkpoint callback.

        Args:
            dirpath: Directory to save checkpoints.
            filename: Checkpoint filename template. Can use {epoch}, {step},
                and any metric name as placeholders.
            monitor: Metric to monitor for best model selection.
            mode: "min" to save when metric decreases, "max" when increases.
            save_top_k: Number of best checkpoints to keep. Set to -1 to keep all.
            save_last: Whether to always save the last checkpoint.
        """
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last

        # State
        self.best_score: float | None = None
        self.best_path: Path | None = None
        self.saved_checkpoints: list[tuple[float, Path]] = []

    def on_train_begin(self) -> None:
        """Create checkpoint directory."""
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Save checkpoint if metric improved."""
        # Get current metric value
        current = metrics.get(self.monitor)
        if current is None:
            return

        # Format filename
        format_dict = {"epoch": epoch, "step": self.trainer.global_step, **metrics}
        filename = self.filename.format(**format_dict)
        filepath = self.dirpath / f"{filename}.pt"

        # Check if this is a new best
        is_best = False
        if self.best_score is None:
            is_best = True
        elif self.mode == "min" and current < self.best_score:
            is_best = True
        elif self.mode == "max" and current > self.best_score:
            is_best = True

        # Save checkpoint
        if is_best or self.save_top_k == -1 or self.save_top_k > len(self.saved_checkpoints):
            self._save_checkpoint(filepath, epoch, metrics)
            self.saved_checkpoints.append((current, filepath))

            if is_best:
                self.best_score = current
                self.best_path = filepath

            # Remove old checkpoints if needed
            if self.save_top_k > 0 and len(self.saved_checkpoints) > self.save_top_k:
                self._cleanup_old_checkpoints()

        # Save last checkpoint
        if self.save_last:
            last_path = self.dirpath / "last.pt"
            self._save_checkpoint(last_path, epoch, metrics)

        # Save best model separately
        if is_best and self.best_path is not None:
            best_path = self.dirpath / "best_model.pt"
            self._save_checkpoint(best_path, epoch, metrics)

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Save a checkpoint file."""
        self.trainer.save_checkpoint(
            str(path),
            metrics=metrics,
            best_score=self.best_score,
            monitor=self.monitor,
        )

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to stay within save_top_k limit."""
        # Sort by score
        reverse = self.mode == "max"
        self.saved_checkpoints.sort(key=lambda x: x[0], reverse=reverse)

        # Remove worst checkpoints
        while len(self.saved_checkpoints) > self.save_top_k:
            _, path = self.saved_checkpoints.pop()
            if path != self.best_path and path.exists():
                path.unlink()
