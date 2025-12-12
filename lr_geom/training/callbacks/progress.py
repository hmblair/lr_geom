"""Progress bar callback.

Displays training progress using tqdm.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from lr_geom.training.callbacks.base import Callback

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class ProgressBar(Callback):
    """Display training progress with tqdm.

    Shows epoch progress and current metrics.

    Example:
        progress = ProgressBar(show_batch_metrics=True)
    """

    def __init__(
        self,
        show_batch_metrics: bool = True,
        leave_epoch_bar: bool = False,
    ) -> None:
        """Initialize progress bar.

        Args:
            show_batch_metrics: Whether to show metrics during batch iteration.
            leave_epoch_bar: Whether to leave batch progress bars after completion.
        """
        super().__init__()
        self.show_batch_metrics = show_batch_metrics
        self.leave_epoch_bar = leave_epoch_bar

        self._epoch_bar: tqdm | None = None
        self._batch_bar: tqdm | None = None
        self._current_phase: str = ""

    def on_train_begin(self) -> None:
        """Check tqdm availability."""
        if not TQDM_AVAILABLE:
            import warnings
            warnings.warn(
                "tqdm not installed. Progress bars will not be displayed. "
                "Install with: pip install tqdm"
            )

    def on_epoch_begin(self, epoch: int) -> None:
        """Start epoch progress tracking."""
        if not TQDM_AVAILABLE:
            return

        # Close any existing batch bar
        if self._batch_bar is not None:
            self._batch_bar.close()
            self._batch_bar = None

    def on_train_batch_begin(self, batch_idx: int) -> None:
        """Initialize batch progress bar if needed."""
        if not TQDM_AVAILABLE:
            return

        if self._current_phase != "train":
            self._current_phase = "train"
            if self._batch_bar is not None:
                self._batch_bar.close()
            self._batch_bar = tqdm(
                desc=f"Epoch {self.trainer.current_epoch} [Train]",
                leave=self.leave_epoch_bar,
                dynamic_ncols=True,
            )

    def on_train_batch_end(
        self,
        batch_idx: int,
        loss: float,
        loss_components: dict[str, float] | None = None,
    ) -> None:
        """Update batch progress."""
        if not TQDM_AVAILABLE or self._batch_bar is None:
            return

        self._batch_bar.update(1)

        if self.show_batch_metrics:
            postfix = {"loss": f"{loss:.4f}"}
            if loss_components:
                for name, value in loss_components.items():
                    if name != "loss":
                        postfix[name] = f"{value:.4f}"
            self._batch_bar.set_postfix(postfix)

    def on_val_batch_begin(self, batch_idx: int) -> None:
        """Switch to validation progress bar."""
        if not TQDM_AVAILABLE:
            return

        if self._current_phase != "val":
            self._current_phase = "val"
            if self._batch_bar is not None:
                self._batch_bar.close()
            self._batch_bar = tqdm(
                desc=f"Epoch {self.trainer.current_epoch} [Val]",
                leave=self.leave_epoch_bar,
                dynamic_ncols=True,
            )

    def on_val_batch_end(
        self,
        batch_idx: int,
        loss: float,
        loss_components: dict[str, float] | None = None,
    ) -> None:
        """Update validation progress."""
        if not TQDM_AVAILABLE or self._batch_bar is None:
            return

        self._batch_bar.update(1)

        if self.show_batch_metrics:
            postfix = {"loss": f"{loss:.4f}"}
            self._batch_bar.set_postfix(postfix)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Display epoch summary."""
        if not TQDM_AVAILABLE:
            return

        # Close batch bar
        if self._batch_bar is not None:
            self._batch_bar.close()
            self._batch_bar = None
        self._current_phase = ""

        # Print epoch summary
        summary_parts = [f"Epoch {epoch}"]
        for name in ["train_loss", "val_loss"]:
            if name in metrics:
                summary_parts.append(f"{name}: {metrics[name]:.4f}")

        # Add other metrics
        for name, value in metrics.items():
            if name not in ["train_loss", "val_loss"]:
                summary_parts.append(f"{name}: {value:.4f}")

        tqdm.write(" | ".join(summary_parts))

    def on_train_end(self) -> None:
        """Clean up progress bars."""
        if self._batch_bar is not None:
            self._batch_bar.close()
            self._batch_bar = None
