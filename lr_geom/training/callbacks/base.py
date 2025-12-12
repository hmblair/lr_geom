"""Base callback classes for training system.

Provides the Callback base class and CallbackList for managing multiple callbacks.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lr_geom.training.trainer import Trainer


class Callback:
    """Base class for training callbacks.

    Callbacks are called at various points during training:
    - on_train_begin/end: Start/end of entire training
    - on_epoch_begin/end: Start/end of each epoch
    - on_train_batch_begin/end: Start/end of each training batch
    - on_val_batch_begin/end: Start/end of each validation batch

    Subclass and override methods as needed.
    """

    def set_trainer(self, trainer: Trainer) -> None:
        """Set reference to the trainer.

        Called automatically when callback is added to trainer.

        Args:
            trainer: The Trainer instance.
        """
        self.trainer = trainer

    def on_train_begin(self) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int) -> None:
        """Called at the start of each epoch.

        Args:
            epoch: Current epoch number (0-indexed).
        """
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            metrics: Dictionary of computed metrics for this epoch.
        """
        pass

    def on_train_batch_begin(self, batch_idx: int) -> None:
        """Called at the start of each training batch.

        Args:
            batch_idx: Index of the current batch.
        """
        pass

    def on_train_batch_end(
        self,
        batch_idx: int,
        loss: float,
        loss_components: dict[str, float] | None = None,
    ) -> None:
        """Called at the end of each training batch.

        Args:
            batch_idx: Index of the current batch.
            loss: Total loss value for this batch.
            loss_components: Optional dict of individual loss components.
        """
        pass

    def on_val_batch_begin(self, batch_idx: int) -> None:
        """Called at the start of each validation batch.

        Args:
            batch_idx: Index of the current batch.
        """
        pass

    def on_val_batch_end(
        self,
        batch_idx: int,
        loss: float,
        loss_components: dict[str, float] | None = None,
    ) -> None:
        """Called at the end of each validation batch.

        Args:
            batch_idx: Index of the current batch.
            loss: Total loss value for this batch.
            loss_components: Optional dict of individual loss components.
        """
        pass


class CallbackList:
    """Container for managing multiple callbacks.

    Dispatches callback events to all registered callbacks.
    """

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        """Initialize callback list.

        Args:
            callbacks: Optional list of callbacks to register.
        """
        self.callbacks: list[Callback] = callbacks or []

    def append(self, callback: Callback) -> None:
        """Add a callback to the list.

        Args:
            callback: Callback to add.
        """
        self.callbacks.append(callback)

    def set_trainer(self, trainer: Trainer) -> None:
        """Set trainer reference on all callbacks.

        Args:
            trainer: The Trainer instance.
        """
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_train_begin(self) -> None:
        """Call on_train_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self) -> None:
        """Call on_train_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end()

    def on_epoch_begin(self, epoch: int) -> None:
        """Call on_epoch_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float]) -> None:
        """Call on_epoch_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics)

    def on_train_batch_begin(self, batch_idx: int) -> None:
        """Call on_train_batch_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_train_batch_begin(batch_idx)

    def on_train_batch_end(
        self,
        batch_idx: int,
        loss: float,
        loss_components: dict[str, float] | None = None,
    ) -> None:
        """Call on_train_batch_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_train_batch_end(batch_idx, loss, loss_components)

    def on_val_batch_begin(self, batch_idx: int) -> None:
        """Call on_val_batch_begin on all callbacks."""
        for callback in self.callbacks:
            callback.on_val_batch_begin(batch_idx)

    def on_val_batch_end(
        self,
        batch_idx: int,
        loss: float,
        loss_components: dict[str, float] | None = None,
    ) -> None:
        """Call on_val_batch_end on all callbacks."""
        for callback in self.callbacks:
            callback.on_val_batch_end(batch_idx, loss, loss_components)

    def __len__(self) -> int:
        """Return number of callbacks."""
        return len(self.callbacks)

    def __iter__(self):
        """Iterate over callbacks."""
        return iter(self.callbacks)
