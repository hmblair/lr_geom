"""Trainer class for training PyTorch models.

Provides a flexible training loop with support for:
- Multi-term loss functions
- Callbacks for customization
- Metrics tracking
- Learning rate scheduling
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lr_geom.training.callbacks.base import Callback, CallbackList
from lr_geom.training.metrics.base import AverageMeter, Metric, MetricCollection
from lr_geom.training.utils import get_device, move_batch_to_device


@dataclass
class TrainerConfig:
    """Configuration for Trainer.

    Attributes:
        loss_weights: Weights for combining multi-term losses.
            Keys should match loss component names returned by loss_fn.
        grad_clip: Maximum gradient norm for clipping. None to disable.
        device: Device preference ("auto", "cpu", "cuda", "mps").
    """

    loss_weights: dict[str, float] = field(default_factory=dict)
    grad_clip: float | None = None
    device: str = "auto"


class Trainer:
    """Flexible trainer for PyTorch models.

    Supports:
    - Generic models with any forward signature
    - Loss functions returning single tensor or dict of components
    - Callbacks for checkpointing, early stopping, progress bars, etc.
    - Metrics computed over epochs

    Example:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=my_loss,
            callbacks=[ModelCheckpoint(), EarlyStopping()],
            metrics=[RMSD()],
        )
        history = trainer.fit(train_loader, val_loader, epochs=100)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable[..., Tensor | dict[str, Tensor]],
        scheduler: LRScheduler | None = None,
        callbacks: list[Callback] | None = None,
        metrics: list[Metric] | None = None,
        config: TrainerConfig | None = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: PyTorch model to train.
            optimizer: Optimizer for updating model parameters.
            loss_fn: Loss function. Can return single tensor or dict of named losses.
            scheduler: Optional learning rate scheduler.
            callbacks: List of callbacks to use during training.
            metrics: List of metrics to track.
            config: Trainer configuration.
        """
        self.config = config or TrainerConfig()
        self.device = get_device(self.config.device)

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

        self.callbacks = CallbackList(callbacks)
        self.callbacks.set_trainer(self)

        self.metrics = MetricCollection(metrics)

        # Training state
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.stop_training: bool = False
        self.history: dict[str, list[float]] = {}

    def fit(
        self,
        train_data: Iterator,
        val_data: Iterator | None = None,
        epochs: int = 1,
    ) -> dict[str, list[float]]:
        """Train the model.

        Args:
            train_data: Training data iterator (DataLoader or similar).
            val_data: Optional validation data iterator.
            epochs: Number of epochs to train.

        Returns:
            Dictionary mapping metric names to lists of values per epoch.
        """
        self.stop_training = False
        self.callbacks.on_train_begin()

        try:
            for epoch in range(epochs):
                if self.stop_training:
                    break

                self.current_epoch = epoch
                self.callbacks.on_epoch_begin(epoch)

                # Training phase
                train_metrics = self._train_epoch(train_data)

                # Validation phase
                val_metrics = {}
                if val_data is not None:
                    val_metrics = self._validate_epoch(val_data)

                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics}

                # Record history
                for name, value in epoch_metrics.items():
                    if name not in self.history:
                        self.history[name] = []
                    self.history[name].append(value)

                self.callbacks.on_epoch_end(epoch, epoch_metrics)

                # Step scheduler if present
                if self.scheduler is not None:
                    self.scheduler.step()

        finally:
            self.callbacks.on_train_end()

        return self.history

    def _train_epoch(self, data: Iterator) -> dict[str, float]:
        """Run one training epoch.

        Args:
            data: Training data iterator.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        loss_meter = AverageMeter()
        component_meters: dict[str, AverageMeter] = {}

        self.metrics.reset()

        for batch_idx, batch in enumerate(data):
            if self.stop_training:
                break

            self.callbacks.on_train_batch_begin(batch_idx)

            # Move batch to device
            batch = self._prepare_batch(batch)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self._forward(batch)
            loss, loss_components = self._compute_loss(batch, outputs)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            self.optimizer.step()
            self.global_step += 1

            # Track losses
            batch_size = self._get_batch_size(batch)
            loss_meter.update(loss.item(), batch_size)

            for name, value in loss_components.items():
                if name not in component_meters:
                    component_meters[name] = AverageMeter()
                component_meters[name].update(value, batch_size)

            # Update metrics
            self.metrics.update(batch, outputs)

            self.callbacks.on_train_batch_end(
                batch_idx,
                loss.item(),
                loss_components,
            )

        # Compute final metrics
        train_metrics = {"train_loss": loss_meter.avg}
        for name, meter in component_meters.items():
            train_metrics[f"train_{name}"] = meter.avg

        computed_metrics = self.metrics.compute()
        for name, value in computed_metrics.items():
            train_metrics[f"train_{name}"] = value

        return train_metrics

    @torch.no_grad()
    def _validate_epoch(self, data: Iterator) -> dict[str, float]:
        """Run one validation epoch.

        Args:
            data: Validation data iterator.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        loss_meter = AverageMeter()
        component_meters: dict[str, AverageMeter] = {}

        self.metrics.reset()

        for batch_idx, batch in enumerate(data):
            self.callbacks.on_val_batch_begin(batch_idx)

            batch = self._prepare_batch(batch)
            outputs = self._forward(batch)
            loss, loss_components = self._compute_loss(batch, outputs)

            batch_size = self._get_batch_size(batch)
            loss_meter.update(loss.item(), batch_size)

            for name, value in loss_components.items():
                if name not in component_meters:
                    component_meters[name] = AverageMeter()
                component_meters[name].update(value, batch_size)

            self.metrics.update(batch, outputs)

            self.callbacks.on_val_batch_end(
                batch_idx,
                loss.item(),
                loss_components,
            )

        # Compute final metrics
        val_metrics = {"val_loss": loss_meter.avg}
        for name, meter in component_meters.items():
            val_metrics[f"val_{name}"] = meter.avg

        computed_metrics = self.metrics.compute()
        for name, value in computed_metrics.items():
            val_metrics[f"val_{name}"] = value

        return val_metrics

    def _prepare_batch(self, batch: Any) -> dict[str, Any]:
        """Prepare batch for model.

        Handles conversion from various batch formats to dict.

        Args:
            batch: Input batch (dict, tuple, or tensor).

        Returns:
            Batch as dictionary with tensors on device.
        """
        if isinstance(batch, dict):
            return move_batch_to_device(batch, self.device)
        elif isinstance(batch, (tuple, list)):
            # Assume (coords, features) or (coords, features, ...)
            batch_dict = {}
            if len(batch) >= 1:
                batch_dict["coords"] = batch[0]
            if len(batch) >= 2:
                batch_dict["features"] = batch[1]
            for i, item in enumerate(batch[2:], start=2):
                batch_dict[f"item_{i}"] = item
            return move_batch_to_device(batch_dict, self.device)
        elif isinstance(batch, Tensor):
            return {"input": batch.to(self.device)}
        else:
            return batch

    def _forward(self, batch: dict[str, Any]) -> dict[str, Tensor] | Tensor:
        """Run forward pass.

        Args:
            batch: Prepared batch dictionary.

        Returns:
            Model outputs.
        """
        # Try standard geometric model signature first
        if "coords" in batch and "features" in batch:
            return self.model(batch["coords"], batch["features"])
        elif "input" in batch:
            return self.model(batch["input"])
        else:
            # Pass entire batch dict
            return self.model(**batch)

    def _compute_loss(
        self,
        batch: dict[str, Any],
        outputs: dict[str, Tensor] | Tensor,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute loss from outputs.

        Args:
            batch: Input batch.
            outputs: Model outputs.

        Returns:
            Tuple of (total_loss, loss_components_dict).
        """
        loss_result = self.loss_fn(batch, outputs)

        if isinstance(loss_result, dict):
            # Multi-term loss
            components = {k: v.item() for k, v in loss_result.items()}

            # Combine with weights
            total = torch.tensor(0.0, device=self.device)
            for name, value in loss_result.items():
                weight = self.config.loss_weights.get(name, 1.0)
                total = total + weight * value

            return total, components
        else:
            # Single loss tensor
            return loss_result, {"loss": loss_result.item()}

    def _get_batch_size(self, batch: dict[str, Any]) -> int:
        """Get batch size from batch dict.

        Args:
            batch: Batch dictionary.

        Returns:
            Batch size.
        """
        for value in batch.values():
            if isinstance(value, Tensor):
                return value.shape[0]
        return 1

    def save_checkpoint(self, path: str, **extra_data: Any) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint.
            **extra_data: Additional data to include in checkpoint.
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "history": self.history,
            **extra_data,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Dictionary of extra data from checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.history = checkpoint.get("history", {})

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Return extra data
        known_keys = {
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "global_step",
            "history",
            "scheduler_state_dict",
        }
        return {k: v for k, v in checkpoint.items() if k not in known_keys}
