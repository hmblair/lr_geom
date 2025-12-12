"""Training system for lr_geom.

Provides a flexible training framework with:
- Trainer class for training PyTorch models
- Callbacks for checkpointing, early stopping, progress bars
- Metrics for tracking reconstruction quality and VAE-specific metrics
- Model factory for building models from configuration

Example:
    from lr_geom.training import (
        Trainer, TrainerConfig,
        ModelCheckpoint, EarlyStopping, ProgressBar,
        RMSD, KLDivergence,
        model_registry,
    )

    # Build model
    model = model_registry.build("equivariant_vae", config=config)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=my_loss,
        callbacks=[
            ModelCheckpoint(dirpath="checkpoints", monitor="val_loss"),
            EarlyStopping(patience=10),
            ProgressBar(),
        ],
        metrics=[RMSD(), KLDivergence()],
        config=TrainerConfig(loss_weights={"recon": 1.0, "kl": 0.01}),
    )

    # Train
    history = trainer.fit(train_loader, val_loader, epochs=100)
"""
from __future__ import annotations

# Trainer
from lr_geom.training.trainer import Trainer, TrainerConfig

# Utilities
from lr_geom.training.utils import get_device, move_batch_to_device, set_seed

# Protocols
from lr_geom.training.protocols import GeometricModel, LossFunction, VAEModel

# Callbacks
from lr_geom.training.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    ModelCheckpoint,
    ProgressBar,
)

# Metrics
from lr_geom.training.metrics import (
    ELBO,
    MSE,
    RMSD,
    AverageMeter,
    KLDivergence,
    Metric,
    MetricCollection,
    ReconstructionLoss,
)

# Factory
from lr_geom.training.factory import (
    ModelRegistry,
    build_embedding_and_vae,
    build_equivariant_vae,
    create_vae_loss_fn,
    model_registry,
)

# Experiment management
from lr_geom.training.experiment_manager import ExperimentManager, ExperimentResult

__all__ = [
    # Trainer
    "Trainer",
    "TrainerConfig",
    # Utilities
    "set_seed",
    "get_device",
    "move_batch_to_device",
    # Protocols
    "GeometricModel",
    "VAEModel",
    "LossFunction",
    # Callbacks
    "Callback",
    "CallbackList",
    "ModelCheckpoint",
    "EarlyStopping",
    "ProgressBar",
    # Metrics
    "Metric",
    "MetricCollection",
    "AverageMeter",
    "RMSD",
    "MSE",
    "KLDivergence",
    "ReconstructionLoss",
    "ELBO",
    # Factory
    "ModelRegistry",
    "model_registry",
    "build_equivariant_vae",
    "build_embedding_and_vae",
    "create_vae_loss_fn",
    # Experiment management
    "ExperimentManager",
    "ExperimentResult",
]
