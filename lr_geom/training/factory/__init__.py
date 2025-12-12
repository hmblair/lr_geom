"""Model factory system.

Provides model registry and builder functions.
"""
from __future__ import annotations

from lr_geom.training.factory.builders import (
    build_embedding_and_vae,
    build_equivariant_vae,
    create_vae_loss_fn,
)
from lr_geom.training.factory.registry import ModelRegistry, model_registry

__all__ = [
    "ModelRegistry",
    "model_registry",
    "build_equivariant_vae",
    "build_embedding_and_vae",
    "create_vae_loss_fn",
]
