"""Protocol definitions for training system.

Defines structural typing protocols for models compatible with the training system.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class GeometricModel(Protocol):
    """Protocol for geometric models that operate on coordinates and features.

    Models implementing this protocol accept:
    - coords: (N, 3) or (B, N, 3) tensor of 3D coordinates
    - features: (N, F) or (B, N, F) tensor of node features

    The output format is model-specific.
    """

    def forward(self, coords: Tensor, features: Tensor) -> Tensor | dict[str, Tensor]:
        """Forward pass through the model.

        Args:
            coords: Coordinate tensor of shape (N, 3) or (B, N, 3).
            features: Feature tensor of shape (N, F) or (B, N, F).

        Returns:
            Model output, either a tensor or dict of tensors.
        """
        ...


@runtime_checkable
class VAEModel(Protocol):
    """Protocol for Variational Autoencoder models.

    VAE models implement encode/decode/sample methods in addition to forward.
    """

    def forward(
        self,
        coords: Tensor,
        features: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass returning reconstruction and latent statistics.

        Args:
            coords: Coordinate tensor of shape (N, 3) or (B, N, 3).
            features: Feature tensor of shape (N, F) or (B, N, F).

        Returns:
            Dictionary containing at minimum:
            - "reconstruction": Reconstructed coordinates
            - "mu": Latent mean
            - "logvar": Latent log-variance
        """
        ...

    def encode(
        self,
        coords: Tensor,
        features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode inputs to latent distribution parameters.

        Args:
            coords: Coordinate tensor.
            features: Feature tensor.

        Returns:
            Tuple of (mu, logvar) for the latent distribution.
        """
        ...

    def decode(
        self,
        z: Tensor,
        coords: Tensor,
        features: Tensor,
    ) -> Tensor:
        """Decode latent vectors to reconstructions.

        Args:
            z: Latent vectors.
            coords: Original coordinates (for conditioning).
            features: Original features (for conditioning).

        Returns:
            Reconstructed coordinates.
        """
        ...

    def sample(
        self,
        coords: Tensor,
        features: Tensor,
    ) -> Tensor:
        """Generate samples from the prior.

        Args:
            coords: Coordinate tensor (for conditioning).
            features: Feature tensor (for conditioning).

        Returns:
            Sampled coordinates.
        """
        ...


@runtime_checkable
class LossFunction(Protocol):
    """Protocol for loss functions.

    Loss functions can return either a single tensor or a dictionary
    mapping loss component names to their values.
    """

    def __call__(
        self,
        batch: dict[str, Any],
        outputs: dict[str, Tensor] | Tensor,
    ) -> Tensor | dict[str, Tensor]:
        """Compute loss.

        Args:
            batch: Input batch dictionary.
            outputs: Model outputs.

        Returns:
            Either a single loss tensor or dict mapping names to loss components.
        """
        ...
