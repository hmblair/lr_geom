"""SO(3)-Equivariant Variational Autoencoder module.

This module provides an SO(3)-equivariant VAE for point clouds with:
- Per-point latent vectors
- Configurable representations for input, latent, and output spaces
- Equivariance through equivariant encoder/decoder transformers
- Flexible conditioning (decoder can use atom types, residue embeddings, etc.)

Classes:
    EquivariantVAE: Main VAE model
    VariationalHead: Produces equivariant mu and invariant logvar
    ConditioningProjection: Projects conditioning to match latent representation

Functions:
    reparameterize: VAE reparameterization trick
    kl_divergence: KL divergence loss for VAE training

Example:
    >>> import lr_geom as lg
    >>> from lr_geom.vae import EquivariantVAE, kl_divergence
    >>>
    >>> # Scalar atom embeddings with vector latent space
    >>> vae = EquivariantVAE(
    ...     in_repr=lg.Repr([0], mult=8),      # Scalar atom embeddings
    ...     latent_repr=lg.Repr([0, 1], mult=4),  # Vector latent space
    ...     out_repr=lg.Repr([1], mult=1),     # Coordinate outputs
    ...     hidden_repr=lg.Repr([0, 1], mult=16),
    ...     k_neighbors=16,
    ... )
    >>>
    >>> recon, mu, logvar = vae(coords, features)
    >>> kl_loss = kl_divergence(mu, logvar)
"""

from .model import (
    EquivariantVAE,
    VariationalHead,
    ConditioningProjection,
    reparameterize,
    kl_divergence,
)

__all__ = [
    "EquivariantVAE",
    "VariationalHead",
    "ConditioningProjection",
    "reparameterize",
    "kl_divergence",
]
