"""SO(3)-Equivariant Variational Autoencoder module.

This module provides an SO(3)-equivariant VAE for point clouds with:
- Per-point latent vectors
- Configurable representations for input, latent, and output spaces
- Equivariance through equivariant encoder/decoder transformers

Classes:
    EquivariantVAE: Main VAE model
    VariationalHead: Produces equivariant mu and invariant logvar

Functions:
    reparameterize: VAE reparameterization trick
    kl_divergence: KL divergence loss for VAE training

Example:
    >>> import lr_geom as lg
    >>> from lr_geom.vae import EquivariantVAE, kl_divergence
    >>>
    >>> vae = EquivariantVAE(
    ...     in_repr=lg.Repr([0, 1], mult=8),
    ...     latent_repr=lg.Repr([0, 1], mult=4),
    ...     out_repr=lg.Repr([1], mult=1),
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
    reparameterize,
    kl_divergence,
)

__all__ = [
    "EquivariantVAE",
    "VariationalHead",
    "reparameterize",
    "kl_divergence",
]
