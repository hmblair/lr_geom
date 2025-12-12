"""Pre-built model configurations.

Builder functions for commonly used model configurations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from lr_geom.training.factory.registry import model_registry

if TYPE_CHECKING:
    from lr_geom.config import ModelConfig


@model_registry.register("equivariant_vae")
def build_equivariant_vae(
    config: ModelConfig,
    num_atom_types: int | None = None,
) -> nn.Module:
    """Build an EquivariantVAE model.

    Args:
        config: Model configuration containing:
            - lmax: Maximum spherical harmonic degree
            - mult: Multiplicity of representations
            - num_encoder_layers: Number of encoder transformer layers
            - num_decoder_layers: Number of decoder transformer layers
            - latent_dim: Dimension of latent space
            - cutoff: Radial cutoff distance
            - num_bases: Number of radial basis functions
            - k_neighbors: Number of neighbors for sparse attention (None for dense)
        num_atom_types: Number of atom types for embedding. If None, no embedding.

    Returns:
        EquivariantVAE model.
    """
    from lr_geom.vae import EquivariantVAE

    vae = EquivariantVAE(
        lmax=config.lmax,
        mult=config.mult,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        latent_dim=config.latent_dim,
        cutoff=config.cutoff,
        num_bases=config.num_bases,
        k_neighbors=config.k_neighbors,
    )

    return vae


@model_registry.register("embedding_and_vae")
def build_embedding_and_vae(
    config: ModelConfig,
    num_atom_types: int,
) -> tuple[nn.Module, nn.Module]:
    """Build an embedding layer and EquivariantVAE model.

    This is the typical setup for molecular structure VAE where
    atom types need to be embedded before passing to the VAE.

    Args:
        config: Model configuration.
        num_atom_types: Number of distinct atom types.

    Returns:
        Tuple of (embedding, vae) modules.
    """
    from lr_geom.vae import EquivariantVAE

    # Build embedding
    embedding_dim = config.mult * (config.lmax + 1) ** 2
    embedding = nn.Embedding(num_atom_types, embedding_dim)

    # Build VAE
    vae = EquivariantVAE(
        lmax=config.lmax,
        mult=config.mult,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        latent_dim=config.latent_dim,
        cutoff=config.cutoff,
        num_bases=config.num_bases,
        k_neighbors=config.k_neighbors,
    )

    return embedding, vae


def create_vae_loss_fn(beta: float = 1.0):
    """Create a VAE loss function with configurable KL weight.

    Args:
        beta: Weight for KL divergence term.

    Returns:
        Loss function compatible with Trainer.
    """
    import torch

    def vae_loss(batch, outputs):
        """Compute VAE loss with reconstruction and KL terms."""
        coords = batch["coords"]
        recon = outputs["reconstruction"]

        # Reconstruction loss (MSE)
        recon_loss = ((recon - coords) ** 2).sum(dim=-1).mean()

        # KL divergence
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / mu.shape[0]  # Average over batch

        return {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    return vae_loss
