"""SO(3)-Equivariant Variational Autoencoder.

This module implements an SO(3)-equivariant VAE for point clouds where:
- Each point gets its own latent vector z_i
- Latent space uses configurable representations (any combination of l=0, l=1, l=2, ...)
- Equivariance is maintained through equivariant encoder/decoder

Key insight: The Gaussian sampling doesn't need to be isotropic. Equivariance is ensured by:
1. Equivariant encoder: encode(R*x) = (R*mu, sigma)
2. Equivariant decoder: decode(R*z, R*coords) = R*decode(z, coords)

Example:
    >>> import lr_geom as lg
    >>> from lr_geom.vae import EquivariantVAE
    >>>
    >>> # Define representations
    >>> in_repr = lg.Repr([0, 1], mult=8)
    >>> latent_repr = lg.Repr([0, 1], mult=4)
    >>> out_repr = lg.Repr([1], mult=1)  # For coordinate reconstruction
    >>> hidden_repr = lg.Repr([0, 1], mult=16)
    >>>
    >>> # Create VAE
    >>> vae = EquivariantVAE(
    ...     in_repr=in_repr,
    ...     latent_repr=latent_repr,
    ...     out_repr=out_repr,
    ...     hidden_repr=hidden_repr,
    ...     k_neighbors=16,
    ... )
    >>>
    >>> # Forward pass
    >>> coords = torch.randn(50, 3)
    >>> features = torch.randn(50, in_repr.mult, in_repr.dim())
    >>> recon, mu, logvar = vae(coords, features)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..representations import Repr
from ..equivariant import RepNorm
from ..layers import EquivariantLinear, EquivariantTransformer


class VariationalHead(nn.Module):
    """Variational head that produces equivariant mu and invariant logvar.

    The mean (mu) is computed via an equivariant linear projection, ensuring
    that mu transforms correctly under SO(3) rotations. The log-variance
    (logvar) is computed from invariant features (norms) via an MLP,
    ensuring it remains invariant under rotations.

    Args:
        in_repr: Input representation from the encoder.
        latent_repr: Target latent space representation.
        hidden_dim: Hidden dimension for the logvar MLP. Defaults to 64.

    Example:
        >>> in_repr = Repr([0, 1], mult=8)
        >>> latent_repr = Repr([0, 1], mult=4)
        >>> head = VariationalHead(in_repr, latent_repr)
        >>> x = torch.randn(50, 8, 4)  # (N, mult, dim)
        >>> mu, logvar = head(x)
        >>> mu.shape
        torch.Size([50, 4, 4])
        >>> logvar.shape
        torch.Size([50, 4])
    """

    def __init__(
        self,
        in_repr: Repr,
        latent_repr: Repr,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.in_repr = in_repr
        self.latent_repr = latent_repr

        # mu: equivariant projection
        self.mu_head = EquivariantLinear(in_repr, latent_repr)

        # logvar: invariant (from norms via MLP)
        self.norm = RepNorm(in_repr)
        norm_features = in_repr.nreps() * in_repr.mult
        self.logvar_head = nn.Sequential(
            nn.Linear(norm_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_repr.mult),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute equivariant mu and invariant logvar.

        Args:
            x: Input features of shape (N, in_mult, in_dim).

        Returns:
            mu: Equivariant mean of shape (N, latent_mult, latent_dim).
            logvar: Invariant log-variance of shape (N, latent_mult).
        """
        mu = self.mu_head(x)  # (N, latent_mult, latent_dim) - equivariant
        norms = self.norm(x).flatten(-2)  # (N, in_mult * nreps) - invariant
        logvar = self.logvar_head(norms)  # (N, latent_mult) - invariant
        return mu, logvar


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Standard VAE reparameterization trick.

    Samples z = mu + std * eps where eps ~ N(0, I).

    Args:
        mu: Mean tensor of shape (N, mult, dim).
        logvar: Log-variance tensor of shape (N, mult).

    Returns:
        Sampled latent tensor of shape (N, mult, dim).
    """
    std = torch.exp(0.5 * logvar)  # (N, mult)
    eps = torch.randn_like(mu)  # (N, mult, dim)
    return mu + std.unsqueeze(-1) * eps  # (N, mult, dim)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence from N(mu, sigma^2) to N(0, I).

    Uses the closed-form expression:
    KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Args:
        mu: Mean tensor of shape (N, mult, dim).
        logvar: Log-variance tensor of shape (N, mult).

    Returns:
        Scalar KL divergence averaged over batch.
    """
    # Expand logvar to match mu dimensions
    logvar_expanded = logvar.unsqueeze(-1)  # (N, mult, 1)
    kl = -0.5 * (1 + logvar_expanded - mu.pow(2) - logvar_expanded.exp())
    return kl.sum(dim=(-1, -2)).mean()  # Sum over repr dims, mean over batch


class EquivariantVAE(nn.Module):
    """SO(3)-Equivariant Variational Autoencoder.

    A VAE that maintains SO(3) equivariance through equivariant encoder and
    decoder transformers. Each point in the input point cloud gets its own
    latent vector, and all representations (input, latent, output) are
    fully configurable.

    The equivariance property ensures:
    - encode(R*x) = (R*mu, sigma) - mu rotates, sigma invariant
    - decode(R*z, R*coords) = R*decode(z, coords)

    Args:
        in_repr: Input feature representation.
        latent_repr: Latent space representation.
        out_repr: Output representation (decoder output).
        hidden_repr: Hidden layer representation for both encoder and decoder.
        encoder_layers: Number of transformer blocks in encoder. Defaults to 4.
        decoder_layers: Number of transformer blocks in decoder. Defaults to 4.
        k_neighbors: Number of neighbors for k-NN graph construction.
        edge_dim: Dimension of edge features. Defaults to 16.
        edge_hidden_dim: Hidden dimension for edge MLP. Defaults to 32.
        nheads: Number of attention heads. Defaults to 4.
        dropout: Dropout probability. Defaults to 0.0.
        attn_dropout: Attention dropout probability. Defaults to 0.0.
        var_hidden_dim: Hidden dimension for variational head MLP. Defaults to 64.
        residual_scale: Scale factor for residual connections. Use < 1.0
            (e.g., 0.1-0.5) for deep networks to improve gradient flow.
        attention_type: "node_wise" or "edge_wise" attention pattern. Defaults to "node_wise".
        scale_type: Attention scaling - "sqrt_head_dim", "sqrt_dim", "learned", "none".
        skip_type: Skip connection type - "scaled", "gated", or "none".
        rbf_type: Radial basis function type - "gaussian", "bessel", or "polynomial".
        rbf_r_min: Minimum radius for RBF initialization.
        rbf_r_max: Maximum radius for RBF initialization/cutoff.
        radial_weight_rank: Rank for low-rank RadialWeight decomposition. None for full rank.

    Example:
        >>> in_repr = Repr([0, 1], mult=8)
        >>> latent_repr = Repr([0, 1], mult=4)
        >>> out_repr = Repr([1], mult=1)  # Coordinates
        >>> hidden_repr = Repr([0, 1], mult=16)
        >>>
        >>> vae = EquivariantVAE(
        ...     in_repr=in_repr,
        ...     latent_repr=latent_repr,
        ...     out_repr=out_repr,
        ...     hidden_repr=hidden_repr,
        ...     k_neighbors=16,
        ... )
        >>>
        >>> coords = torch.randn(50, 3)
        >>> features = torch.randn(50, in_repr.mult, in_repr.dim())
        >>> recon, mu, logvar = vae(coords, features)
    """

    def __init__(
        self,
        in_repr: Repr,
        latent_repr: Repr,
        out_repr: Repr,
        hidden_repr: Repr,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        k_neighbors: int = 16,
        edge_dim: int = 16,
        edge_hidden_dim: int = 32,
        nheads: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        var_hidden_dim: int = 64,
        residual_scale: float = 1.0,
        attention_type: str = "node_wise",
        scale_type: str = "sqrt_head_dim",
        skip_type: str = "scaled",
        rbf_type: str = "gaussian",
        rbf_r_min: float = 0.0,
        rbf_r_max: float = 10.0,
        radial_weight_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.in_repr = in_repr
        self.latent_repr = latent_repr
        self.out_repr = out_repr
        self.hidden_repr = hidden_repr

        # Encoder: in_repr -> latent_repr
        self.encoder = EquivariantTransformer(
            in_repr=in_repr,
            out_repr=latent_repr,
            hidden_repr=hidden_repr,
            hidden_layers=encoder_layers,
            k_neighbors=k_neighbors,
            edge_dim=edge_dim,
            edge_hidden_dim=edge_hidden_dim,
            nheads=nheads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            residual_scale=residual_scale,
            attention_type=attention_type,
            scale_type=scale_type,
            skip_type=skip_type,
            rbf_type=rbf_type,
            rbf_r_min=rbf_r_min,
            rbf_r_max=rbf_r_max,
            radial_weight_rank=radial_weight_rank,
        )

        # Variational head
        self.var_head = VariationalHead(
            latent_repr, latent_repr, hidden_dim=var_hidden_dim
        )

        # Decoder: latent_repr -> out_repr
        self.decoder = EquivariantTransformer(
            in_repr=latent_repr,
            out_repr=out_repr,
            hidden_repr=hidden_repr,
            hidden_layers=decoder_layers,
            k_neighbors=k_neighbors,
            edge_dim=edge_dim,
            edge_hidden_dim=edge_hidden_dim,
            nheads=nheads,
            dropout=dropout,
            attn_dropout=attn_dropout,
            residual_scale=residual_scale,
            attention_type=attention_type,
            scale_type=scale_type,
            skip_type=skip_type,
            rbf_type=rbf_type,
            rbf_r_min=rbf_r_min,
            rbf_r_max=rbf_r_max,
            radial_weight_rank=radial_weight_rank,
        )

    def encode(
        self, coords: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            coords: Point coordinates of shape (N, 3).
            features: Input features of shape (N, in_mult, in_dim).

        Returns:
            mu: Equivariant mean of shape (N, latent_mult, latent_dim).
            logvar: Invariant log-variance of shape (N, latent_mult).
        """
        x = self.encoder(coords, features)
        return self.var_head(x)

    def decode(self, coords: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode latent samples to output representation.

        Args:
            coords: Point coordinates of shape (N, 3).
            z: Latent samples of shape (N, latent_mult, latent_dim).

        Returns:
            Decoded output of shape (N, out_mult, out_dim).
        """
        return self.decoder(coords, z)

    def forward(
        self, coords: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode, sample, decode.

        Args:
            coords: Point coordinates of shape (N, 3).
            features: Input features of shape (N, in_mult, in_dim).

        Returns:
            recon: Reconstructed output of shape (N, out_mult, out_dim).
            mu: Latent mean of shape (N, latent_mult, latent_dim).
            logvar: Latent log-variance of shape (N, latent_mult).
        """
        mu, logvar = self.encode(coords, features)
        z = reparameterize(mu, logvar)
        recon = self.decode(coords, z)
        return recon, mu, logvar

    def sample(
        self,
        coords: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Sample from prior and decode.

        Args:
            coords: Point coordinates of shape (N, 3).
            num_samples: Number of samples to generate. Currently only 1 supported.

        Returns:
            Decoded samples of shape (N, out_mult, out_dim).
        """
        N = coords.size(0)
        device = coords.device
        dtype = coords.dtype

        z = torch.randn(
            N,
            self.latent_repr.mult,
            self.latent_repr.dim(),
            device=device,
            dtype=dtype,
        )
        return self.decode(coords, z)

    def compile(
        self,
        mode: str = "default",
        fullgraph: bool = False,
    ) -> "EquivariantVAE":
        """Compile the model using torch.compile for faster execution.

        Applies torch.compile to optimize the forward pass with kernel fusion
        and graph optimizations. Compiles encoder and decoder separately for
        better optimization.

        Args:
            mode: Compilation mode. Options:
                - "default": Balanced compilation (default, good memory usage)
                - "reduce-overhead": Uses CUDA graphs (faster but high memory usage)
                - "max-autotune": Maximum optimization (slower compile, faster runtime)
            fullgraph: If True, requires the entire forward to compile as one graph.
                Set to False (default) to allow graph breaks for compatibility.

        Returns:
            Self for method chaining.

        Example:
            >>> vae = EquivariantVAE(...).compile()
            >>> recon, mu, logvar = vae(coords, features)  # Compiled execution
        """
        # Compile encoder and decoder transformers
        self.encoder.compile(mode=mode, fullgraph=fullgraph)
        self.decoder.compile(mode=mode, fullgraph=fullgraph)
        return self
