"""SO(3)-Equivariant Variational Autoencoder.

This module implements an SO(3)-equivariant VAE for point clouds where:
- Each point gets its own latent vector z_i
- Latent space uses configurable representations (any combination of l=0, l=1, l=2, ...)
- Equivariance is maintained through equivariant encoder/decoder
- Decoder receives conditioning features (e.g., atom types) to inform reconstruction

Key insight: The Gaussian sampling doesn't need to be isotropic. Equivariance is ensured by:
1. Equivariant encoder: encode(R*x) = (R*mu, sigma)
2. Equivariant decoder: decode(R*z, R*cond, R*coords) = R*decode(z, cond, coords)

The conditioning mechanism allows the decoder to know what type of atom/residue is at
each position, which is essential for learning chemically meaningful reconstructions.

Example:
    >>> import lr_geom as lg
    >>> from lr_geom.vae import EquivariantVAE
    >>>
    >>> # Define representations
    >>> in_repr = lg.Repr([0, 1], mult=8)   # Atom type embeddings
    >>> latent_repr = lg.Repr([0, 1], mult=4)
    >>> out_repr = lg.Repr([1], mult=1)     # For coordinate reconstruction
    >>> hidden_repr = lg.Repr([0, 1], mult=16)
    >>>
    >>> # Create VAE - decoder will receive latent + atom embeddings
    >>> vae = EquivariantVAE(
    ...     in_repr=in_repr,
    ...     latent_repr=latent_repr,
    ...     out_repr=out_repr,
    ...     hidden_repr=hidden_repr,
    ...     k_neighbors=16,
    ...     cond_repr=in_repr,  # Use atom embeddings as conditioning
    ... )
    >>>
    >>> # Forward pass
    >>> coords = torch.randn(50, 3)
    >>> features = torch.randn(50, in_repr.mult, in_repr.dim())
    >>> recon, mu, logvar = vae(coords, features)  # features used as conditioning
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from ..representations import Repr
from ..equivariant import RepNorm
from ..layers import EquivariantLinear, EquivariantTransformer


def _combine_reprs(repr1: Repr, repr2: Repr) -> Repr:
    """Combine two representations by concatenating multiplicities.

    Both representations must have the same lvals. The result has
    mult = repr1.mult + repr2.mult.

    Args:
        repr1: First representation.
        repr2: Second representation.

    Returns:
        Combined representation.

    Raises:
        ValueError: If lvals don't match.
    """
    if repr1.lvals != repr2.lvals:
        raise ValueError(
            f"Cannot combine representations with different lvals: "
            f"{repr1.lvals} vs {repr2.lvals}"
        )
    combined = deepcopy(repr1)
    combined.mult = repr1.mult + repr2.mult
    return combined


class ConditioningProjection(nn.Module):
    """Project conditioning features to match target representation's lvals.

    When conditioning has fewer angular momenta than the target (e.g., scalars
    only while target has scalars + vectors), this layer:
    1. Applies an equivariant linear to shared lvals (projecting multiplicity)
    2. Pads missing lvals with zeros

    This allows flexible conditioning where atom type embeddings (scalars only)
    can be used with latent spaces that include higher-order components.

    Args:
        cond_repr: Source conditioning representation.
        target_repr: Target representation (typically latent_repr).
            The output will have target_repr's lvals with cond_repr's projected mult.
    """

    def __init__(self, cond_repr: Repr, target_repr: Repr) -> None:
        super().__init__()
        self.cond_repr = cond_repr
        self.target_repr = target_repr

        # Check which lvals are shared vs need padding
        self.shared_lvals = [l for l in cond_repr.lvals if l in target_repr.lvals]
        self.pad_lvals = [l for l in target_repr.lvals if l not in cond_repr.lvals]

        # Create projected representation with shared lvals
        if self.shared_lvals:
            self.shared_repr = Repr(self.shared_lvals, mult=cond_repr.mult)
            self.projected_repr = Repr(self.shared_lvals, mult=target_repr.mult)
            self.proj = EquivariantLinear(self.shared_repr, self.projected_repr)
        else:
            self.proj = None

        # Compute output representation (target's lvals, target's mult)
        self.out_repr = Repr(target_repr.lvals, mult=target_repr.mult)

        # Precompute indices for assembling output
        self._compute_indices()

    def _compute_indices(self) -> None:
        """Precompute source and destination indices for efficient forward."""
        # Source indices: where to read from projected/cond features
        # Dest indices: where to write in output

        self.src_indices = []
        self.dst_indices = []

        # For each l in target, find where it comes from
        cond_cdims = self.cond_repr.cumdims()
        target_cdims = self.target_repr.cumdims()

        for i, l in enumerate(self.target_repr.lvals):
            dst_start = target_cdims[i]
            dst_end = target_cdims[i + 1]

            if l in self.cond_repr.lvals:
                # This l exists in conditioning - will come from projection
                cond_idx = self.cond_repr.lvals.index(l)
                src_start = cond_cdims[cond_idx]
                src_end = cond_cdims[cond_idx + 1]
                self.src_indices.append((src_start, src_end))
            else:
                # This l doesn't exist - will be zeros
                self.src_indices.append(None)

            self.dst_indices.append((dst_start, dst_end))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """Project conditioning features to target representation.

        Args:
            cond: Conditioning features of shape (N, cond_mult, cond_dim).

        Returns:
            Projected features of shape (N, target_mult, target_dim).
        """
        N = cond.size(0)
        device = cond.device
        dtype = cond.dtype

        # Initialize output with zeros
        out = torch.zeros(
            N, self.target_repr.mult, self.target_repr.dim(),
            device=device, dtype=dtype
        )

        if self.proj is None:
            # No shared lvals - return zeros
            return out

        # Extract shared lvals from conditioning
        shared_features = []
        cond_cdims = self.cond_repr.cumdims()
        for l in self.shared_lvals:
            idx = self.cond_repr.lvals.index(l)
            start, end = cond_cdims[idx], cond_cdims[idx + 1]
            shared_features.append(cond[:, :, start:end])

        # Concatenate shared features
        shared_cond = torch.cat(shared_features, dim=-1)  # (N, cond_mult, shared_dim)

        # Project to target multiplicity
        projected = self.proj(shared_cond)  # (N, target_mult, shared_dim)

        # Place projected features in output at correct positions
        proj_cdims = self.projected_repr.cumdims()
        target_cdims = self.target_repr.cumdims()

        for i, l in enumerate(self.target_repr.lvals):
            if l in self.shared_lvals:
                shared_idx = self.shared_lvals.index(l)
                proj_start = proj_cdims[shared_idx]
                proj_end = proj_cdims[shared_idx + 1]
                tgt_start = target_cdims[i]
                tgt_end = target_cdims[i + 1]
                out[:, :, tgt_start:tgt_end] = projected[:, :, proj_start:proj_end]
            # else: remains zeros (already initialized)

        return out


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

    The decoder receives conditioning features (e.g., atom type embeddings)
    concatenated with the latent samples. This allows the decoder to know
    what type of atom/residue is at each position, which is essential for
    learning chemically meaningful reconstructions.

    The equivariance property ensures:
    - encode(R*x) = (R*mu, sigma) - mu rotates, sigma invariant
    - decode(R*z, R*cond, R*coords) = R*decode(z, cond, coords)

    Args:
        in_repr: Input feature representation (encoder input).
        latent_repr: Latent space representation.
        out_repr: Output representation (decoder output).
        hidden_repr: Hidden layer representation for both encoder and decoder.
        cond_repr: Conditioning representation for decoder. If None, defaults to
            in_repr. The decoder receives latent_repr + cond_repr as input.
            This allows flexible conditioning on atom types, residue types,
            element embeddings, or any combination thereof.
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
        >>> # Basic usage - conditioning on input features (atom embeddings)
        >>> in_repr = Repr([0, 1], mult=8)   # Atom type embeddings
        >>> latent_repr = Repr([0, 1], mult=4)
        >>> out_repr = Repr([1], mult=1)     # Coordinates
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
        >>>
        >>> # Advanced: separate conditioning (e.g., residue + element embeddings)
        >>> cond_repr = Repr([0, 1], mult=12)  # Larger conditioning
        >>> vae = EquivariantVAE(..., cond_repr=cond_repr)
        >>> cond_features = torch.randn(50, cond_repr.mult, cond_repr.dim())
        >>> recon, mu, logvar = vae(coords, features, cond_features)
    """

    def __init__(
        self,
        in_repr: Repr,
        latent_repr: Repr,
        out_repr: Repr,
        hidden_repr: Repr,
        cond_repr: Repr | None = None,
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

        # Conditioning representation defaults to input representation
        # This allows decoder to know atom types, residue types, etc.
        self.cond_repr = cond_repr if cond_repr is not None else in_repr

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

        # Conditioning projection: map cond_repr -> latent_repr's structure
        # This handles cases where cond_repr has different lvals than latent_repr
        # (e.g., scalar atom embeddings with vector latent space)
        self.cond_proj = ConditioningProjection(self.cond_repr, latent_repr)

        # Decoder: (latent_repr + projected_cond_repr) -> out_repr
        # The decoder receives concatenated [z, projected_conditioning] as input
        # Both z and projected_cond have latent_repr's lvals
        decoder_in_repr = _combine_reprs(latent_repr, latent_repr)  # 2x latent mult
        self.decoder = EquivariantTransformer(
            in_repr=decoder_in_repr,
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

    def decode(
        self,
        coords: torch.Tensor,
        z: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Decode latent samples to output representation.

        The decoder receives both the latent samples and conditioning features
        (e.g., atom type embeddings) to inform the reconstruction.

        Args:
            coords: Point coordinates of shape (N, 3).
            z: Latent samples of shape (N, latent_mult, latent_dim).
            cond: Conditioning features of shape (N, cond_mult, cond_dim).
                Must match cond_repr specification.

        Returns:
            Decoded output of shape (N, out_mult, out_dim).
        """
        # Project conditioning to match latent representation's lvals
        cond_projected = self.cond_proj(cond)  # (N, latent_mult, latent_dim)

        # Concatenate latent samples with projected conditioning along multiplicity dim
        decoder_input = torch.cat([z, cond_projected], dim=1)  # (N, 2*latent_mult, latent_dim)
        return self.decoder(coords, decoder_input)

    def forward(
        self,
        coords: torch.Tensor,
        features: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode, sample, decode.

        Args:
            coords: Point coordinates of shape (N, 3).
            features: Input features of shape (N, in_mult, in_dim).
            cond: Optional conditioning features of shape (N, cond_mult, cond_dim).
                If None, uses `features` as conditioning (default behavior).
                This allows separate encoder input and decoder conditioning.

        Returns:
            recon: Reconstructed output of shape (N, out_mult, out_dim).
            mu: Latent mean of shape (N, latent_mult, latent_dim).
            logvar: Latent log-variance of shape (N, latent_mult).
        """
        # Default: use input features as conditioning
        if cond is None:
            cond = features

        mu, logvar = self.encode(coords, features)
        z = reparameterize(mu, logvar)
        recon = self.decode(coords, z, cond)
        return recon, mu, logvar

    def sample(
        self,
        coords: torch.Tensor,
        cond: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Sample from prior and decode.

        Requires conditioning features since the decoder needs to know
        atom types to generate meaningful structures.

        Args:
            coords: Point coordinates of shape (N, 3).
            cond: Conditioning features of shape (N, cond_mult, cond_dim).
                Required - decoder needs atom type information.
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
        return self.decode(coords, z, cond)

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
