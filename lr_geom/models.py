"""Pre-built geometric deep learning models.

This module provides ready-to-use models built from geometric deep
learning primitives.

Classes:
    RadialWeight: Neural network for computing tensor product weights
    GNMA: Gaussian Network Model Attention
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .representations import ProductRepr
from .alignment import _gnm_correlations
from .equivariant import RadialBasisFunctions
from .nn import DenseNetwork


class RadialWeight(nn.Module):
    """Compute tensor product weights from invariant edge features.

    A two-layer neural network that maps edge features to weights
    for tensor product contractions. Used in equivariant message
    passing networks.

    Args:
        edge_dim: Dimension of input edge features.
        hidden_dim: Hidden layer dimension.
        repr: ProductRepr specifying the tensor product structure.
        in_dim: Input multiplicity.
        out_dim: Output multiplicity.
        dropout: Dropout probability.

    Example:
        >>> repr = ProductRepr(Repr([1]), Repr([1]))
        >>> weight_net = RadialWeight(16, 32, repr, 8, 8)
        >>> edge_features = torch.randn(100, 16)
        >>> weights = weight_net(edge_features)
    """

    def __init__(
        self: RadialWeight,
        edge_dim: int,
        hidden_dim: int,
        repr: ProductRepr,
        in_dim: int,
        out_dim: int,
        dropout: float = 0,
    ) -> None:
        super().__init__()

        self.nl1 = repr.rep1.nreps()
        self.nl2 = repr.rep2.nreps()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_dim_flat = self.nl1 * self.nl2 * in_dim * out_dim

        self.layer1 = nn.Linear(edge_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, self.out_dim_flat)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self: RadialWeight, x: torch.Tensor) -> torch.Tensor:
        """Compute weights from edge features.

        Args:
            x: Edge features of shape (..., edge_dim).

        Returns:
            Weights of shape (..., nl2 * out_dim, nl1 * in_dim).
        """
        *b, _ = x.size()

        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)

        return self.layer2(x).view(
            *b, self.nl2 * self.out_dim,
            self.nl1 * self.in_dim,
        )


class GNMA(nn.Module):
    """Gaussian Network Model Attention.

    Applies attention based on Gaussian Network Model correlations,
    which capture the expected correlations between molecular positions
    under a harmonic potential model.

    Args:
        dim: Dimension of radial basis function embedding.
        out: Output dimension per coordinate.

    Example:
        >>> model = GNMA(dim=16)
        >>> coords = torch.randn(100, 3)  # 100 atoms
        >>> output = model(coords)  # shape: (100, 3)
    """

    def __init__(self: GNMA, dim: int, out: int = 1) -> None:
        super().__init__()

        self.rbf = RadialBasisFunctions(dim)
        self.linear = DenseNetwork(dim, out)

    def embed(self: GNMA, dists: torch.Tensor) -> torch.Tensor:
        """Embed pairwise distances using RBF and linear layer.

        Args:
            dists: Pairwise distances of shape (N, N).

        Returns:
            Embedded adjacency of shape (N, N).
        """
        adj = self.rbf(dists)
        return self.linear(adj).squeeze(-1)

    def forward(self: GNMA, coords: torch.Tensor) -> torch.Tensor:
        """Apply GNM attention to coordinates.

        Args:
            coords: Atomic coordinates of shape (N, 3).

        Returns:
            Attention-weighted coordinates of shape (N, 3).
        """
        pw = torch.cdist(coords, coords)
        emb = self.embed(pw)

        attn = _gnm_correlations(emb)
        return torch.softmax(attn, 1) @ coords
