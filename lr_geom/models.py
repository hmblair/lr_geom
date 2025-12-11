"""Pre-built geometric deep learning models.

This module provides ready-to-use models built from geometric deep
learning primitives.

Functions:
    graph_laplacian: Compute graph Laplacian (Kirchhoff matrix)
    gnm_correlations: Compute Gaussian Network Model correlations
    gnm_variances: Compute GNM position variances

Classes:
    GNMA: Gaussian Network Model Attention

Note:
    RadialWeight has been moved to layers.py but is re-exported here
    for backward compatibility.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .equivariant import RadialBasisFunctions
from .nn import DenseNetwork

# Re-export RadialWeight for backward compatibility
# RadialWeight is now defined in layers.py as it's a primitive layer
from .layers import RadialWeight

__all__ = [
    "RadialWeight",
    "GNMA",
    "graph_laplacian",
    "gnm_correlations",
    "gnm_variances",
]


def graph_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """Compute the graph Laplacian (Kirchhoff matrix).

    The graph Laplacian is defined as L = D - A, where D is the
    degree matrix and A is the adjacency matrix.

    Args:
        adj: Adjacency matrix of shape (N, N).

    Returns:
        Laplacian matrix of shape (N, N).

    Example:
        >>> adj = torch.tensor([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        >>> L = graph_laplacian(adj)
    """
    deg = torch.diag(adj.sum(1))
    return deg - adj


def gnm_correlations(adj: torch.Tensor) -> torch.Tensor:
    """Compute correlations under a Gaussian Network Model.

    The GNM models molecular dynamics as a network of springs,
    with correlations given by the pseudo-inverse of the Laplacian.

    Args:
        adj: Adjacency/connectivity matrix of shape (N, N).

    Returns:
        Correlation matrix of shape (N, N).

    Example:
        >>> adj = torch.tensor([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        >>> corr = gnm_correlations(adj)
    """
    lap = graph_laplacian(adj)
    return torch.linalg.pinv(lap, rtol=1e-2)


def gnm_variances(adj: torch.Tensor) -> torch.Tensor:
    """Compute position variances under a Gaussian Network Model.

    Returns the diagonal of the GNM correlation matrix, which
    represents the variance in position for each node.

    Args:
        adj: Adjacency/connectivity matrix of shape (N, N).

    Returns:
        Variance vector of shape (N,).

    Example:
        >>> adj = torch.tensor([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])
        >>> var = gnm_variances(adj)
    """
    return torch.diagonal(gnm_correlations(adj))


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

        attn = gnm_correlations(emb)
        return torch.softmax(attn, 1) @ coords
