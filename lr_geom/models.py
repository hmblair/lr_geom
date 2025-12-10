"""Pre-built geometric deep learning models.

This module provides ready-to-use models built from geometric deep
learning primitives.

Classes:
    GNMA: Gaussian Network Model Attention

Note:
    RadialWeight has been moved to layers.py but is re-exported here
    for backward compatibility.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .alignment import gnm_correlations
from .equivariant import RadialBasisFunctions
from .nn import DenseNetwork

# Re-export RadialWeight for backward compatibility
# RadialWeight is now defined in layers.py as it's a primitive layer
from .layers import RadialWeight

__all__ = ["RadialWeight", "GNMA"]


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
