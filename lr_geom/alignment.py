"""Point cloud alignment and scoring utilities.

This module provides functions for aligning point clouds using the
Kabsch algorithm and computing structural similarity metrics.

Functions:
    rmsd: Compute root mean square deviation between point clouds
    get_kabsch_rotation_matrix: Find optimal rotation matrix
    kabsch_align: Align two point clouds
    graph_laplacian: Compute graph Laplacian
    gnm_correlations: Compute Gaussian Network Model correlations
    gnm_variances: Compute GNM position variances

Classes:
    RMSD: nn.Module for computing alignment loss
"""
from __future__ import annotations

import torch
import torch.nn as nn


def rmsd(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor | None:
    """Compute the RMSD between two point clouds using Kabsch algorithm.

    The root mean square deviation (RMSD) measures the average distance
    between atoms of two superimposed point clouds after optimal alignment.

    Args:
        x: First point cloud of shape (..., N, 3).
        y: Second point cloud of shape (..., N, 3).

    Returns:
        The RMSD value, or None if SVD computation failed.

    Example:
        >>> x = torch.randn(100, 3)
        >>> y = torch.randn(100, 3)
        >>> distance = rmsd(x, y)
    """
    # Center the coordinates
    x = x - x.mean(-2, keepdim=True)
    y = y - y.mean(-2, keepdim=True)

    # Compute the covariance matrix
    cov = torch.einsum('...ji,...jk->...ik', x, y) / x.size(-2)

    # Get the singular values, handling potential numerical issues
    try:
        sigma = torch.linalg.svdvals(cov)
    except Exception:
        return None

    det = torch.linalg.det(cov)

    # Clone to preserve gradients and handle reflection case
    sigma = sigma.clone()
    sigma[det < 0, -1] = -sigma[det < 0, -1]
    sigma = sigma.mean()

    # Get the variances of the point clouds
    var1 = (x ** 2).mean()
    var2 = (y ** 2).mean()

    # Compute the Kabsch distance
    return var1 + var2 - 2 * sigma


def get_kabsch_rotation_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Find the optimal rotation matrix using the Kabsch algorithm.

    Computes the rotation matrix in SO(n) that best aligns point cloud x
    with point cloud y using singular value decomposition.

    Args:
        x: Source point cloud of shape (..., N, D).
        y: Target point cloud of shape (..., N, D).
        weight: Optional weights for each point, shape (..., N).

    Returns:
        Rotation matrix of shape (..., D, D).

    Example:
        >>> x = torch.randn(100, 3)
        >>> y = torch.randn(100, 3)
        >>> R = get_kabsch_rotation_matrix(x, y)
        >>> aligned = x @ R
    """
    # Calculate the covariance matrix
    if weight is not None:
        C = torch.einsum(
            '...ji,...j,...jk->...ik',
            x, weight, y,
        ) / weight.mean(-1)
    else:
        C = torch.einsum('...ji,...jk->...ik', x, y)

    # Calculate the singular value decomposition
    U, _, Vh = torch.linalg.svd(C)

    # Get the correction factor for proper rotation (not reflection)
    d = torch.linalg.det(U) * torch.linalg.det(Vh)
    Vh[:, -1] = Vh[:, -1] * d[..., None]

    # Calculate the optimal rotation matrix
    return U @ Vh


def kabsch_align(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align two point clouds using the Kabsch algorithm.

    Finds the optimal rotation to minimize RMSD between the point clouds.
    Both point clouds are centered at the origin before alignment.

    Args:
        x: Source point cloud of shape (..., N, 3).
        y: Target point cloud of shape (..., N, 3).
        weight: Optional weights for each point, shape (..., N).

    Returns:
        Tuple of (aligned_x, centered_y) where aligned_x is the rotated
        and centered source cloud, and centered_y is the centered target.

    Example:
        >>> source = torch.randn(100, 3)
        >>> target = torch.randn(100, 3)
        >>> aligned, centered_target = kabsch_align(source, target)
    """
    # Center the point clouds
    x = x - x.mean(-2, keepdim=True)
    y = y - y.mean(-2, keepdim=True)

    # Get the optimal rotation matrix
    R = get_kabsch_rotation_matrix(x, y, weight)

    # Apply the rotation to the source cloud
    return x @ R, y


class RMSD(nn.Module):
    """Loss function based on RMSD after Kabsch alignment.

    Computes the mean square deviation between two point clouds after
    applying the Kabsch algorithm to optimally align them.

    Args:
        weight: Optional weights for alignment, shape (N,).

    Example:
        >>> loss_fn = RMSD()
        >>> pred = model(input)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self: RMSD,
        weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.weight: torch.Tensor | None
        if weight is not None:
            self.register_buffer('weight', weight)
        else:
            self.weight = None

    def forward(
        self: RMSD,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute RMSD loss between input and target point clouds.

        Args:
            input: Predicted point cloud of shape (..., N, 3).
            target: Target point cloud of shape (..., N, 3).

        Returns:
            Mean square deviation after optimal alignment.
        """
        # Align the point clouds using the Kabsch algorithm
        input, target = kabsch_align(input, target, self.weight)
        # Calculate the mean square deviation
        return ((input - target) ** 2).mean()


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
