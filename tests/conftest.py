"""Shared fixtures for lr_geom tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch

import lr_geom as lg


# Conditional sphericart import
try:
    import sphericart
    HAS_SPHERICART = True
except ImportError:
    HAS_SPHERICART = False


# ============================================================================
# TEST CONSTANTS
# ============================================================================

RTOL = 1e-3  # Relative tolerance for float comparisons
ATOL = 1e-4  # Absolute tolerance for float comparisons


# ============================================================================
# SKIP MARKERS
# ============================================================================

requires_sphericart = pytest.mark.skipif(not HAS_SPHERICART, reason="sphericart not installed")


# ============================================================================
# REPRESENTATION FIXTURES
# ============================================================================

@pytest.fixture
def simple_repr():
    """Simple representation with lvals=[0, 1] for basic tests."""
    return lg.Repr(lvals=[0, 1], mult=4)  # dim=4 (1+3)


@pytest.fixture
def standard_repr():
    """Standard representation with lvals=[0, 1, 2], mult=8."""
    return lg.Repr(lvals=[0, 1, 2], mult=8)  # dim=9 (1+3+5)


@pytest.fixture
def hidden_repr(standard_repr):
    """Hidden representation with 4x multiplicity for transition layers."""
    return lg.Repr(lvals=standard_repr.lvals, mult=standard_repr.mult * 4)


@pytest.fixture
def product_repr(simple_repr):
    """Product representation for convolution tests."""
    return lg.ProductRepr(simple_repr, simple_repr)


# ============================================================================
# SAMPLE DATA FIXTURES
# ============================================================================

@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 32


@pytest.fixture
def num_nodes():
    """Number of nodes for graph tests."""
    return 50


@pytest.fixture
def num_edges():
    """Number of edges for graph tests."""
    return 200


@pytest.fixture
def edge_dim():
    """Dimension of edge features."""
    return 16


@pytest.fixture
def hidden_dim():
    """Hidden dimension for radial networks."""
    return 32


@pytest.fixture
def sample_input(batch_size, standard_repr):
    """Sample spherical tensor input: (batch, mult, dim)."""
    return torch.randn(batch_size, standard_repr.mult, standard_repr.dim())


@pytest.fixture
def sample_input_simple(batch_size, simple_repr):
    """Sample input for simple representation."""
    return torch.randn(batch_size, simple_repr.mult, simple_repr.dim())


@pytest.fixture
def edge_features(num_edges, edge_dim):
    """Sample edge features: (E, edge_dim)."""
    return torch.randn(num_edges, edge_dim)


@pytest.fixture
def edge_vectors(num_edges):
    """Sample 3D edge displacement vectors: (E, 3)."""
    return torch.randn(num_edges, 3)


@pytest.fixture
def node_features(num_nodes, simple_repr):
    """Sample node features for graph layers: (N, mult, dim)."""
    return torch.randn(num_nodes, simple_repr.mult, simple_repr.dim())


@pytest.fixture
def coordinates(num_nodes):
    """Sample 3D coordinates: (N, 3)."""
    return torch.randn(num_nodes, 3)


# ============================================================================
# ROTATION FIXTURES
# ============================================================================

@pytest.fixture(params=[
    ([1., 0., 0.], np.pi / 4),      # 45 degrees around x
    ([0., 1., 0.], np.pi / 3),      # 60 degrees around y
    ([0., 0., 1.], np.pi / 6),      # 30 degrees around z
    ([1., 1., 1.], np.pi / 2),      # 90 degrees around [1,1,1]
    ([1., 2., 3.], np.pi / 5),      # Arbitrary axis
], ids=['x-45deg', 'y-60deg', 'z-30deg', 'diag-90deg', 'arbitrary'])
def rotation_params(request):
    """Parametrized rotation axis and angle."""
    axis, angle = request.param
    axis = torch.tensor(axis, dtype=torch.float32)
    axis = axis / axis.norm()  # Normalize
    angle = torch.tensor(angle, dtype=torch.float32)
    return axis, angle


# ============================================================================
# ROTATION HELPER FUNCTIONS
# ============================================================================

def get_wigner_d_matrix(repr: lg.Repr, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Compute Wigner D-matrix for rotating spherical tensors.

    Uses Repr.rot() which computes D via matrix exponential of Lie algebra.

    Args:
        repr: The representation defining the tensor structure.
        axis: Rotation axis (normalized).
        angle: Rotation angle in radians.

    Returns:
        Wigner D-matrix of shape (dim, dim).
    """
    D = repr.rot(axis, angle)
    # Handle potential batch dimension
    if D.dim() == 3:
        D = D.squeeze(0)
    return D


def get_3d_rotation_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Compute 3x3 rotation matrix using Rodrigues' formula.

    R = I + sin(theta)*K + (1-cos(theta))*K^2
    where K is the skew-symmetric matrix of the axis.

    Args:
        axis: Rotation axis (will be normalized).
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    axis = axis / axis.norm()
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=axis.dtype)
    R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    return R


# ============================================================================
# K-NN GRAPH FIXTURES
# ============================================================================

@pytest.fixture
def k_neighbors():
    """Number of neighbors for k-NN attention."""
    return 16


@pytest.fixture
def neighbor_idx(num_nodes, k_neighbors):
    """Create k-NN neighbor indices for testing."""
    return lg.build_knn_graph(torch.randn(num_nodes, 3), k_neighbors)
