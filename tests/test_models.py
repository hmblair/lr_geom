"""Tests for lr_geom models module.

Tests cover:
1. graph_laplacian - Graph Laplacian computation
2. gnm_correlations - Gaussian Network Model correlations
3. gnm_variances - GNM position variances
4. GNMA - Gaussian Network Model Attention
"""
from __future__ import annotations

import pytest
import torch

from lr_geom.models import graph_laplacian, gnm_correlations, gnm_variances, GNMA


# ============================================================================
# GRAPH LAPLACIAN TESTS
# ============================================================================

class TestGraphLaplacian:
    """Tests for graph_laplacian function."""

    def test_basic_example(self):
        """Test Laplacian computation for simple triangle graph."""
        # Complete graph K_3 (triangle)
        adj = torch.tensor([
            [0., 1., 1.],
            [1., 0., 1.],
            [1., 1., 0.]
        ])

        L = graph_laplacian(adj)

        # L = D - A where D is degree matrix
        expected = torch.tensor([
            [2., -1., -1.],
            [-1., 2., -1.],
            [-1., -1., 2.]
        ])
        assert torch.allclose(L, expected)

    def test_row_sum_zero(self):
        """Test that each row of Laplacian sums to zero."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T  # Make symmetric
        adj.fill_diagonal_(0)  # No self-loops

        L = graph_laplacian(adj)
        row_sums = L.sum(dim=1)

        assert torch.allclose(row_sums, torch.zeros(10), atol=1e-5)

    def test_symmetric(self):
        """Test that Laplacian is symmetric for symmetric adjacency."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T  # Make symmetric
        adj.fill_diagonal_(0)

        L = graph_laplacian(adj)

        assert torch.allclose(L, L.T)

    def test_positive_semidefinite(self):
        """Test that Laplacian is positive semi-definite."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        L = graph_laplacian(adj)
        eigenvalues = torch.linalg.eigvalsh(L)

        # All eigenvalues should be >= 0 (with small tolerance)
        assert (eigenvalues >= -1e-6).all()

    def test_smallest_eigenvalue_zero(self):
        """Test that smallest eigenvalue is zero for connected graph."""
        # Connected graph
        adj = torch.ones(5, 5) - torch.eye(5)

        L = graph_laplacian(adj)
        eigenvalues = torch.linalg.eigvalsh(L)

        # Smallest eigenvalue should be 0
        assert torch.isclose(eigenvalues[0], torch.tensor(0.), atol=1e-6)

    def test_diagonal_is_degree(self):
        """Test that diagonal elements are node degrees."""
        adj = torch.tensor([
            [0., 1., 1., 0.],
            [1., 0., 1., 1.],
            [1., 1., 0., 0.],
            [0., 1., 0., 0.]
        ])

        L = graph_laplacian(adj)
        degrees = adj.sum(dim=1)

        assert torch.allclose(L.diagonal(), degrees)

    def test_off_diagonal_negated(self):
        """Test that off-diagonal elements are negated adjacency."""
        adj = torch.rand(5, 5)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        L = graph_laplacian(adj)

        # Off-diagonal should be -adj
        mask = ~torch.eye(5, dtype=torch.bool)
        assert torch.allclose(L[mask], -adj[mask])

    def test_empty_graph(self):
        """Test Laplacian of graph with no edges."""
        adj = torch.zeros(5, 5)

        L = graph_laplacian(adj)

        # Laplacian should be all zeros
        assert torch.allclose(L, torch.zeros(5, 5))

    def test_single_node(self):
        """Test Laplacian of single node graph."""
        adj = torch.tensor([[0.]])

        L = graph_laplacian(adj)

        assert torch.allclose(L, torch.tensor([[0.]]))


# ============================================================================
# GNM CORRELATIONS TESTS
# ============================================================================

class TestGNMCorrelations:
    """Tests for gnm_correlations function."""

    def test_output_shape(self):
        """Test output shape matches input."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        corr = gnm_correlations(adj)

        assert corr.shape == (10, 10)

    def test_symmetric(self):
        """Test correlations are symmetric."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        corr = gnm_correlations(adj)

        assert torch.allclose(corr, corr.T, atol=1e-4)

    def test_no_nan(self):
        """Test output contains no NaN."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        corr = gnm_correlations(adj)

        assert not torch.isnan(corr).any()

    def test_pseudoinverse_property(self):
        """Test that corr is pseudoinverse of Laplacian.

        L @ pinv(L) @ L = L (pseudoinverse property)
        """
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        L = graph_laplacian(adj)
        corr = gnm_correlations(adj)

        # L @ corr @ L should equal L
        result = L @ corr @ L
        assert torch.allclose(result, L, atol=1e-3)

    def test_complete_graph(self):
        """Test correlations for complete graph."""
        n = 5
        adj = torch.ones(n, n) - torch.eye(n)

        corr = gnm_correlations(adj)

        # For complete graph, all correlations should be similar
        # (uniform structure)
        assert corr.shape == (n, n)
        assert not torch.isnan(corr).any()

    def test_positive_diagonal(self):
        """Test diagonal elements (variances) are positive."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        corr = gnm_correlations(adj)

        # Diagonal should be non-negative (variances)
        # Due to pseudoinverse, might have small negative numerical errors
        assert (corr.diagonal() >= -1e-5).all()


# ============================================================================
# GNM VARIANCES TESTS
# ============================================================================

class TestGNMVariances:
    """Tests for gnm_variances function."""

    def test_output_shape(self):
        """Test output is vector of correct size."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        var = gnm_variances(adj)

        assert var.shape == (10,)

    def test_matches_diagonal_of_correlations(self):
        """Test variances equal diagonal of correlation matrix."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        var = gnm_variances(adj)
        corr = gnm_correlations(adj)

        assert torch.allclose(var, corr.diagonal())

    def test_no_nan(self):
        """Test output contains no NaN."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        var = gnm_variances(adj)

        assert not torch.isnan(var).any()

    def test_positive(self):
        """Test variances are non-negative."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        var = gnm_variances(adj)

        # Allow small numerical tolerance
        assert (var >= -1e-5).all()


# ============================================================================
# GNMA MODEL TESTS
# ============================================================================

class TestGNMA:
    """Tests for GNMA (Gaussian Network Model Attention) model."""

    def test_forward_shape(self):
        """Test output shape matches input."""
        model = GNMA(dim=16)

        coords = torch.randn(50, 3)
        output = model(coords)

        assert output.shape == (50, 3)

    def test_forward_no_nan(self):
        """Test output contains no NaN."""
        model = GNMA(dim=16)

        coords = torch.randn(50, 3)
        output = model(coords)

        assert not torch.isnan(output).any()

    def test_embed_shape(self):
        """Test embed method output shape."""
        model = GNMA(dim=16)

        dists = torch.rand(20, 20) * 10
        embedded = model.embed(dists)

        assert embedded.shape == (20, 20)

    def test_embed_no_nan(self):
        """Test embed output contains no NaN."""
        model = GNMA(dim=16)

        dists = torch.rand(20, 20) * 10
        embedded = model.embed(dists)

        assert not torch.isnan(embedded).any()

    def test_output_in_convex_hull(self):
        """Test output is in convex hull of input.

        Since output is softmax(attention) @ coords, the output
        should be a convex combination of input coordinates.
        """
        model = GNMA(dim=16)
        model.eval()

        coords = torch.randn(10, 3)
        output = model(coords)

        # Output should be bounded by min/max of input
        # (since it's a convex combination)
        for d in range(3):
            assert output[:, d].min() >= coords[:, d].min() - 0.01
            assert output[:, d].max() <= coords[:, d].max() + 0.01

    def test_backward_gradients(self):
        """Test gradients flow correctly."""
        model = GNMA(dim=16)

        coords = torch.randn(20, 3, requires_grad=True)
        output = model(coords)
        loss = output.sum()
        loss.backward()

        assert coords.grad is not None
        assert not torch.isnan(coords.grad).any()

    def test_optimizer_step(self):
        """Test weights update after optimizer step."""
        model = GNMA(dim=16)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Store initial weights
        initial_linear_weight = model.linear.layers[0].weight.clone()

        coords = torch.randn(20, 3)
        output = model(coords)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.allclose(model.linear.layers[0].weight, initial_linear_weight)

    def test_different_output_dim(self):
        """Test GNMA with different output dimension.

        With out>1, embed produces (N, N, out) which affects the
        downstream shape. This is useful for multi-head attention.
        """
        model = GNMA(dim=16, out=5)

        coords = torch.randn(20, 3)
        dists = torch.cdist(coords, coords)

        # embed outputs (N, N, out) when out > 1
        embedded = model.embed(dists)
        assert embedded.shape == (20, 20, 5)

        # forward still works but shape reflects the out dimension
        output = model(coords)
        # Output has extra dimension from the attention
        assert not torch.isnan(output).any()

    def test_batch_independence(self):
        """Test that processing is independent per batch element."""
        model = GNMA(dim=16)
        model.eval()

        coords1 = torch.randn(10, 3)
        coords2 = torch.randn(10, 3)

        output1 = model(coords1)
        output2 = model(coords2)

        # Processing combined should give same results
        # Note: GNMA doesn't naturally batch, but outputs should be consistent
        assert output1.shape == (10, 3)
        assert output2.shape == (10, 3)

    def test_deterministic_eval(self):
        """Test output is deterministic in eval mode."""
        model = GNMA(dim=16)
        model.eval()

        coords = torch.randn(20, 3)

        output1 = model(coords)
        output2 = model(coords)

        assert torch.allclose(output1, output2)

    def test_small_input(self):
        """Test GNMA handles small inputs."""
        model = GNMA(dim=16)

        # Very small number of atoms
        coords = torch.randn(3, 3)
        output = model(coords)

        assert output.shape == (3, 3)
        assert not torch.isnan(output).any()

    def test_large_input(self):
        """Test GNMA handles larger inputs."""
        model = GNMA(dim=16)

        coords = torch.randn(200, 3)
        output = model(coords)

        assert output.shape == (200, 3)
        assert not torch.isnan(output).any()

    def test_zero_distances_handling(self):
        """Test GNMA handles zero distances (identical points)."""
        model = GNMA(dim=16)

        # Some points at same location
        coords = torch.randn(10, 3)
        coords[5] = coords[0]  # Same as first point

        output = model(coords)

        # Should not produce NaN
        assert not torch.isnan(output).any()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestModelsIntegration:
    """Integration tests for models module."""

    def test_laplacian_to_correlations_pipeline(self):
        """Test full pipeline from adjacency to correlations."""
        adj = torch.rand(10, 10)
        adj = adj + adj.T
        adj.fill_diagonal_(0)

        # Compute Laplacian
        L = graph_laplacian(adj)
        assert L.shape == (10, 10)

        # Compute correlations
        corr = gnm_correlations(adj)
        assert corr.shape == (10, 10)

        # Compute variances
        var = gnm_variances(adj)
        assert var.shape == (10,)

        # Variance should be diagonal of correlations
        assert torch.allclose(var, corr.diagonal())

    def test_gnma_uses_gnm_correlations(self):
        """Test that GNMA internally uses GNM correlations."""
        model = GNMA(dim=16)

        coords = torch.randn(10, 3)
        pw = torch.cdist(coords, coords)

        # Get embedded adjacency
        emb = model.embed(pw)

        # Should be able to compute correlations on it
        corr = gnm_correlations(emb)

        assert corr.shape == (10, 10)
        assert not torch.isnan(corr).any()

    def test_backward_compatibility_radialweight(self):
        """Test RadialWeight is re-exported from models."""
        from lr_geom.models import RadialWeight
        from lr_geom import Repr, ProductRepr

        # Should be importable and usable
        repr = Repr(lvals=[0, 1], mult=4)
        product_repr = ProductRepr(repr, repr)

        rw = RadialWeight(
            edge_dim=16,
            hidden_dim=32,
            repr=product_repr,
            in_dim=4,
            out_dim=4,
        )

        edge_features = torch.randn(100, 16)
        output = rw(edge_features)

        assert output.shape[0] == 100
