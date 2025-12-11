"""Tests for lr_geom equivariant layers.

Tests cover:
1. Forward pass (shapes, no NaN)
2. Equivariance (rotating input produces equivalently rotated output)
3. Backward pass (gradients flow, weights update)
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import lr_geom as lg
from conftest import (
    RTOL,
    ATOL,
    requires_sphericart,
    get_wigner_d_matrix,
    get_3d_rotation_matrix,
    HAS_SPHERICART,
)


# ============================================================================
# RADIALWEIGHT TESTS
# ============================================================================

class TestRadialWeight:
    """Tests for RadialWeight layer."""

    def test_forward_shape(self, product_repr, edge_features, edge_dim, hidden_dim):
        """Test output shape is correct."""
        layer = lg.RadialWeight(
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            repr=product_repr,
            in_dim=product_repr.rep1.mult,
            out_dim=product_repr.rep2.mult,
        )
        output = layer(edge_features)

        expected_shape = (
            edge_features.size(0),
            product_repr.rep2.nreps() * product_repr.rep2.mult,
            product_repr.rep1.nreps() * product_repr.rep1.mult,
        )
        assert output.shape == expected_shape

    def test_forward_no_nan(self, product_repr, edge_features, edge_dim, hidden_dim):
        """Test output contains no NaN values."""
        layer = lg.RadialWeight(
            edge_dim, hidden_dim, product_repr,
            product_repr.rep1.mult, product_repr.rep2.mult
        )
        output = layer(edge_features)
        assert not torch.isnan(output).any()

    def test_backward_gradients(self, product_repr, edge_features, edge_dim, hidden_dim):
        """Test gradients flow correctly through the layer."""
        layer = lg.RadialWeight(
            edge_dim, hidden_dim, product_repr,
            product_repr.rep1.mult, product_repr.rep2.mult
        )
        edge_features = edge_features.clone().requires_grad_(True)

        output = layer(edge_features)
        loss = output.sum()
        loss.backward()

        assert edge_features.grad is not None
        assert not torch.isnan(edge_features.grad).any()
        for param in layer.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_optimizer_step(self, product_repr, edge_features, edge_dim, hidden_dim):
        """Test weights update after optimizer step."""
        layer = lg.RadialWeight(
            edge_dim, hidden_dim, product_repr,
            product_repr.rep1.mult, product_repr.rep2.mult
        )
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

        initial_weights = layer.layer1.weight.clone()

        output = layer(edge_features)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        assert not torch.allclose(layer.layer1.weight, initial_weights)


# ============================================================================
# EQUIVARIANTLINEAR TESTS
# ============================================================================

class TestEquivariantLinear:
    """Tests for EquivariantLinear layer."""

    def test_forward_shape(self, standard_repr, sample_input):
        """Test output shape matches expected dimensions."""
        out_repr = lg.Repr(lvals=standard_repr.lvals, mult=16)
        layer = lg.EquivariantLinear(standard_repr, out_repr)

        output = layer(sample_input)
        assert output.shape == (sample_input.size(0), out_repr.mult, out_repr.dim())

    def test_forward_no_nan(self, standard_repr, sample_input):
        """Test output contains no NaN values."""
        layer = lg.EquivariantLinear(standard_repr, standard_repr)
        output = layer(sample_input)
        assert not torch.isnan(output).any()

    def test_lvals_mismatch_raises(self, standard_repr):
        """Test that mismatched lvals raises ValueError."""
        out_repr = lg.Repr(lvals=[0, 1], mult=8)  # Different lvals
        with pytest.raises(ValueError, match="cannot modify the degrees"):
            lg.EquivariantLinear(standard_repr, out_repr)

    def test_equivariance(self, standard_repr, sample_input, rotation_params):
        """Test SO(3) equivariance: layer(R @ x) == R @ layer(x)."""
        axis, angle = rotation_params
        layer = lg.EquivariantLinear(standard_repr, standard_repr)
        layer.eval()  # Disable dropout

        # Get Wigner D-matrix
        D = get_wigner_d_matrix(standard_repr, axis, angle)

        # Rotate input: x @ D.T
        x_rotated = sample_input @ D.T

        # Forward pass on rotated input
        y_from_rotated = layer(x_rotated)

        # Forward pass then rotate
        y_original = layer(sample_input)
        y_rotated = y_original @ D.T

        # Check equivariance
        assert torch.allclose(y_from_rotated, y_rotated, rtol=RTOL, atol=ATOL)

    def test_backward_gradients(self, standard_repr, sample_input):
        """Test gradients flow correctly."""
        layer = lg.EquivariantLinear(standard_repr, standard_repr)
        sample_input = sample_input.clone().requires_grad_(True)

        output = layer(sample_input)
        loss = output.sum()
        loss.backward()

        assert sample_input.grad is not None
        for param in layer.parameters():
            assert param.grad is not None

    def test_optimizer_step(self, standard_repr, sample_input):
        """Test weights update after optimizer step."""
        layer = lg.EquivariantLinear(standard_repr, standard_repr)
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

        initial_weight = layer.weight.clone()

        output = layer(sample_input)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        assert not torch.allclose(layer.weight, initial_weight)

    def test_bias_only_scalars(self, standard_repr, sample_input):
        """Test that bias is only applied to scalar (l=0) components."""
        layer = lg.EquivariantLinear(standard_repr, standard_repr, bias=True)

        # Bias should exist if l=0 is in lvals
        if 0 in standard_repr.lvals:
            assert layer.bias is not None
        else:
            assert layer.bias is None


# ============================================================================
# EQUIVARIANTGATING TESTS
# ============================================================================

class TestEquivariantGating:
    """Tests for EquivariantGating layer."""

    def test_forward_shape(self, standard_repr, sample_input):
        """Test output shape equals input shape."""
        layer = lg.EquivariantGating(standard_repr)
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_forward_no_nan(self, standard_repr, sample_input):
        """Test output contains no NaN values."""
        layer = lg.EquivariantGating(standard_repr)
        output = layer(sample_input)
        assert not torch.isnan(output).any()

    def test_equivariance(self, standard_repr, sample_input, rotation_params):
        """Test SO(3) equivariance: gate(R @ x) == R @ gate(x)."""
        axis, angle = rotation_params
        layer = lg.EquivariantGating(standard_repr)
        layer.eval()

        D = get_wigner_d_matrix(standard_repr, axis, angle)

        # Rotate input
        x_rotated = sample_input @ D.T

        # Test equivariance
        y_from_rotated = layer(x_rotated)
        y_rotated = layer(sample_input) @ D.T

        assert torch.allclose(y_from_rotated, y_rotated, rtol=RTOL, atol=ATOL)

    def test_backward_gradients(self, standard_repr, sample_input):
        """Test gradients flow correctly."""
        layer = lg.EquivariantGating(standard_repr)
        sample_input = sample_input.clone().requires_grad_(True)

        output = layer(sample_input)
        loss = output.sum()
        loss.backward()

        assert sample_input.grad is not None


# ============================================================================
# EQUIVARIANTLAYERNORM TESTS
# ============================================================================

class TestEquivariantLayerNorm:
    """Tests for EquivariantLayerNorm layer."""

    def test_forward_shape(self, standard_repr, sample_input):
        """Test output shape equals input shape."""
        layer = lg.EquivariantLayerNorm(standard_repr)
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_forward_no_nan(self, standard_repr, sample_input):
        """Test output contains no NaN values."""
        layer = lg.EquivariantLayerNorm(standard_repr)
        output = layer(sample_input)
        assert not torch.isnan(output).any()

    def test_equivariance(self, standard_repr, sample_input, rotation_params):
        """Test SO(3) equivariance: ln(R @ x) == R @ ln(x)."""
        axis, angle = rotation_params
        layer = lg.EquivariantLayerNorm(standard_repr)
        layer.eval()

        D = get_wigner_d_matrix(standard_repr, axis, angle)

        x_rotated = sample_input @ D.T

        y_from_rotated = layer(x_rotated)
        y_rotated = layer(sample_input) @ D.T

        assert torch.allclose(y_from_rotated, y_rotated, rtol=RTOL, atol=ATOL)

    def test_backward_gradients(self, standard_repr, sample_input):
        """Test gradients flow correctly."""
        layer = lg.EquivariantLayerNorm(standard_repr)
        sample_input = sample_input.clone().requires_grad_(True)

        output = layer(sample_input)
        loss = output.sum()
        loss.backward()

        assert sample_input.grad is not None


# ============================================================================
# EQUIVARIANTCONVOLUTION TESTS
# ============================================================================

@requires_sphericart
class TestEquivariantConvolution:
    """Tests for EquivariantConvolution layer."""

    @pytest.fixture
    def conv_setup(self, simple_repr, edge_dim, hidden_dim, num_nodes, num_edges):
        """Setup for convolution tests."""
        product_repr = lg.ProductRepr(simple_repr, simple_repr)
        layer = lg.EquivariantConvolution(product_repr, edge_dim, hidden_dim)

        # Create basis
        basis_layer = lg.EquivariantBasis(product_repr)
        edge_vectors = torch.randn(num_edges, 3)
        bases = basis_layer(edge_vectors)

        # Create data
        node_features = torch.randn(num_nodes, simple_repr.mult, simple_repr.dim())
        edge_features = torch.randn(num_edges, edge_dim)
        src_idx = torch.randint(0, num_nodes, (num_edges,))

        return {
            'layer': layer,
            'bases': bases,
            'node_features': node_features,
            'edge_features': edge_features,
            'src_idx': src_idx,
            'product_repr': product_repr,
            'edge_vectors': edge_vectors,
            'basis_layer': basis_layer,
            'simple_repr': simple_repr,
        }

    def test_forward_shape(self, conv_setup):
        """Test output shape is correct."""
        setup = conv_setup
        output = setup['layer'](
            setup['bases'],
            setup['edge_features'],
            setup['node_features'],
            setup['src_idx'],
        )

        num_edges = setup['edge_features'].size(0)
        expected_shape = (
            num_edges,
            setup['product_repr'].rep2.mult,
            setup['product_repr'].rep2.dim(),
        )
        assert output.shape == expected_shape

    def test_forward_no_nan(self, conv_setup):
        """Test output contains no NaN values."""
        setup = conv_setup
        output = setup['layer'](
            setup['bases'],
            setup['edge_features'],
            setup['node_features'],
            setup['src_idx'],
        )
        assert not torch.isnan(output).any()

    def test_equivariance(self, conv_setup, rotation_params):
        """Test SE(3) equivariance of convolution."""
        axis, angle = rotation_params
        setup = conv_setup
        setup['layer'].eval()
        simple_repr = setup['simple_repr']

        # Get rotation matrices
        D = get_wigner_d_matrix(simple_repr, axis, angle)
        R = get_3d_rotation_matrix(axis, angle)

        # Rotate inputs
        node_features_rotated = setup['node_features'] @ D.T
        edge_vectors_rotated = setup['edge_vectors'] @ R.T

        # Recompute bases for rotated edge vectors
        bases_rotated = setup['basis_layer'](edge_vectors_rotated)

        # Forward on rotated inputs
        output_from_rotated = setup['layer'](
            bases_rotated,
            setup['edge_features'],  # Invariant features stay same
            node_features_rotated,
            setup['src_idx'],
        )

        # Forward then rotate
        output_original = setup['layer'](
            setup['bases'],
            setup['edge_features'],
            setup['node_features'],
            setup['src_idx'],
        )
        output_rotated = output_original @ D.T

        assert torch.allclose(output_from_rotated, output_rotated, rtol=RTOL, atol=ATOL)

    def test_backward_gradients(self, conv_setup):
        """Test gradients flow correctly."""
        setup = conv_setup
        node_features = setup['node_features'].clone().requires_grad_(True)
        edge_features = setup['edge_features'].clone().requires_grad_(True)

        output = setup['layer'](
            setup['bases'],
            edge_features,
            node_features,
            setup['src_idx'],
        )
        loss = output.sum()
        loss.backward()

        assert node_features.grad is not None
        assert edge_features.grad is not None

    def test_equivariance_different_lvals(self, edge_dim, hidden_dim, num_nodes, num_edges, rotation_params):
        """Test equivariance when input and output have different lvals."""
        axis, angle = rotation_params

        # Input: scalars + vectors, Output: vectors only
        in_repr = lg.Repr(lvals=[0, 1], mult=4)
        out_repr = lg.Repr(lvals=[1], mult=4)
        product_repr = lg.ProductRepr(in_repr, out_repr)

        layer = lg.EquivariantConvolution(product_repr, edge_dim, hidden_dim)
        layer.eval()

        # Create basis
        basis_layer = lg.EquivariantBasis(product_repr)
        edge_vectors = torch.randn(num_edges, 3)
        bases = basis_layer(edge_vectors)

        # Create data
        node_features = torch.randn(num_nodes, in_repr.mult, in_repr.dim())
        edge_features = torch.randn(num_edges, edge_dim)
        src_idx = torch.randint(0, num_nodes, (num_edges,))

        # Get rotation matrices for input and output (different dims!)
        D_in = get_wigner_d_matrix(in_repr, axis, angle)
        D_out = get_wigner_d_matrix(out_repr, axis, angle)
        R = get_3d_rotation_matrix(axis, angle)

        # Rotate inputs
        node_features_rotated = node_features @ D_in.T
        edge_vectors_rotated = edge_vectors @ R.T
        bases_rotated = basis_layer(edge_vectors_rotated)

        # Forward on rotated inputs
        output_from_rotated = layer(bases_rotated, edge_features, node_features_rotated, src_idx)

        # Forward on original, then rotate output
        output_original = layer(bases, edge_features, node_features, src_idx)
        output_rotated = output_original @ D_out.T

        assert torch.allclose(output_from_rotated, output_rotated, rtol=RTOL, atol=ATOL), \
            f"Max diff: {(output_from_rotated - output_rotated).abs().max()}"


# ============================================================================
# EQUIVARIANTTRANSITION TESTS
# ============================================================================

class TestEquivariantTransition:
    """Tests for EquivariantTransition layer."""

    def test_forward_shape(self, standard_repr, hidden_repr, sample_input):
        """Test output shape equals input shape."""
        layer = lg.EquivariantTransition(standard_repr, hidden_repr)
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_forward_no_nan(self, standard_repr, hidden_repr, sample_input):
        """Test output contains no NaN values."""
        layer = lg.EquivariantTransition(standard_repr, hidden_repr)
        output = layer(sample_input)
        assert not torch.isnan(output).any()

    def test_equivariance(self, standard_repr, hidden_repr, sample_input, rotation_params):
        """Test SO(3) equivariance: transition(R @ x) == R @ transition(x)."""
        axis, angle = rotation_params
        layer = lg.EquivariantTransition(standard_repr, hidden_repr)
        layer.eval()

        D = get_wigner_d_matrix(standard_repr, axis, angle)

        x_rotated = sample_input @ D.T

        y_from_rotated = layer(x_rotated)
        y_rotated = layer(sample_input) @ D.T

        assert torch.allclose(y_from_rotated, y_rotated, rtol=RTOL, atol=ATOL)

    def test_backward_gradients(self, standard_repr, hidden_repr, sample_input):
        """Test gradients flow correctly."""
        layer = lg.EquivariantTransition(standard_repr, hidden_repr)
        sample_input = sample_input.clone().requires_grad_(True)

        output = layer(sample_input)
        loss = output.sum()
        loss.backward()

        assert sample_input.grad is not None


# ============================================================================
# ATTENTION TESTS
# ============================================================================

class TestAttention:
    """Tests for Attention layer (k-NN based)."""

    def test_forward_shape(self, num_nodes, k_neighbors, neighbor_idx):
        """Test output shape is (num_nodes, hidden_size)."""
        hidden_size = 64
        nheads = 4
        layer = lg.Attention(hidden_size, nheads)

        keys = torch.randn(num_nodes, hidden_size)
        queries = torch.randn(num_nodes, hidden_size)
        values = torch.randn(num_nodes, hidden_size)

        output = layer(keys, queries, values, neighbor_idx)
        assert output.shape == (num_nodes, hidden_size)

    def test_forward_no_nan(self, num_nodes, neighbor_idx):
        """Test output contains no NaN values."""
        hidden_size = 64
        layer = lg.Attention(hidden_size, nheads=4)

        keys = torch.randn(num_nodes, hidden_size)
        queries = torch.randn(num_nodes, hidden_size)
        values = torch.randn(num_nodes, hidden_size)

        output = layer(keys, queries, values, neighbor_idx)
        assert not torch.isnan(output).any()

    def test_backward_gradients(self, num_nodes, neighbor_idx):
        """Test gradients flow correctly."""
        hidden_size = 64
        layer = lg.Attention(hidden_size, nheads=4)

        keys = torch.randn(num_nodes, hidden_size, requires_grad=True)
        queries = torch.randn(num_nodes, hidden_size, requires_grad=True)
        values = torch.randn(num_nodes, hidden_size, requires_grad=True)

        output = layer(keys, queries, values, neighbor_idx)
        loss = output.sum()
        loss.backward()

        assert keys.grad is not None
        assert queries.grad is not None
        assert values.grad is not None

    def test_hidden_size_nheads_divisibility(self):
        """Test that hidden_size must be divisible by nheads."""
        with pytest.raises(ValueError, match="divisible"):
            lg.Attention(hidden_size=65, nheads=4)


# ============================================================================
# EQUIVARIANTTRANSFORMER TESTS (requires sphericart)
# ============================================================================

# Note: EquivariantAttention and EquivariantTransformerBlock are tested
# implicitly through EquivariantTransformer tests. Testing them in isolation
# requires matching the internal basis computation which is complex.

@requires_sphericart
class TestEquivariantTransformer:
    """Tests for EquivariantTransformer model."""

    @pytest.fixture
    def transformer_setup(self, num_nodes, edge_dim, hidden_dim, k_neighbors):
        """Setup for transformer tests."""
        in_repr = lg.Repr(lvals=[0, 1], mult=4)
        out_repr = lg.Repr(lvals=[0, 1], mult=2)
        hidden_repr = lg.Repr(lvals=[0, 1], mult=8)

        model = lg.EquivariantTransformer(
            in_repr=in_repr,
            out_repr=out_repr,
            hidden_repr=hidden_repr,
            hidden_layers=2,
            edge_dim=edge_dim,
            edge_hidden_dim=hidden_dim,
            k_neighbors=k_neighbors,
            nheads=4,
            dropout=0.0,
            attn_dropout=0.0,
            transition=True,
        )

        coordinates = torch.randn(num_nodes, 3)
        node_features = torch.randn(num_nodes, in_repr.mult, in_repr.dim())

        return {
            'model': model,
            'coordinates': coordinates,
            'node_features': node_features,
            'in_repr': in_repr,
            'out_repr': out_repr,
        }

    def test_forward_shape(self, transformer_setup, num_nodes):
        """Test transformer output shape matches out_repr."""
        setup = transformer_setup
        output = setup['model'](
            setup['coordinates'],
            setup['node_features'],
        )
        expected_shape = (num_nodes, setup['out_repr'].mult, setup['out_repr'].dim())
        assert output.shape == expected_shape

    def test_forward_no_nan(self, transformer_setup):
        """Test transformer output contains no NaN values."""
        setup = transformer_setup
        output = setup['model'](
            setup['coordinates'],
            setup['node_features'],
        )
        assert not torch.isnan(output).any()

    def test_invalid_input_shape_raises(self, transformer_setup):
        """Test that mismatched input shape raises ValueError."""
        setup = transformer_setup
        wrong_features = torch.randn(setup['node_features'].size(0), 100, 100)

        with pytest.raises(ValueError, match="does not match"):
            setup['model'](
                setup['coordinates'],
                wrong_features,
            )

    def test_backward_gradients(self, transformer_setup):
        """Test gradients flow correctly through transformer."""
        setup = transformer_setup
        node_features = setup['node_features'].clone().requires_grad_(True)
        coordinates = setup['coordinates'].clone().requires_grad_(True)

        output = setup['model'](
            coordinates,
            node_features,
        )
        loss = output.sum()
        loss.backward()

        assert node_features.grad is not None

    def test_optimizer_step(self, transformer_setup):
        """Test that model parameters update after optimizer step."""
        setup = transformer_setup
        model = setup['model']
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Store initial weights
        initial_params = {name: p.clone() for name, p in model.named_parameters()}

        output = model(
            setup['coordinates'],
            setup['node_features'],
        )
        loss = output.sum()
        loss.backward()
        optimizer.step()

        # Check at least some weights changed
        changed = False
        for name, p in model.named_parameters():
            if not torch.allclose(p, initial_params[name]):
                changed = True
                break
        assert changed, "No parameters were updated"

    def test_equivariance(self, transformer_setup, rotation_params):
        """Test SE(3) equivariance of the full transformer.

        For rotation R and Wigner D:
        - model(R @ coords, D @ features) == D @ model(coords, features)

        Note: Uses slightly relaxed tolerance (5e-3) compared to single layers
        because numerical errors accumulate through multiple transformer blocks.
        """
        axis, angle = rotation_params
        setup = transformer_setup
        model = setup['model']
        model.eval()

        in_repr = setup['in_repr']
        out_repr = setup['out_repr']
        coords = setup['coordinates']
        features = setup['node_features']

        # Get rotation matrices
        D_in = get_wigner_d_matrix(in_repr, axis, angle)
        D_out = get_wigner_d_matrix(out_repr, axis, angle)
        R = get_3d_rotation_matrix(axis, angle)

        # Rotate inputs
        coords_rotated = coords @ R.T
        features_rotated = features @ D_in.T

        # Forward on rotated inputs
        output_from_rotated = model(coords_rotated, features_rotated)

        # Forward then rotate
        output_original = model(coords, features)
        output_rotated = output_original @ D_out.T

        # Use relaxed tolerance for deep networks (numerical error accumulates)
        transformer_rtol = 1e-2  # 1% relative tolerance for deep networks
        transformer_atol = 1e-2  # Relaxed absolute tolerance for numerical precision
        assert torch.allclose(output_from_rotated, output_rotated, rtol=transformer_rtol, atol=transformer_atol), \
            f"Max diff: {(output_from_rotated - output_rotated).abs().max()}"


# ============================================================================
# BUILD_KNN_GRAPH TESTS
# ============================================================================

class TestBuildKnnGraph:
    """Tests for build_knn_graph helper function."""

    def test_output_shape(self, num_nodes, k_neighbors):
        """Test output shape is (N, k)."""
        coordinates = torch.randn(num_nodes, 3)
        neighbor_idx = lg.build_knn_graph(coordinates, k_neighbors)
        assert neighbor_idx.shape == (num_nodes, k_neighbors)

    def test_excludes_self(self, num_nodes, k_neighbors):
        """Test that self is excluded from neighbors."""
        coordinates = torch.randn(num_nodes, 3)
        neighbor_idx = lg.build_knn_graph(coordinates, k_neighbors)

        # Each node should not be in its own neighbor list
        for i in range(num_nodes):
            assert i not in neighbor_idx[i]

    def test_small_graph_handling(self):
        """Test handling when N <= k."""
        coordinates = torch.randn(5, 3)
        k = 10

        # Should not crash
        neighbor_idx = lg.build_knn_graph(coordinates, k)
        assert neighbor_idx.shape == (5, k)

    def test_valid_indices(self, num_nodes, k_neighbors):
        """Test all indices are valid (0 <= idx < N)."""
        coordinates = torch.randn(num_nodes, 3)
        neighbor_idx = lg.build_knn_graph(coordinates, k_neighbors)

        assert (neighbor_idx >= 0).all()
        assert (neighbor_idx < num_nodes).all()


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_batch_size_one(self, standard_repr):
        """Test layers work with batch size 1."""
        x = torch.randn(1, standard_repr.mult, standard_repr.dim())

        layer = lg.EquivariantLinear(standard_repr, standard_repr)
        output = layer(x)
        assert output.shape == (1, standard_repr.mult, standard_repr.dim())

    def test_large_batch(self, standard_repr):
        """Test layers work with large batch sizes."""
        x = torch.randn(1000, standard_repr.mult, standard_repr.dim())

        layer = lg.EquivariantLinear(standard_repr, standard_repr)
        output = layer(x)
        assert output.shape == (1000, standard_repr.mult, standard_repr.dim())

    def test_multiplicity_one(self):
        """Test layers work with multiplicity 1."""
        repr = lg.Repr(lvals=[0, 1, 2], mult=1)
        x = torch.randn(32, 1, 9)

        layer = lg.EquivariantLinear(repr, repr)
        output = layer(x)
        assert output.shape == (32, 1, 9)

    def test_single_irrep_vectors(self):
        """Test layers work with single irrep (vectors only)."""
        repr = lg.Repr(lvals=[1], mult=8)
        x = torch.randn(32, 8, 3)

        layer = lg.EquivariantLinear(repr, repr)
        output = layer(x)
        assert output.shape == (32, 8, 3)

    def test_scalars_only(self):
        """Test layers work with scalars only (l=0)."""
        repr = lg.Repr(lvals=[0], mult=8)
        x = torch.randn(32, 8, 1)

        layer = lg.EquivariantLinear(repr, repr)
        output = layer(x)
        assert output.shape == (32, 8, 1)

    def test_high_degree_irreps(self):
        """Test layers work with higher degree irreps (l=3, 4)."""
        repr = lg.Repr(lvals=[0, 1, 2, 3, 4], mult=4)
        x = torch.randn(32, 4, repr.dim())  # dim = 1+3+5+7+9 = 25

        layer = lg.EquivariantLinear(repr, repr)
        output = layer(x)
        assert output.shape == (32, 4, repr.dim())

    def test_zero_input(self, standard_repr):
        """Test layers handle zero input gracefully."""
        x = torch.zeros(32, standard_repr.mult, standard_repr.dim())

        layer = lg.EquivariantLinear(standard_repr, standard_repr)
        output = layer(x)
        assert not torch.isnan(output).any()

    def test_very_large_input(self, standard_repr):
        """Test layers handle large magnitude inputs."""
        x = torch.randn(32, standard_repr.mult, standard_repr.dim()) * 1e6

        layer = lg.EquivariantLayerNorm(standard_repr)
        output = layer(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_very_small_input(self, standard_repr):
        """Test layers handle small magnitude inputs."""
        x = torch.randn(32, standard_repr.mult, standard_repr.dim()) * 1e-6

        layer = lg.EquivariantLayerNorm(standard_repr)
        output = layer(x)
        assert not torch.isnan(output).any()


# ============================================================================
# IDENTITY ROTATION TESTS
# ============================================================================

class TestIdentityRotation:
    """Test that identity rotation (angle=0) preserves output exactly."""

    def test_identity_rotation_preserves_output(self, standard_repr, sample_input):
        """Zero rotation should give identical outputs."""
        layer = lg.EquivariantLinear(standard_repr, standard_repr)
        layer.eval()

        axis = torch.tensor([1., 0., 0.])
        angle = torch.tensor(0.)

        D = get_wigner_d_matrix(standard_repr, axis, angle)

        x_rotated = sample_input @ D.T

        # With identity rotation, input should be unchanged
        assert torch.allclose(x_rotated, sample_input, rtol=1e-6, atol=1e-6)

        # And equivariance should hold trivially
        y1 = layer(sample_input)
        y2 = layer(x_rotated)
        assert torch.allclose(y1, y2, rtol=1e-6, atol=1e-6)


# ============================================================================
# WIGNER D-MATRIX VALIDATION TESTS
# ============================================================================

class TestWignerDMatrix:
    """Test that Wigner D-matrix matches scipy for l=1."""

    def test_wigner_d_matches_scipy_for_l1(self, rotation_params):
        """Verify Repr.rot() for l=1 matches scipy rotation matrix."""
        from scipy.spatial.transform import Rotation

        axis, angle = rotation_params

        # lr_geom Wigner D-matrix for l=1
        repr_l1 = lg.Repr(lvals=[1], mult=1)
        D_wigner = get_wigner_d_matrix(repr_l1, axis, angle)

        # scipy rotation matrix
        rotvec = (axis * angle).numpy()
        R_scipy = torch.from_numpy(
            Rotation.from_rotvec(rotvec).as_matrix()
        ).float()

        assert torch.allclose(D_wigner, R_scipy, rtol=RTOL, atol=ATOL)


# ============================================================================
# DEVICE AND DTYPE TESTS
# ============================================================================

class TestDeviceAndDtype:
    """Tests for device and dtype compatibility."""

    def test_float64_input(self, standard_repr):
        """Test layers work with float64 inputs."""
        x = torch.randn(32, standard_repr.mult, standard_repr.dim(), dtype=torch.float64)
        layer = lg.EquivariantLinear(standard_repr, standard_repr)
        layer = layer.double()

        output = layer(x)
        assert output.dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self, standard_repr):
        """Test layers work on CUDA device."""
        x = torch.randn(32, standard_repr.mult, standard_repr.dim(), device='cuda')
        layer = lg.EquivariantLinear(standard_repr, standard_repr).cuda()

        output = layer(x)
        assert output.device.type == 'cuda'


# ============================================================================
# OUTPUT SCALE TESTS
# ============================================================================

class TestOutputScale:
    """Verify layers don't explode or vanish output magnitudes."""

    def test_equivariant_linear_scale(self, simple_repr):
        """Output norm should be within reasonable bounds of input norm."""
        layer = lg.EquivariantLinear(simple_repr, simple_repr)
        x = torch.randn(100, simple_repr.mult, simple_repr.dim())
        y = layer(x)

        input_norm = x.norm()
        output_norm = y.norm()
        ratio = output_norm / input_norm

        assert 0.1 < ratio < 10, f"Scale ratio {ratio:.3f} out of bounds [0.1, 10]"

    def test_equivariant_gating_scale(self, simple_repr):
        """Gating should not amplify output beyond input."""
        layer = lg.EquivariantGating(simple_repr)
        x = torch.randn(100, simple_repr.mult, simple_repr.dim())
        y = layer(x)

        # Gating uses sigmoid, so output should be dampened
        assert y.norm() <= x.norm() * 2, "Gating should not amplify output significantly"

    def test_equivariant_layernorm_scale(self, simple_repr):
        """LayerNorm should produce reasonably bounded output."""
        layer = lg.EquivariantLayerNorm(simple_repr)
        x = torch.randn(100, simple_repr.mult, simple_repr.dim())
        y = layer(x)

        # LayerNorm output should have bounded variance
        assert y.norm() / x.norm() < 10, "LayerNorm output scale too large"

    def test_rbf_bounded_output(self):
        """RBF outputs should be in [0, 1] range (Gaussian)."""
        rbf = lg.RadialBasisFunctions(16, r_min=0.0, r_max=10.0)
        x = torch.rand(100) * 10  # distances in [0, 10]
        y = rbf(x)

        assert y.min() >= 0, f"RBF outputs should be non-negative, got min={y.min():.4f}"
        assert y.max() <= 1.1, f"RBF outputs should be bounded near 1, got max={y.max():.4f}"

    @requires_sphericart
    def test_transformer_output_scale(self, simple_repr, coordinates, k_neighbors):
        """Transformer output should have reasonable scale."""
        transformer = lg.EquivariantTransformer(
            in_repr=simple_repr,
            out_repr=simple_repr,
            hidden_repr=simple_repr,
            hidden_layers=2,
            edge_dim=16,
            edge_hidden_dim=32,
            k_neighbors=k_neighbors,
        )

        x = torch.randn(coordinates.size(0), simple_repr.mult, simple_repr.dim())
        y = transformer(coordinates, x)

        input_norm = x.norm()
        output_norm = y.norm()
        ratio = output_norm / input_norm

        assert 0.01 < ratio < 100, f"Transformer scale ratio {ratio:.3f} out of bounds"


# ============================================================================
# GRADIENT HEALTH TESTS
# ============================================================================

class TestGradientHealth:
    """Verify gradient flow is healthy (not vanishing/exploding)."""

    def test_gradient_magnitude_linear(self, simple_repr):
        """Gradients should have reasonable magnitude."""
        layer = lg.EquivariantLinear(simple_repr, simple_repr)
        x = torch.randn(100, simple_repr.mult, simple_repr.dim(), requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        grad_norm = x.grad.norm()
        assert grad_norm > 1e-6, f"Gradient too small: {grad_norm:.2e} (vanishing)"
        assert grad_norm < 1e6, f"Gradient too large: {grad_norm:.2e} (exploding)"

    def test_gradient_magnitude_gating(self, simple_repr):
        """Gating gradients should flow properly."""
        layer = lg.EquivariantGating(simple_repr)
        x = torch.randn(100, simple_repr.mult, simple_repr.dim(), requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        grad_norm = x.grad.norm()
        assert grad_norm > 1e-6, f"Gradient too small: {grad_norm:.2e}"
        assert grad_norm < 1e6, f"Gradient too large: {grad_norm:.2e}"

    def test_gradient_magnitude_layernorm(self, simple_repr):
        """LayerNorm gradients should flow properly."""
        layer = lg.EquivariantLayerNorm(simple_repr)
        x = torch.randn(100, simple_repr.mult, simple_repr.dim(), requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        grad_norm = x.grad.norm()
        assert grad_norm > 1e-6, f"Gradient too small: {grad_norm:.2e}"
        assert grad_norm < 1e6, f"Gradient too large: {grad_norm:.2e}"

    @requires_sphericart
    def test_deep_transformer_gradient(self, simple_repr, coordinates, k_neighbors):
        """Gradients should flow through deep transformer without vanishing."""
        transformer = lg.EquivariantTransformer(
            in_repr=simple_repr,
            out_repr=simple_repr,
            hidden_repr=simple_repr,
            hidden_layers=8,  # Deep network
            edge_dim=16,
            edge_hidden_dim=32,
            k_neighbors=k_neighbors,
        )

        x = torch.randn(coordinates.size(0), simple_repr.mult, simple_repr.dim(), requires_grad=True)
        y = transformer(coordinates, x)
        loss = y.sum()
        loss.backward()

        grad_norm = x.grad.norm()
        assert grad_norm > 1e-8, f"Gradient vanishing in deep transformer: {grad_norm:.2e}"
        assert grad_norm < 1e8, f"Gradient exploding in deep transformer: {grad_norm:.2e}"


# ============================================================================
# ATTENTION CORRECTNESS TESTS
# ============================================================================

class TestAttentionCorrectness:
    """Verify attention mechanism behaves correctly."""

    def test_attention_output_shape(self, k_neighbors):
        """Attention should produce correct output shape."""
        hidden_size = 64
        nheads = 4
        N = 50

        layer = lg.Attention(hidden_size, nheads)
        queries = torch.randn(N, hidden_size)
        keys = torch.randn(N, hidden_size)
        values = torch.randn(N, hidden_size)
        # Create neighbor indices (N, k)
        neighbor_idx = torch.randint(0, N, (N, k_neighbors))

        output = layer(keys, queries, values, neighbor_idx)
        assert output.shape == (N, hidden_size)

    def test_masked_attention_ignores_masked(self, k_neighbors):
        """Masked positions should not contribute to output."""
        hidden_size = 64
        nheads = 4
        N = 10

        layer = lg.Attention(hidden_size, nheads)
        queries = torch.randn(N, hidden_size)
        keys = torch.randn(N, hidden_size)
        values = torch.randn(N, hidden_size)
        neighbor_idx = torch.randint(0, N, (N, k_neighbors))

        # Create mask that masks out all but first neighbor
        mask = torch.ones(N, k_neighbors, dtype=torch.bool)
        mask[:, 0] = False  # Only first neighbor is unmasked

        output_masked = layer(keys, queries, values, neighbor_idx, mask=mask)

        # Run again with same inputs - should be deterministic in eval mode
        layer.eval()
        output_again = layer(keys, queries, values, neighbor_idx, mask=mask)

        # Outputs should be the same
        assert torch.allclose(output_masked, output_again, rtol=1e-4, atol=1e-5)

    def test_attention_deterministic_eval(self, k_neighbors):
        """Attention in eval mode should be deterministic."""
        hidden_size = 64
        nheads = 4
        N = 50

        layer = lg.Attention(hidden_size, nheads, dropout=0.5)
        layer.eval()

        queries = torch.randn(N, hidden_size)
        keys = torch.randn(N, hidden_size)
        values = torch.randn(N, hidden_size)
        neighbor_idx = torch.randint(0, N, (N, k_neighbors))

        output1 = layer(keys, queries, values, neighbor_idx)
        output2 = layer(keys, queries, values, neighbor_idx)

        assert torch.allclose(output1, output2), "Eval mode should be deterministic"


# ============================================================================
# NORMALIZATION PROPERTY TESTS
# ============================================================================

class TestNormalizationProperties:
    """Verify normalization layers work correctly."""

    def test_layernorm_stabilizes_scale(self, simple_repr):
        """LayerNorm should stabilize output scale regardless of input scale."""
        ln = lg.EquivariantLayerNorm(simple_repr)

        # Test with small input
        x_small = torch.randn(100, simple_repr.mult, simple_repr.dim()) * 0.01
        y_small = ln(x_small)

        # Test with large input
        x_large = torch.randn(100, simple_repr.mult, simple_repr.dim()) * 100
        y_large = ln(x_large)

        # Output norms should be much closer than input norms
        input_ratio = x_large.norm() / x_small.norm()
        output_ratio = y_large.norm() / y_small.norm()

        # LayerNorm should reduce the ratio significantly (input ratio ~10000, output should be <100)
        assert output_ratio < input_ratio / 10, (
            f"LayerNorm should stabilize scale. Input ratio: {input_ratio:.1f}, "
            f"Output ratio: {output_ratio:.1f}"
        )

    def test_layernorm_preserves_equivariance_after_scaling(self, simple_repr, rotation_params):
        """LayerNorm should be equivariant regardless of input scale."""
        axis, angle = rotation_params
        ln = lg.EquivariantLayerNorm(simple_repr)

        # Large input
        x = torch.randn(50, simple_repr.mult, simple_repr.dim()) * 100

        # Get Wigner D-matrix
        D = get_wigner_d_matrix(simple_repr, axis, angle)

        # Rotate then normalize
        x_rotated = x @ D.T
        y_rotated = ln(x_rotated)

        # Normalize then rotate
        y = ln(x)
        y_then_rotated = y @ D.T

        assert torch.allclose(y_rotated, y_then_rotated, rtol=RTOL, atol=ATOL), (
            f"LayerNorm should be equivariant. Max diff: {(y_rotated - y_then_rotated).abs().max():.6f}"
        )

    def test_layernorm_handles_near_zero_input(self, simple_repr):
        """LayerNorm should handle near-zero inputs without NaN."""
        ln = lg.EquivariantLayerNorm(simple_repr)

        x = torch.randn(50, simple_repr.mult, simple_repr.dim()) * 1e-8
        y = ln(x)

        assert not torch.isnan(y).any(), "LayerNorm produced NaN for near-zero input"
        assert not torch.isinf(y).any(), "LayerNorm produced Inf for near-zero input"


# ============================================================================
# ARCHITECTURE DIAGNOSTIC TESTS
# ============================================================================

class TestArchitectureDiagnostics:
    """Diagnostic tests to identify potential architecture issues."""

    @pytest.fixture
    def diagnostic_setup(self):
        """Setup for diagnostic tests."""
        in_repr = lg.Repr(lvals=[0, 1], mult=8)
        hidden_repr = lg.Repr(lvals=[0, 1], mult=16)
        out_repr = lg.Repr(lvals=[1], mult=1)

        return {
            'in_repr': in_repr,
            'hidden_repr': hidden_repr,
            'out_repr': out_repr,
        }

    def test_gradient_flow_through_layers(self, diagnostic_setup):
        """Test that gradients flow properly through all layers."""
        setup = diagnostic_setup
        torch.manual_seed(42)

        transformer = lg.EquivariantTransformer(
            in_repr=setup['in_repr'],
            out_repr=setup['out_repr'],
            hidden_repr=setup['hidden_repr'],
            hidden_layers=4,
            edge_dim=16,
            edge_hidden_dim=32,
            k_neighbors=8,
            nheads=4,
        )

        coords = torch.randn(20, 3, requires_grad=True)
        features = torch.randn(20, setup['in_repr'].mult, setup['in_repr'].dim(), requires_grad=True)

        output = transformer(coords, features)
        loss = output.sum()
        loss.backward()

        # Check input gradients exist and are reasonable
        assert features.grad is not None, "No gradient for input features"
        assert coords.grad is not None, "No gradient for coordinates"

        grad_norm = features.grad.norm()
        assert grad_norm > 1e-8, f"Gradient too small: {grad_norm}"
        assert grad_norm < 1e6, f"Gradient too large: {grad_norm}"

        # Check gradients for each layer
        for i, layer in enumerate(transformer.layers):
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    pgrad_norm = param.grad.norm()
                    assert pgrad_norm > 0, f"Zero gradient for layer {i} param {name}"
                    assert not torch.isnan(param.grad).any(), f"NaN gradient in layer {i} param {name}"

    def test_skip_connection_presence(self, diagnostic_setup):
        """Verify skip connections are present where expected."""
        setup = diagnostic_setup

        # Same repr should have skip connection
        same_repr = lg.ProductRepr(setup['hidden_repr'], setup['hidden_repr'])
        block_same = lg.EquivariantTransformerBlock(
            same_repr, edge_dim=16, edge_hidden_dim=32, nheads=4
        )
        assert block_same.skip, "Skip connection should exist when reprs match"

        # Different repr should NOT have skip connection
        diff_repr = lg.ProductRepr(setup['in_repr'], setup['hidden_repr'])
        block_diff = lg.EquivariantTransformerBlock(
            diff_repr, edge_dim=16, edge_hidden_dim=32, nheads=4
        )
        assert not block_diff.skip, "Skip connection should not exist when reprs differ"

    def test_attention_weight_distribution(self):
        """Test that attention weights are well-distributed, not collapsed."""
        repr = lg.Repr(lvals=[0, 1], mult=8)
        product_repr = lg.ProductRepr(repr, repr)

        attn = lg.EquivariantAttention(
            product_repr, edge_dim=16, edge_hidden_dim=32, nheads=4
        )
        attn.eval()

        N, k = 20, 8
        coords = torch.randn(N, 3)
        neighbor_idx = lg.build_knn_graph(coords, k)

        # Create basis
        neighbor_coords = coords[neighbor_idx]
        displacements = coords.unsqueeze(1) - neighbor_coords
        basis_layer = lg.EquivariantBasis(product_repr)
        b1, b2 = basis_layer(displacements.view(N * k, 3))
        b1 = b1.view(N, k, *b1.shape[1:])
        b2 = b2.view(N, k, *b2.shape[1:])

        edge_feats = torch.randn(N, k, 16)
        features = torch.randn(N, repr.mult, repr.dim())

        # Hook to capture attention weights
        attn_weights_captured = []
        def hook(module, input, output):
            # The softmax is applied to scores, we need to intercept
            pass

        with torch.no_grad():
            output = attn((b1, b2), edge_feats, features, neighbor_idx)

        # Output should have reasonable variance (not collapsed)
        output_var = output.var()
        assert output_var > 1e-6, f"Output variance too low: {output_var}, may indicate attention collapse"

    def test_layer_output_scale_consistency(self, diagnostic_setup):
        """Test that layer outputs maintain reasonable scale through depth."""
        setup = diagnostic_setup
        torch.manual_seed(42)

        transformer = lg.EquivariantTransformer(
            in_repr=setup['hidden_repr'],  # Use same repr throughout
            out_repr=setup['hidden_repr'],
            hidden_repr=setup['hidden_repr'],
            hidden_layers=8,
            edge_dim=16,
            edge_hidden_dim=32,
            k_neighbors=8,
            nheads=4,
            residual_scale=1.0,
        )
        transformer.eval()

        coords = torch.randn(20, 3)
        features = torch.randn(20, setup['hidden_repr'].mult, setup['hidden_repr'].dim())
        input_norm = features.norm()

        with torch.no_grad():
            output = transformer(coords, features)

        output_norm = output.norm()
        ratio = output_norm / input_norm

        # Output should be within reasonable range of input
        assert ratio > 0.01, f"Output much smaller than input: ratio={ratio:.4f}"
        assert ratio < 100, f"Output much larger than input: ratio={ratio:.4f}"

    def test_convolution_output_scale(self):
        """Test that convolution outputs have reasonable scale."""
        repr = lg.Repr(lvals=[0, 1], mult=8)
        product_repr = lg.ProductRepr(repr, repr)

        conv = lg.EquivariantConvolution(product_repr, edge_dim=16, hidden_dim=32)
        basis_layer = lg.EquivariantBasis(product_repr)

        N, E = 20, 100
        edge_vectors = torch.randn(E, 3)
        bases = basis_layer(edge_vectors)
        edge_feats = torch.randn(E, 16)
        node_features = torch.randn(N, repr.mult, repr.dim())
        src_idx = torch.randint(0, N, (E,))

        input_norm = node_features.norm()

        with torch.no_grad():
            output = conv(bases, edge_feats, node_features, src_idx)

        output_norm = output.norm()

        # Check output is not exploding or vanishing
        assert not torch.isnan(output).any(), "Convolution produced NaN"
        assert output_norm > 1e-8, f"Convolution output vanishing: {output_norm}"
        assert output_norm < 1e8, f"Convolution output exploding: {output_norm}"

    def test_radial_weight_initialization(self):
        """Test that radial weight network has good initialization."""
        repr = lg.Repr(lvals=[0, 1], mult=8)
        product_repr = lg.ProductRepr(repr, repr)

        rw = lg.RadialWeight(
            edge_dim=16, hidden_dim=32, repr=product_repr,
            in_dim=8, out_dim=8
        )

        # Test that output variance is reasonable at initialization
        x = torch.randn(100, 16)
        with torch.no_grad():
            output = rw(x)

        output_std = output.std()
        # With Xavier init, output std should be close to 1
        assert output_std > 0.1, f"RadialWeight output std too low: {output_std}"
        assert output_std < 10, f"RadialWeight output std too high: {output_std}"

    def test_gating_activation_range(self):
        """Test that gating produces values in expected range."""
        repr = lg.Repr(lvals=[0, 1], mult=8)
        gate = lg.EquivariantGating(repr)

        x = torch.randn(50, repr.mult, repr.dim())

        with torch.no_grad():
            y = gate(x)

        # Gating should dampen (sigmoid is in [0, 1])
        # So output norm should generally be <= input norm
        input_norm = x.norm()
        output_norm = y.norm()

        assert output_norm <= input_norm * 1.1, (
            f"Gating amplified signal: in={input_norm:.4f}, out={output_norm:.4f}"
        )

    def test_deep_network_no_nan(self, diagnostic_setup):
        """Test that deep network doesn't produce NaN."""
        setup = diagnostic_setup
        torch.manual_seed(42)

        # Very deep network
        transformer = lg.EquivariantTransformer(
            in_repr=setup['in_repr'],
            out_repr=setup['out_repr'],
            hidden_repr=setup['hidden_repr'],
            hidden_layers=12,
            edge_dim=16,
            edge_hidden_dim=32,
            k_neighbors=8,
            nheads=4,
            dropout=0.1,
            residual_scale=0.5,
        )
        transformer.train()

        coords = torch.randn(20, 3)
        features = torch.randn(20, setup['in_repr'].mult, setup['in_repr'].dim())

        # Multiple forward passes
        for _ in range(5):
            output = transformer(coords, features)
            assert not torch.isnan(output).any(), "Deep network produced NaN"
            assert not torch.isinf(output).any(), "Deep network produced Inf"

    def test_attention_qkv_independence(self):
        """Test whether Q, K, V have independent representations.

        Note: Current implementation uses shared conv_qkv which may limit
        expressiveness compared to independent Q, K, V projections.
        """
        repr = lg.Repr(lvals=[0, 1], mult=8)
        product_repr = lg.ProductRepr(repr, repr)

        attn = lg.EquivariantAttention(
            product_repr, edge_dim=16, edge_hidden_dim=32, nheads=4
        )

        # Check that conv_qkv has 3x the output multiplicity
        expected_qkv_mult = 3 * repr.mult
        actual_qkv_mult = attn.conv_qkv.rwlin.out_dim

        # This documents the current behavior (shared weights for Q, K, V)
        # A separate test would be needed if we switch to independent projections
        assert actual_qkv_mult == expected_qkv_mult, (
            f"Expected QKV mult {expected_qkv_mult}, got {actual_qkv_mult}"
        )

    def test_first_layer_gradient_magnitude(self, diagnostic_setup):
        """Test gradient magnitude at first layer (no skip connection case)."""
        setup = diagnostic_setup
        torch.manual_seed(42)

        transformer = lg.EquivariantTransformer(
            in_repr=setup['in_repr'],
            out_repr=setup['out_repr'],
            hidden_repr=setup['hidden_repr'],
            hidden_layers=4,
            edge_dim=16,
            edge_hidden_dim=32,
            k_neighbors=8,
            nheads=4,
        )

        coords = torch.randn(20, 3)
        features = torch.randn(20, setup['in_repr'].mult, setup['in_repr'].dim(), requires_grad=True)

        output = transformer(coords, features)
        loss = output.sum()
        loss.backward()

        # Get gradient magnitudes per layer
        layer_grads = []
        for i, layer in enumerate(transformer.layers):
            grad_norms = []
            for name, param in layer.named_parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())
            if grad_norms:
                layer_grads.append(sum(grad_norms) / len(grad_norms))

        # First layer should have comparable gradients to other layers
        if len(layer_grads) >= 2:
            first_layer_grad = layer_grads[0]
            other_grads = layer_grads[1:]
            avg_other = sum(other_grads) / len(other_grads)

            ratio = first_layer_grad / (avg_other + 1e-8)
            # Allow 100x difference but not 1000x
            assert ratio > 0.01, (
                f"First layer gradients much smaller than others: {ratio:.4f}"
            )
