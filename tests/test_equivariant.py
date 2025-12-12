"""Tests for lr_geom equivariant module components.

Tests cover:
1. RepNorm - Spherical tensor norm computation
2. SphericalHarmonic - Spherical harmonics computation
3. RadialBasisFunctions - Radial basis function expansion
4. EquivariantBasis - Equivariant basis matrices
5. EquivariantBases - Multiple equivariant bases
"""
from __future__ import annotations

import math
import pytest
import torch

import lr_geom as lg
from lr_geom.equivariant import RepNorm, SphericalHarmonic, RadialBasisFunctions, EquivariantBasis, EquivariantBases
from conftest import requires_sphericart, get_wigner_d_matrix, get_3d_rotation_matrix, RTOL, ATOL


# ============================================================================
# REPNORM TESTS
# ============================================================================

class TestRepNorm:
    """Tests for RepNorm layer."""

    def test_forward_shape(self):
        """Test output shape matches number of irreps."""
        repr = lg.Repr(lvals=[0, 1, 2], mult=1)  # 3 irreps, dim = 1+3+5 = 9
        norm = RepNorm(repr)

        x = torch.randn(32, repr.dim())
        output = norm(x)

        assert output.shape == (32, 3)  # One norm per irrep

    def test_forward_no_nan(self):
        """Test output contains no NaN values."""
        repr = lg.Repr(lvals=[0, 1, 2], mult=1)
        norm = RepNorm(repr)

        x = torch.randn(32, repr.dim())
        output = norm(x)

        assert not torch.isnan(output).any()

    def test_norms_positive(self):
        """Test all norms are non-negative."""
        repr = lg.Repr(lvals=[0, 1, 2], mult=1)
        norm = RepNorm(repr)

        x = torch.randn(32, repr.dim())
        output = norm(x)

        assert (output >= 0).all()

    def test_zero_input_zero_norms(self):
        """Test zero input produces zero norms."""
        repr = lg.Repr(lvals=[0, 1, 2], mult=1)
        norm = RepNorm(repr)

        x = torch.zeros(32, repr.dim())
        output = norm(x)

        assert torch.allclose(output, torch.zeros_like(output))

    def test_scalar_norm(self):
        """Test norm of scalars equals absolute value."""
        repr = lg.Repr(lvals=[0], mult=1)  # Scalars only
        norm = RepNorm(repr)

        x = torch.randn(32, 1)
        output = norm(x)

        expected = x.abs()
        assert torch.allclose(output, expected)

    def test_vector_norm(self):
        """Test norm of vectors matches torch.norm."""
        repr = lg.Repr(lvals=[1], mult=1)  # Vectors only
        norm = RepNorm(repr)

        x = torch.randn(32, 3)
        output = norm(x)

        expected = x.norm(dim=-1, keepdim=True)
        assert torch.allclose(output, expected)

    def test_rotation_invariance(self, rotation_params):
        """Test norms are invariant under rotation."""
        axis, angle = rotation_params
        repr = lg.Repr(lvals=[0, 1, 2], mult=1)
        norm_layer = RepNorm(repr)

        # Get Wigner D-matrix
        D = get_wigner_d_matrix(repr, axis, angle)

        x = torch.randn(32, repr.dim())
        x_rotated = x @ D.T

        # Norms should be the same
        norms_original = norm_layer(x)
        norms_rotated = norm_layer(x_rotated)

        assert torch.allclose(norms_original, norms_rotated, rtol=RTOL, atol=ATOL)

    def test_multiple_batches(self):
        """Test with batched input."""
        repr = lg.Repr(lvals=[0, 1], mult=1)
        norm = RepNorm(repr)

        x = torch.randn(10, 20, repr.dim())  # Extra batch dimension
        output = norm(x)

        assert output.shape == (10, 20, 2)

    def test_split_sizes_match_repr(self):
        """Test split sizes are computed correctly."""
        repr = lg.Repr(lvals=[0, 1, 2], mult=1)
        norm = RepNorm(repr)

        expected_sizes = [1, 3, 5]  # 2*l+1 for each l
        assert norm.split_sizes == expected_sizes


# ============================================================================
# SPHERICALHARMONIC TESTS
# ============================================================================

@requires_sphericart
class TestSphericalHarmonic:
    """Tests for SphericalHarmonic layer."""

    def test_forward_shape(self):
        """Test output shape is (N, (lmax+1)^2)."""
        lmax = 3
        sh = SphericalHarmonic(lmax)

        coords = torch.randn(100, 3)
        output = sh(coords)

        expected_features = (lmax + 1) ** 2  # 16
        assert output.shape == (100, expected_features)

    def test_forward_no_nan(self):
        """Test output contains no NaN values."""
        sh = SphericalHarmonic(lmax=2)

        coords = torch.randn(100, 3)
        output = sh(coords)

        assert not torch.isnan(output).any()

    def test_zero_vector_handling(self):
        """Test zero vectors don't produce NaN.

        Note: Y_0^0 = 1/sqrt(4*pi) is constant for all directions,
        so even zero vectors get this constant value for l=0.
        Higher l components are set to zero.
        """
        sh = SphericalHarmonic(lmax=2)

        coords = torch.zeros(10, 3)
        output = sh(coords)

        # Should not have NaN
        assert not torch.isnan(output).any()
        # l>0 components should be zero
        assert torch.allclose(output[:, 1:], torch.zeros_like(output[:, 1:]))

    def test_batched_input(self):
        """Test with batched input."""
        sh = SphericalHarmonic(lmax=2)

        coords = torch.randn(10, 20, 3)  # Extra batch dimension
        output = sh(coords)

        assert output.shape == (10, 20, 9)  # (lmax+1)^2 = 9

    def test_l0_constant(self):
        """Test l=0 spherical harmonic is constant (Y_0^0)."""
        sh = SphericalHarmonic(lmax=0)

        coords = torch.randn(100, 3)
        output = sh(coords)

        # All l=0 values should be the same (constant)
        # Note: normalized SH has Y_0^0 = 1/sqrt(4*pi) for unit vectors
        std = output.std()
        assert std < 0.1, f"l=0 should be approximately constant, got std={std}"

    def test_equivariance(self, rotation_params):
        """Test spherical harmonics transform correctly under rotation.

        For rotation R and Wigner D:
        Y_l^m(R @ x) = sum_m' D_l^{m'm} Y_l^{m'}(x)
        """
        axis, angle = rotation_params
        sh = SphericalHarmonic(lmax=2)

        # Create representation for the SH output
        sh_repr = lg.Repr(lvals=list(range(3)), mult=1)  # lvals=[0,1,2]

        coords = torch.randn(50, 3)

        # Get rotation matrices
        D = get_wigner_d_matrix(sh_repr, axis, angle)
        R = get_3d_rotation_matrix(axis, angle)

        # Rotate coordinates
        coords_rotated = coords @ R.T

        # Compute SH
        sh_original = sh(coords)
        sh_rotated = sh(coords_rotated)

        # SH should transform: Y(R @ x) = D @ Y(x)
        sh_transformed = sh_original @ D.T

        assert torch.allclose(sh_rotated, sh_transformed, rtol=1e-3, atol=1e-3)

    def test_pairwise_shape(self):
        """Test pairwise SH computation shape."""
        sh = SphericalHarmonic(lmax=2)

        coords = torch.randn(10, 3)
        output = sh.pairwise(coords)

        assert output.shape == (10, 10, 9)  # (N, N, (lmax+1)^2)

    def test_pairwise_no_nan(self):
        """Test pairwise SH contains no NaN."""
        sh = SphericalHarmonic(lmax=2)

        coords = torch.randn(10, 3)
        output = sh.pairwise(coords)

        assert not torch.isnan(output).any()

    def test_invalid_lmax_negative(self):
        """Test negative lmax raises ValueError."""
        with pytest.raises(ValueError, match="lmax must be a non-negative integer"):
            SphericalHarmonic(lmax=-1)

    def test_invalid_lmax_float(self):
        """Test float lmax raises ValueError."""
        with pytest.raises(ValueError, match="lmax must be a non-negative integer"):
            SphericalHarmonic(lmax=1.5)

    def test_dtype_handling(self):
        """Test float16 input is handled correctly."""
        sh = SphericalHarmonic(lmax=2)

        coords = torch.randn(10, 3, dtype=torch.float16)
        output = sh(coords)

        # Output should be same dtype as input
        assert output.dtype == torch.float16


# ============================================================================
# RADIALBASISFUNCTION TESTS
# ============================================================================

class TestRadialBasisFunctions:
    """Tests for RadialBasisFunctions layer."""

    def test_gaussian_forward_shape(self):
        """Test Gaussian RBF output shape."""
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="gaussian")

        distances = torch.rand(100) * 10
        output = rbf(distances)

        assert output.shape == (100, 16)

    def test_gaussian_forward_no_nan(self):
        """Test Gaussian RBF output contains no NaN values."""
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="gaussian")

        distances = torch.rand(100) * 10
        output = rbf(distances)

        assert not torch.isnan(output).any()

    def test_gaussian_bounded(self):
        """Test Gaussian RBF output is in [0, 1]."""
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="gaussian")

        distances = torch.rand(100) * 10
        output = rbf(distances)

        assert output.min() >= 0
        assert output.max() <= 1.01  # Small tolerance for numerical errors

    def test_bessel_forward_shape(self):
        """Test Bessel RBF output shape."""
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="bessel")

        distances = torch.rand(100) * 10
        output = rbf(distances)

        assert output.shape == (100, 16)

    def test_bessel_forward_no_nan(self):
        """Test Bessel RBF output contains no NaN values."""
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="bessel")

        distances = torch.rand(100) * 10
        output = rbf(distances)

        assert not torch.isnan(output).any()

    def test_bessel_smooth_cutoff(self):
        """Test Bessel RBF has smooth cutoff at r_max."""
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="bessel")

        # Values at cutoff should be zero
        distances = torch.tensor([10.0, 10.5, 11.0])
        output = rbf(distances)

        assert torch.allclose(output[0], torch.zeros(16), atol=1e-5)
        assert torch.allclose(output[1], torch.zeros(16), atol=1e-5)
        assert torch.allclose(output[2], torch.zeros(16), atol=1e-5)

    def test_polynomial_forward_shape(self):
        """Test Polynomial RBF output shape."""
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="polynomial")

        distances = torch.rand(100) * 10
        output = rbf(distances)

        assert output.shape == (100, 16)

    def test_polynomial_forward_no_nan(self):
        """Test Polynomial RBF output contains no NaN values."""
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="polynomial")

        distances = torch.rand(100) * 10
        output = rbf(distances)

        assert not torch.isnan(output).any()

    def test_polynomial_bounded(self):
        """Test Polynomial RBF output is in [0, 1]."""
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="polynomial")

        distances = torch.rand(100) * 10
        output = rbf(distances)

        assert output.min() >= 0
        assert output.max() <= 1.01

    def test_polynomial_smooth_cutoff(self):
        """Test Polynomial RBF has smooth cutoff at r_max."""
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="polynomial")

        # Values at cutoff should be zero
        distances = torch.tensor([10.0, 10.5])
        output = rbf(distances)

        assert torch.allclose(output[0], torch.zeros(16), atol=1e-5)

    def test_invalid_rbf_type(self):
        """Test invalid rbf_type raises ValueError."""
        with pytest.raises(ValueError, match="rbf_type must be one of"):
            RadialBasisFunctions(16, rbf_type="invalid")

    def test_invalid_num_functions(self):
        """Test invalid num_functions raises ValueError."""
        with pytest.raises(ValueError, match="num_functions must be a positive integer"):
            RadialBasisFunctions(0, r_min=0.0, r_max=10.0)

        with pytest.raises(ValueError, match="num_functions must be a positive integer"):
            RadialBasisFunctions(-5, r_min=0.0, r_max=10.0)

    def test_invalid_r_range(self):
        """Test r_max <= r_min raises ValueError."""
        with pytest.raises(ValueError, match="r_max .* must be greater than r_min"):
            RadialBasisFunctions(16, r_min=10.0, r_max=5.0)

    def test_gaussian_learnable_parameters(self):
        """Test Gaussian RBF has learnable mu and sigma."""
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="gaussian")

        assert hasattr(rbf, 'mu')
        assert hasattr(rbf, 'sigma')
        assert rbf.mu.requires_grad
        assert rbf.sigma.requires_grad

    def test_bessel_has_buffer(self):
        """Test Bessel RBF has bessel_freqs buffer."""
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="bessel")

        assert hasattr(rbf, 'bessel_freqs')
        assert not rbf.bessel_freqs.requires_grad

    def test_polynomial_has_buffer(self):
        """Test Polynomial RBF has poly_powers buffer."""
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="polynomial")

        assert hasattr(rbf, 'poly_powers')
        assert not rbf.poly_powers.requires_grad

    def test_backward_gaussian(self):
        """Test gradients flow through Gaussian RBF."""
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="gaussian")

        # Create leaf tensor with requires_grad
        distances = torch.rand(100) * 10
        distances = distances.clone().requires_grad_(True)

        output = rbf(distances)
        loss = output.sum()
        loss.backward()

        assert distances.grad is not None
        assert rbf.mu.grad is not None
        assert rbf.sigma.grad is not None

    def test_batched_input(self):
        """Test with batched input."""
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="gaussian")

        distances = torch.rand(10, 20) * 10
        output = rbf(distances)

        assert output.shape == (10, 20, 16)


# ============================================================================
# EQUIVARIANTBASIS TESTS
# ============================================================================

@requires_sphericart
class TestEquivariantBasis:
    """Tests for EquivariantBasis layer."""

    def test_forward_shape(self):
        """Test output shapes are correct."""
        repr1 = lg.Repr(lvals=[0, 1], mult=4)
        repr2 = lg.Repr(lvals=[0, 1], mult=8)
        product_repr = lg.ProductRepr(repr1, repr2)

        basis = EquivariantBasis(product_repr)

        displacements = torch.randn(100, 3)
        coeff1, coeff2 = basis(displacements)

        # coeff1: (N, rep1.dim(), rep1.nreps())
        assert coeff1.shape == (100, repr1.dim(), repr1.nreps())
        # coeff2: (N, rep2.nreps(), rep2.dim())
        assert coeff2.shape == (100, repr2.nreps(), repr2.dim())

    def test_forward_no_nan(self):
        """Test output contains no NaN values."""
        repr = lg.Repr(lvals=[0, 1, 2], mult=4)
        product_repr = lg.ProductRepr(repr, repr)

        basis = EquivariantBasis(product_repr)

        displacements = torch.randn(100, 3)
        coeff1, coeff2 = basis(displacements)

        assert not torch.isnan(coeff1).any()
        assert not torch.isnan(coeff2).any()

    def test_zero_displacement_handling(self):
        """Test zero displacements are handled correctly."""
        repr = lg.Repr(lvals=[0, 1], mult=4)
        product_repr = lg.ProductRepr(repr, repr)

        basis = EquivariantBasis(product_repr)

        displacements = torch.zeros(10, 3)
        coeff1, coeff2 = basis(displacements)

        # Should not have NaN
        assert not torch.isnan(coeff1).any()
        assert not torch.isnan(coeff2).any()

    def test_equivariance(self, rotation_params):
        """Test basis transforms correctly under rotation.

        For rotation R and Wigner D:
        basis(R @ x) should transform with the corresponding Wigner matrices.
        """
        axis, angle = rotation_params
        repr = lg.Repr(lvals=[0, 1], mult=4)
        product_repr = lg.ProductRepr(repr, repr)

        basis = EquivariantBasis(product_repr)

        # Get rotation matrices
        D = get_wigner_d_matrix(repr, axis, angle)
        R = get_3d_rotation_matrix(axis, angle)

        displacements = torch.randn(50, 3)
        displacements_rotated = displacements @ R.T

        # Compute bases
        coeff1, coeff2 = basis(displacements)
        coeff1_rot, coeff2_rot = basis(displacements_rotated)

        # coeff1 transforms as: coeff1_rot = D @ coeff1 @ D.T
        # But structure is (N, dim, nreps), and D acts on dim axis
        # coeff1_rot[i, :, j] = D @ coeff1[i, :, j] for each j
        coeff1_expected = torch.einsum('ab,nbc->nac', D, coeff1)

        assert torch.allclose(coeff1_rot, coeff1_expected, rtol=1e-3, atol=1e-3)

    def test_backward_gradients(self):
        """Test gradients flow through basis computation."""
        repr = lg.Repr(lvals=[0, 1], mult=4)
        product_repr = lg.ProductRepr(repr, repr)

        basis = EquivariantBasis(product_repr)

        displacements = torch.randn(50, 3, requires_grad=True)
        coeff1, coeff2 = basis(displacements)

        loss = coeff1.sum() + coeff2.sum()
        loss.backward()

        assert displacements.grad is not None
        assert not torch.isnan(displacements.grad).any()

    def test_different_representations(self):
        """Test with different input/output representations."""
        repr1 = lg.Repr(lvals=[0, 1], mult=4)
        repr2 = lg.Repr(lvals=[0, 1, 2], mult=2)
        product_repr = lg.ProductRepr(repr1, repr2)

        basis = EquivariantBasis(product_repr)

        displacements = torch.randn(50, 3)
        coeff1, coeff2 = basis(displacements)

        assert coeff1.shape == (50, repr1.dim(), repr1.nreps())
        assert coeff2.shape == (50, repr2.nreps(), repr2.dim())


# ============================================================================
# EQUIVARIANTBASES TESTS
# ============================================================================

@requires_sphericart
class TestEquivariantBases:
    """Tests for EquivariantBases layer (multiple bases)."""

    def test_forward_returns_correct_count(self):
        """Test output has correct number of basis pairs."""
        repr1 = lg.Repr(lvals=[0, 1], mult=4)
        repr2 = lg.Repr(lvals=[0, 1, 2], mult=2)
        product_repr1 = lg.ProductRepr(repr1, repr1)
        product_repr2 = lg.ProductRepr(repr1, repr2)

        bases = EquivariantBases(product_repr1, product_repr2)

        displacements = torch.randn(100, 3)
        output = bases(displacements)

        assert len(output) == 2  # Two product reprs

    def test_deduplication(self):
        """Test that duplicate representations are computed only once."""
        repr = lg.Repr(lvals=[0, 1], mult=4)
        product_repr = lg.ProductRepr(repr, repr)

        # Pass same repr twice
        bases = EquivariantBases(product_repr, product_repr, product_repr)

        # Should only have one unique computation
        assert len(bases.unique_reprs) == 1
        assert len(bases.comps) == 1

        displacements = torch.randn(50, 3)
        output = bases(displacements)

        # But should return 3 results
        assert len(output) == 3

        # All should be identical
        assert torch.allclose(output[0][0], output[1][0])
        assert torch.allclose(output[0][0], output[2][0])

    def test_forward_no_nan(self):
        """Test output contains no NaN values."""
        repr = lg.Repr(lvals=[0, 1, 2], mult=4)
        product_repr = lg.ProductRepr(repr, repr)

        bases = EquivariantBases(product_repr)

        displacements = torch.randn(100, 3)
        output = bases(displacements)

        for coeff1, coeff2 in output:
            assert not torch.isnan(coeff1).any()
            assert not torch.isnan(coeff2).any()

    def test_mixed_representations(self):
        """Test with different representations."""
        repr1 = lg.Repr(lvals=[0, 1], mult=4)
        repr2 = lg.Repr(lvals=[1, 2], mult=2)
        repr3 = lg.Repr(lvals=[0, 1, 2], mult=1)

        product1 = lg.ProductRepr(repr1, repr1)
        product2 = lg.ProductRepr(repr1, repr2)
        product3 = lg.ProductRepr(repr2, repr3)

        bases = EquivariantBases(product1, product2, product3)

        # Should have 3 unique reprs
        assert len(bases.unique_reprs) == 3

        displacements = torch.randn(50, 3)
        output = bases(displacements)

        assert len(output) == 3

        # Check shapes
        c1_1, c2_1 = output[0]
        assert c1_1.shape == (50, repr1.dim(), repr1.nreps())

        c1_2, c2_2 = output[1]
        assert c1_2.shape == (50, repr1.dim(), repr1.nreps())
        assert c2_2.shape == (50, repr2.nreps(), repr2.dim())

        c1_3, c2_3 = output[2]
        assert c1_3.shape == (50, repr2.dim(), repr2.nreps())
        assert c2_3.shape == (50, repr3.nreps(), repr3.dim())

    def test_backward_gradients(self):
        """Test gradients flow through all bases."""
        repr = lg.Repr(lvals=[0, 1], mult=4)
        product_repr = lg.ProductRepr(repr, repr)

        bases = EquivariantBases(product_repr)

        displacements = torch.randn(50, 3, requires_grad=True)
        output = bases(displacements)

        loss = sum(c1.sum() + c2.sum() for c1, c2 in output)
        loss.backward()

        assert displacements.grad is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@requires_sphericart
class TestEquivariantIntegration:
    """Integration tests for equivariant module components."""

    def test_repnorm_with_sh_output(self):
        """Test RepNorm works with SphericalHarmonic output."""
        sh = SphericalHarmonic(lmax=2)
        repr = lg.Repr(lvals=[0, 1, 2], mult=1)
        norm = RepNorm(repr)

        coords = torch.randn(50, 3)
        sh_output = sh(coords)
        norms = norm(sh_output)

        assert norms.shape == (50, 3)
        assert not torch.isnan(norms).any()

    def test_basis_with_rbf(self):
        """Test EquivariantBasis with RadialBasisFunctions for edge features."""
        repr = lg.Repr(lvals=[0, 1], mult=4)
        product_repr = lg.ProductRepr(repr, repr)

        basis = EquivariantBasis(product_repr)
        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0)

        # Edge displacements
        displacements = torch.randn(100, 3)
        distances = displacements.norm(dim=-1)

        # Compute both
        coeff1, coeff2 = basis(displacements)
        edge_features = rbf(distances)

        assert coeff1.shape == (100, repr.dim(), repr.nreps())
        assert edge_features.shape == (100, 16)

    def test_full_pipeline(self):
        """Test full pipeline: coordinates -> SH -> basis -> norm."""
        repr = lg.Repr(lvals=[0, 1, 2], mult=1)
        product_repr = lg.ProductRepr(repr, repr)

        sh = SphericalHarmonic(lmax=2)
        basis = EquivariantBasis(product_repr)
        norm = RepNorm(repr)

        # Point cloud
        coords = torch.randn(20, 3)

        # Pairwise displacements (edges)
        edges = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 3)
        edges = edges.reshape(-1, 3)  # (N*N, 3)

        # Compute spherical harmonics
        sh_features = sh(edges)  # (N*N, 9)

        # Compute basis
        coeff1, coeff2 = basis(edges)

        # Compute norms (invariants)
        norms = norm(sh_features)

        assert not torch.isnan(sh_features).any()
        assert not torch.isnan(coeff1).any()
        assert not torch.isnan(norms).any()
