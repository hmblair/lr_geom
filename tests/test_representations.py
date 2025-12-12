"""Tests for SO(3) representation theory module."""
from __future__ import annotations

import pytest
import torch

import lr_geom as lg
from lr_geom.representations import Irrep, ProductIrrep, Repr, ProductRepr


# ============================================================================
# TEST: Irrep
# ============================================================================


class TestIrrep:
    """Tests for the Irrep class."""

    def test_construction(self):
        """Test basic Irrep construction."""
        irrep = Irrep(l=2, mult=3)
        assert irrep.l == 2
        assert irrep.mult == 3

    def test_default_multiplicity(self):
        """Test default multiplicity is 1."""
        irrep = Irrep(l=1)
        assert irrep.mult == 1

    def test_dimension(self):
        """Test dimension calculation (2l+1)."""
        assert Irrep(l=0).dim() == 1  # Scalar
        assert Irrep(l=1).dim() == 3  # Vector
        assert Irrep(l=2).dim() == 5  # Rank-2 tensor
        assert Irrep(l=3).dim() == 7
        assert Irrep(l=4).dim() == 9

    def test_mvals(self):
        """Test magnetic quantum numbers."""
        assert Irrep(l=0).mvals() == [0]
        assert Irrep(l=1).mvals() == [-1, 0, 1]
        assert Irrep(l=2).mvals() == [-2, -1, 0, 1, 2]

    def test_offset(self):
        """Test offset calculation."""
        # l=0: offset=0 (no lower degrees)
        assert Irrep(l=0).offset() == 0
        # l=1: offset=1 (dim of l=0)
        assert Irrep(l=1).offset() == 1
        # l=2: offset=1+3=4 (dims of l=0,1)
        assert Irrep(l=2).offset() == 4
        # l=3: offset=1+3+5=9
        assert Irrep(l=3).offset() == 9

    def test_equality(self):
        """Test Irrep equality."""
        assert Irrep(l=2, mult=3) == Irrep(l=2, mult=3)
        assert Irrep(l=2, mult=3) != Irrep(l=2, mult=4)
        assert Irrep(l=2, mult=3) != Irrep(l=3, mult=3)
        assert Irrep(l=1) != "not an irrep"

    def test_hash(self):
        """Test Irrep hashing (for use in sets/dicts)."""
        irreps = {Irrep(l=1, mult=2), Irrep(l=1, mult=2), Irrep(l=2, mult=2)}
        assert len(irreps) == 2

    def test_invalid_degree_negative(self):
        """Test that negative degree raises error."""
        with pytest.raises(ValueError, match="non-negative integer"):
            Irrep(l=-1)

    def test_invalid_degree_float(self):
        """Test that float degree raises error."""
        with pytest.raises(ValueError, match="non-negative integer"):
            Irrep(l=1.5)

    def test_invalid_multiplicity_zero(self):
        """Test that zero multiplicity raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            Irrep(l=1, mult=0)

    def test_invalid_multiplicity_negative(self):
        """Test that negative multiplicity raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            Irrep(l=1, mult=-1)

    def test_raising_operator_shape(self):
        """Test raising operator has correct shape."""
        for l in range(5):
            irrep = Irrep(l=l)
            J_plus = irrep.raising()
            assert J_plus.shape == (irrep.dim(), irrep.dim())

    def test_lowering_operator_shape(self):
        """Test lowering operator has correct shape."""
        for l in range(5):
            irrep = Irrep(l=l)
            J_minus = irrep.lowering()
            assert J_minus.shape == (irrep.dim(), irrep.dim())

    def test_raising_lowering_relation(self):
        """Test relationship between J_+ and J_-."""
        for l in range(1, 4):
            irrep = Irrep(l=l)
            J_plus = irrep.raising()
            J_minus = irrep.lowering()
            # J_- = -(J_+)^T (for the specific convention used)
            assert torch.allclose(J_minus, -J_plus.T, atol=1e-6)

    def test_toreal_unitary(self):
        """Test conversion matrix is unitary."""
        for l in range(4):
            irrep = Irrep(l=l)
            Q = irrep.toreal()
            identity = torch.eye(irrep.dim(), dtype=torch.complex128)
            assert torch.allclose(Q @ Q.conj().T, identity, atol=1e-10)

    def test_generators_shape(self):
        """Test generators have correct shape."""
        for l in range(4):
            irrep = Irrep(l=l)
            gens = irrep._generators()
            assert gens.shape == (3, irrep.dim(), irrep.dim())

    def test_generators_antisymmetric(self):
        """Test generators are antisymmetric (property of so(3))."""
        for l in range(1, 4):
            irrep = Irrep(l=l)
            gens = irrep._generators()
            for i in range(3):
                assert torch.allclose(gens[i], -gens[i].T, atol=1e-6)

    def test_str_representation(self):
        """Test string representation."""
        irrep = Irrep(l=2, mult=3)
        assert "l=2" in str(irrep)
        assert "mult=3" in str(irrep)


# ============================================================================
# TEST: ProductIrrep
# ============================================================================


class TestProductIrrep:
    """Tests for the ProductIrrep class."""

    def test_construction(self):
        """Test ProductIrrep construction."""
        rep1 = Irrep(l=1)
        rep2 = Irrep(l=2)
        prod = ProductIrrep(rep1, rep2)
        assert prod.rep1 == rep1
        assert prod.rep2 == rep2

    def test_lmin_lmax(self):
        """Test min/max degree calculation."""
        # 1 x 2 = 1 + 2 + 3
        prod = ProductIrrep(Irrep(l=1), Irrep(l=2))
        assert prod.lmin == 1  # |1-2| = 1
        assert prod.lmax == 3  # 1+2 = 3

        # 2 x 2 = 0 + 1 + 2 + 3 + 4
        prod = ProductIrrep(Irrep(l=2), Irrep(l=2))
        assert prod.lmin == 0
        assert prod.lmax == 4

        # 0 x 3 = 3
        prod = ProductIrrep(Irrep(l=0), Irrep(l=3))
        assert prod.lmin == 3
        assert prod.lmax == 3

    def test_decomposition_irreps(self):
        """Test irreps in decomposition."""
        # 1 x 1 = 0 + 1 + 2
        prod = ProductIrrep(Irrep(l=1), Irrep(l=1))
        degrees = [rep.l for rep in prod.reps]
        assert degrees == [0, 1, 2]

    def test_dimension(self):
        """Test total dimension equals product of input dims."""
        for l1 in range(4):
            for l2 in range(4):
                prod = ProductIrrep(Irrep(l=l1), Irrep(l=l2))
                expected_dim = (2*l1+1) * (2*l2+1)
                assert prod.dim() == expected_dim

    def test_nreps(self):
        """Test number of irreps in decomposition."""
        # l1 x l2 -> |l1-l2|, ..., l1+l2 has (min(l1,l2)*2 + 1) irreps
        prod = ProductIrrep(Irrep(l=2), Irrep(l=3))
        # 2 x 3 -> 1, 2, 3, 4, 5 (5 irreps)
        assert prod.nreps() == 5

    def test_maxdim(self):
        """Test maximum dimension among decomposition irreps."""
        prod = ProductIrrep(Irrep(l=1), Irrep(l=2))
        # Irreps: l=1,2,3 with dims 3,5,7
        assert prod.maxdim() == 7

    def test_cumdims(self):
        """Test cumulative dimensions."""
        prod = ProductIrrep(Irrep(l=1), Irrep(l=1))
        # l=0,1,2 with dims 1,3,5
        assert prod.cumdims() == [0, 1, 4, 9]

    def test_offset(self):
        """Test offset from degree 0."""
        prod = ProductIrrep(Irrep(l=2), Irrep(l=3))
        # lmin=1, offset = dim(l=0) = 1
        assert prod.offset() == 1


# ============================================================================
# TEST: Repr
# ============================================================================


class TestRepr:
    """Tests for the Repr class."""

    def test_construction(self):
        """Test basic Repr construction."""
        repr = Repr(lvals=[0, 1, 2], mult=4)
        assert repr.lvals == [0, 1, 2]
        assert repr.mult == 4

    def test_default_lvals(self):
        """Test default lvals is [1]."""
        repr = Repr()
        assert repr.lvals == [1]
        assert repr.mult == 1

    def test_dimension(self):
        """Test total dimension calculation."""
        # [0, 1, 2] has dims 1 + 3 + 5 = 9
        assert Repr(lvals=[0, 1, 2]).dim() == 9
        # [1] has dim 3
        assert Repr(lvals=[1]).dim() == 3
        # [0] has dim 1
        assert Repr(lvals=[0]).dim() == 1

    def test_nreps(self):
        """Test number of irreps."""
        assert Repr(lvals=[0, 1, 2]).nreps() == 3
        assert Repr(lvals=[1]).nreps() == 1
        assert Repr(lvals=[0, 1, 1, 2]).nreps() == 4

    def test_lmax(self):
        """Test maximum degree."""
        assert Repr(lvals=[0, 1, 2]).lmax() == 2
        assert Repr(lvals=[1, 3]).lmax() == 3
        assert Repr(lvals=[0]).lmax() == 0

    def test_cumdims(self):
        """Test cumulative dimensions."""
        repr = Repr(lvals=[0, 1, 2])
        # Dims: 1, 3, 5 -> cumdims: 0, 1, 4, 9
        assert repr.cumdims() == [0, 1, 4, 9]

    def test_indices(self):
        """Test indices mapping dims to irreps."""
        repr = Repr(lvals=[0, 1])
        # l=0 has 1 dim, l=1 has 3 dims
        # indices: [0, 1, 1, 1]
        assert repr.indices() == [0, 1, 1, 1]

    def test_iteration(self):
        """Test iterating over irreps."""
        repr = Repr(lvals=[0, 1, 2], mult=2)
        irreps = list(repr)
        assert len(irreps) == 3
        assert irreps[0].l == 0
        assert irreps[1].l == 1
        assert irreps[2].l == 2
        assert all(ir.mult == 2 for ir in irreps)

    def test_equality(self):
        """Test Repr equality."""
        assert Repr(lvals=[0, 1], mult=2) == Repr(lvals=[0, 1], mult=2)
        assert Repr(lvals=[0, 1], mult=2) != Repr(lvals=[0, 1], mult=3)
        assert Repr(lvals=[0, 1]) != Repr(lvals=[1, 2])

    def test_hash(self):
        """Test Repr hashing."""
        reprs = {Repr(lvals=[0, 1]), Repr(lvals=[0, 1]), Repr(lvals=[0, 2])}
        assert len(reprs) == 2

    def test_verify_correct_shape(self):
        """Test verify with correct shape."""
        repr = Repr(lvals=[0, 1], mult=4)
        tensor = torch.randn(10, 4, 4)  # 4 = 1 + 3
        assert repr.verify(tensor) is True

    def test_verify_wrong_mult(self):
        """Test verify with wrong multiplicity."""
        repr = Repr(lvals=[0, 1], mult=4)
        tensor = torch.randn(10, 3, 4)  # Wrong mult
        assert repr.verify(tensor) is False

    def test_verify_wrong_dim(self):
        """Test verify with wrong dimension."""
        repr = Repr(lvals=[0, 1], mult=4)
        tensor = torch.randn(10, 4, 5)  # Wrong dim
        assert repr.verify(tensor) is False

    def test_dot_product(self):
        """Test dot product between spherical tensors."""
        repr = Repr(lvals=[0, 1], mult=2)
        st1 = torch.randn(5, 2, 4)
        st2 = torch.randn(5, 2, 4)
        result = repr.dot(st1, st2)
        # Result has one value per irrep
        assert result.shape == (5, 2, 2)

    def test_dot_product_same_tensor(self):
        """Test dot product of tensor with itself is positive."""
        repr = Repr(lvals=[0, 1], mult=2)
        st = torch.randn(5, 2, 4)
        result = repr.dot(st, st)
        # Dot product with self should be non-negative
        assert (result >= -1e-6).all()

    def test_find_scalar(self):
        """Test finding scalar representations."""
        repr = Repr(lvals=[0, 1, 0, 2])
        count, locs = repr.find_scalar()
        assert count == 2
        assert locs == [0, 4]  # Scalars at index 0 and 4 (after 0+1+3)

    def test_find_scalar_no_scalars(self):
        """Test finding scalars when none exist."""
        repr = Repr(lvals=[1, 2])
        count, locs = repr.find_scalar()
        assert count == 0
        assert locs == []

    def test_invalid_empty_lvals(self):
        """Test that empty lvals raises error."""
        with pytest.raises(ValueError, match="at least one degree"):
            Repr(lvals=[])

    def test_invalid_lvals_type(self):
        """Test that non-list lvals raises error."""
        with pytest.raises(TypeError, match="list or tuple"):
            Repr(lvals=1)

    def test_invalid_multiplicity(self):
        """Test that invalid multiplicity raises error."""
        with pytest.raises(ValueError, match="positive integer"):
            Repr(lvals=[0, 1], mult=0)

    def test_generators_shape(self):
        """Test that generators have correct shape."""
        repr = Repr(lvals=[0, 1, 2])
        gens = repr._generators()
        dim = repr.dim()
        assert gens.shape == (3, dim, dim)

    def test_generators_block_diagonal(self):
        """Test generators are block-diagonal."""
        repr = Repr(lvals=[0, 1, 2])
        gens = repr._generators()

        # Check off-diagonal blocks are zero
        # l=0 at indices [0], l=1 at [1,2,3], l=2 at [4,5,6,7,8]
        # The (0,1:4) block should be zero
        assert torch.allclose(gens[:, 0, 1:4], torch.zeros(3, 3), atol=1e-6)
        # The (1:4, 4:9) block should be zero
        assert torch.allclose(gens[:, 1:4, 4:9], torch.zeros(3, 3, 5), atol=1e-6)


# ============================================================================
# TEST: ProductRepr
# ============================================================================


class TestProductRepr:
    """Tests for the ProductRepr class."""

    def test_construction(self):
        """Test ProductRepr construction."""
        rep1 = Repr(lvals=[0, 1], mult=2)
        rep2 = Repr(lvals=[1, 2], mult=3)
        prod = ProductRepr(rep1, rep2)
        assert prod.rep1 == rep1
        assert prod.rep2 == rep2

    def test_nreps(self):
        """Test number of product irreps."""
        rep1 = Repr(lvals=[0, 1], mult=1)
        rep2 = Repr(lvals=[1], mult=1)
        prod = ProductRepr(rep1, rep2)
        # 0x1 = 1 (1 irrep), 1x1 = 0+1+2 (3 irreps) -> total 4
        assert prod.nreps() == 4

    def test_dim(self):
        """Test total dimension."""
        rep1 = Repr(lvals=[0, 1], mult=1)
        rep2 = Repr(lvals=[1], mult=1)
        prod = ProductRepr(rep1, rep2)
        # 0x1 gives 3, 1x1 gives 9
        assert prod.dim() == 12

    def test_equality(self):
        """Test ProductRepr equality."""
        rep1 = Repr(lvals=[0, 1], mult=1)
        rep2 = Repr(lvals=[1], mult=1)
        prod1 = ProductRepr(rep1, rep2)
        prod2 = ProductRepr(rep1, rep2)
        assert prod1 == prod2

    def test_hash(self):
        """Test ProductRepr hashing."""
        rep1 = Repr(lvals=[0, 1], mult=1)
        rep2 = Repr(lvals=[1], mult=1)
        prods = {ProductRepr(rep1, rep2), ProductRepr(rep1, rep2)}
        assert len(prods) == 1

    def test_lmax(self):
        """Test maximum degree in decomposition."""
        rep1 = Repr(lvals=[1], mult=1)
        rep2 = Repr(lvals=[2], mult=1)
        prod = ProductRepr(rep1, rep2)
        # 1x2 = 1+2+3, lmax = 3
        assert prod.lmax() == 3

    def test_maxdim(self):
        """Test maxdim calculation."""
        rep1 = Repr(lvals=[1], mult=1)
        rep2 = Repr(lvals=[2], mult=1)
        prod = ProductRepr(rep1, rep2)
        # maxdim uses a specific formula for equivariant basis computation
        # Just verify it returns a positive integer
        assert prod.maxdim() > 0
        assert isinstance(prod.maxdim(), int)

    def test_reps_list(self):
        """Test that reps contains ProductIrrep objects."""
        rep1 = Repr(lvals=[0, 1], mult=1)
        rep2 = Repr(lvals=[1], mult=1)
        prod = ProductRepr(rep1, rep2)
        # Should have 2 ProductIrreps (0x1, 1x1)
        assert len(prod.reps) == 2
        assert all(isinstance(r, ProductIrrep) for r in prod.reps)


# ============================================================================
# TEST: Integration
# ============================================================================


class TestRepresentationIntegration:
    """Integration tests for representation classes."""

    def test_repr_with_single_irrep(self):
        """Test Repr behaves correctly with single irrep."""
        repr = Repr(lvals=[2], mult=4)
        assert repr.dim() == 5
        assert repr.nreps() == 1
        assert repr.lmax() == 2

    def test_repr_repeated_lvals(self):
        """Test Repr with repeated l values."""
        repr = Repr(lvals=[1, 1, 1], mult=2)
        assert repr.dim() == 9  # 3 + 3 + 3
        assert repr.nreps() == 3
        assert repr.lmax() == 1

    def test_high_degree_irrep(self):
        """Test high degree irreps work correctly."""
        irrep = Irrep(l=10)
        assert irrep.dim() == 21
        assert len(irrep.mvals()) == 21

        # Generators should still be well-formed
        gens = irrep._generators()
        assert gens.shape == (3, 21, 21)
        assert not torch.isnan(gens).any()

    def test_large_multiplicity(self):
        """Test large multiplicity works."""
        repr = Repr(lvals=[0, 1], mult=100)
        assert repr.mult == 100
        tensor = torch.randn(10, 100, 4)
        assert repr.verify(tensor)
