"""Equivariant neural network layers.

This module provides neural network layers that are equivariant to
3D rotations (SO(3)). These are the building blocks for constructing
rotation-equivariant networks for 3D data like molecules and point clouds.

Classes:
    RepNorm: Compute norms of spherical tensor components
    SphericalHarmonic: Compute spherical harmonics features
    RadialBasisFunctions: Learnable radial basis function expansion
    EquivariantBasis: Compute equivariant basis matrices for a product representation
    EquivariantBases: Compute equivariant bases for multiple product representations
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .representations import Repr, ProductRepr

# Dimension constants for spherical tensor indexing
FEATURE_DIM = -2  # Multiplicity dimension
REPR_DIM = -1     # Representation dimension

# Lazy import for sphericart
_sphericart: Any = None
_sphericart_available: bool | None = None


def _get_sphericart() -> Any:
    """Lazily import sphericart and cache the result."""
    global _sphericart, _sphericart_available

    if _sphericart_available is None:
        try:
            import sphericart.torch as sc
            _sphericart = sc
            _sphericart_available = True
        except ImportError:
            _sphericart_available = False

    return _sphericart


class RepNorm(nn.Module):
    """Compute norms of spherical tensor components.

    For a spherical tensor with multiple irrep components, computes
    the norm of each component separately. This produces rotation-invariant
    features that can be used for gating or as input to invariant networks.

    Args:
        repr: The representation specifying the tensor structure.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2])  # scalar + vector + rank-2
        >>> norm = RepNorm(repr)
        >>> st = torch.randn(32, 9)  # batch of spherical tensors
        >>> norms = norm(st)  # shape: (32, 3) - one norm per irrep
    """

    def __init__(self: RepNorm, repr: Repr) -> None:
        super().__init__()
        self.nreps = repr.nreps()
        self.cdims = repr.cumdims()

    def forward(self: RepNorm, st: torch.Tensor) -> torch.Tensor:
        """Compute the norm of each irrep component.

        Args:
            st: Spherical tensor of shape (..., dim).

        Returns:
            Norms of shape (..., nreps).
        """
        norms = torch.zeros(
            st.shape[:-1] + (self.nreps,),
            device=st.device,
            dtype=st.dtype,
        )

        for i in range(self.nreps):
            norms[..., i] = st[
                ..., self.cdims[i]:self.cdims[i + 1],
            ].norm(dim=-1)  # Fixed: was REPR_DIM (undefined)

        return norms


class SphericalHarmonic(nn.Module):
    """Compute spherical harmonic features for 3D coordinates.

    Spherical harmonics form a complete orthonormal basis for functions
    on the sphere. They are the natural features for SO(3)-equivariant
    networks, as they transform predictably under rotations.

    Uses the sphericart library for efficient computation.

    Args:
        lmax: Maximum degree of spherical harmonics to compute.
            Total features = (lmax + 1)^2.
        normalized: If True, compute normalized spherical harmonics.

    Raises:
        ImportError: If sphericart is not installed.

    Example:
        >>> sh = SphericalHarmonic(lmax=3)
        >>> coords = torch.randn(100, 3)
        >>> features = sh(coords)  # shape: (100, 16)
    """

    def __init__(
        self: SphericalHarmonic,
        lmax: int,
        normalized: bool = True,
    ) -> None:
        super().__init__()

        sc = _get_sphericart()
        if sc is None:
            raise ImportError(
                "sphericart is required for SphericalHarmonic. "
                "Install with: pip install sphericart"
            )

        self.sh = sc.SphericalHarmonics(lmax, normalized)
        self.lmax = lmax

        # Index permutation for sphericart coordinate convention
        self.ix = torch.tensor([2, 0, 1], dtype=torch.int64)

    def forward(self: SphericalHarmonic, x: torch.Tensor) -> torch.Tensor:
        """Compute spherical harmonic features for points.

        Args:
            x: Coordinates of shape (..., N, 3).

        Returns:
            Spherical harmonic features of shape (..., N, (lmax+1)^2).
            NaN values (from zero vectors) are replaced with zeros.
        """
        *b, n, _ = x.shape
        x = x.view(-1, 3)

        # Permute coordinates for sphericart convention
        x = x[:, self.ix]

        # Handle dtype (sphericart only supports float32/64)
        dtype = x.dtype
        if dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        # Compute spherical harmonics
        sh = self.sh.compute(x)

        # Restore original dtype and handle NaN
        sh = sh.to(dtype)
        sh = torch.nan_to_num(sh, nan=0.0)

        return sh.view(*b, n, -1)

    def pairwise(self: SphericalHarmonic, x: torch.Tensor) -> torch.Tensor:
        """Compute spherical harmonics for pairwise relative positions.

        Args:
            x: Point cloud of shape (N, 3).

        Returns:
            Pairwise features of shape (N, N, (lmax+1)^2).
        """
        # Compute pairwise relative positions
        pairwise = x[:, None] - x[None, :]

        # Compute spherical harmonics
        sh = self(pairwise)

        return torch.nan_to_num(sh, nan=0.0)


class RadialBasisFunctions(nn.Module):
    """Learnable radial basis function expansion.

    Expands scalar distance values into a set of learnable Gaussian
    basis functions. Useful for encoding distances in equivariant
    networks where edge features should be rotation-invariant.

    Args:
        num_functions: Number of basis functions.
        r_min: Minimum distance for center initialization.
        r_max: Maximum distance for center initialization.

    Attributes:
        mu: Learnable centers of the Gaussians, initialized evenly spaced.
        sigma: Learnable widths of the Gaussians, initialized based on spacing.

    Example:
        >>> rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0)
        >>> distances = torch.rand(100) * 10  # distances in [0, 10]
        >>> features = rbf(distances)  # shape: (100, 16)
    """

    def __init__(
        self: RadialBasisFunctions,
        num_functions: int,
        r_min: float = 0.0,
        r_max: float = 10.0,
    ) -> None:
        super().__init__()

        # Initialize centers evenly spaced in [r_min, r_max]
        self.mu = nn.Parameter(
            torch.linspace(r_min, r_max, num_functions),
            requires_grad=True,
        )

        # Initialize widths based on spacing between centers
        spacing = (r_max - r_min) / max(num_functions - 1, 1)
        self.sigma = nn.Parameter(
            torch.full((num_functions,), spacing),
            requires_grad=True,
        )

    def forward(self: RadialBasisFunctions, x: torch.Tensor) -> torch.Tensor:
        """Evaluate radial basis functions at input values.

        Args:
            x: Input distances of shape (...).

        Returns:
            Basis function values of shape (..., num_functions).
        """
        # Standard Gaussian RBF: exp(-((x - mu) / sigma)^2)
        diff = (x[..., None] - self.mu) / self.sigma.abs().clamp(min=1e-6)
        return torch.exp(-diff ** 2)


class EquivariantBasis(nn.Module):
    """Compute equivariant basis matrices for tensor product operations.

    Given 3D displacement vectors, computes the spherical harmonic coefficients
    organized as matrices suitable for equivariant convolutions. The output
    consists of two coefficient tensors that can be used in low-rank
    tensor product contractions.

    Args:
        repr: ProductRepr specifying the input and output representations.

    Example:
        >>> repr = ProductRepr(Repr([0, 1]), Repr([0, 1]))
        >>> basis = EquivariantBasis(repr)
        >>> displacements = torch.randn(100, 3)  # 100 edge vectors
        >>> coeff1, coeff2 = basis(displacements)
    """

    def __init__(self: EquivariantBasis, repr: ProductRepr) -> None:
        super().__init__()

        self.outdims1 = (repr.rep1.dim(), repr.rep1.nreps())
        self.outdims2 = (repr.rep2.nreps(), repr.rep2.dim())

        # Spherical harmonic calculator
        self.sh = SphericalHarmonic(repr.lmax())
        self.repr = repr

    def forward(
        self: EquivariantBasis,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute equivariant basis coefficients.

        Args:
            x: Displacement vectors of shape (N, 3).

        Returns:
            Tuple of (coeff1, coeff2) where:
            - coeff1 has shape (N, rep1.dim(), rep1.nreps())
            - coeff2 has shape (N, rep2.nreps(), rep2.dim())
        """
        # Get spherical harmonic features
        sh = self.sh(x)
        sh = torch.nan_to_num(sh, nan=0.0)

        # Build coefficient matrices
        coeff1 = torch.zeros(x.size(0), *self.outdims1, device=x.device)
        coeff2 = torch.zeros(x.size(0), *self.outdims2, device=x.device)

        for j in range(self.repr.rep1.nreps()):
            low = j ** 2
            high = (j + 1) ** 2
            coeff1[..., low:high, j] = sh[..., low:high]

        for j in range(self.repr.rep2.nreps()):
            low = j ** 2
            high = (j + 1) ** 2
            coeff2[..., j, low:high] = sh[..., low:high]

        return coeff1, coeff2


class EquivariantBases(nn.Module):
    """Compute equivariant bases for multiple product representations.

    Efficiently computes basis matrices for multiple ProductRepr objects,
    avoiding redundant computation for identical representations.

    Args:
        *reprs: Variable number of ProductRepr objects.

    Example:
        >>> repr1 = ProductRepr(Repr([0, 1]), Repr([0, 1]))
        >>> repr2 = ProductRepr(Repr([0, 1, 2]), Repr([0, 1, 2]))
        >>> bases = EquivariantBases(repr1, repr2)
        >>> displacements = torch.randn(100, 3)
        >>> basis_list = bases(displacements)  # tuple of basis pairs
    """

    def __init__(self: EquivariantBases, *reprs: ProductRepr) -> None:
        super().__init__()

        # Deduplicate representations to avoid redundant computation
        self.unique_reprs: list[ProductRepr] = []
        self.comps = nn.ModuleList()
        self.repr_ix: list[int] = []

        repr_count = -1
        for repr in reprs:
            if repr not in self.unique_reprs:
                self.unique_reprs.append(repr)
                self.comps.append(EquivariantBasis(repr))
                repr_count += 1
            self.repr_ix.append(repr_count)

    def forward(
        self: EquivariantBases,
        x: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Compute basis matrices for all representations.

        Args:
            x: Displacement vectors of shape (N, 3).

        Returns:
            Tuple of basis pairs, one for each input ProductRepr.
        """
        # Compute unique bases
        ms = [comp(x) for comp in self.comps]
        # Expand to required outputs without recomputation
        return tuple(ms[ix] for ix in self.repr_ix)
