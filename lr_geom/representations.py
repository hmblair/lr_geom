"""SO(3) representation theory for equivariant neural networks.

This module implements the real irreducible representations of SO(3),
the 3D rotation group. These are the mathematical foundation for
building rotation-equivariant neural networks.

Key concepts:
- **Irreducible representation (irrep)**: An irrep of degree l has
  dimension 2l+1 and transforms vectors according to Wigner D-matrices.

- **Wigner D-matrix**: D^l(R) gives the action of rotation R on the
  degree-l irrep. Computed via matrix exponential of the Lie algebra.

- **Tensor product decomposition**: The tensor product of irreps of
  degrees l1 and l2 decomposes as a direct sum of irreps with degrees
  |l1-l2|, |l1-l2|+1, ..., l1+l2 (Clebsch-Gordan decomposition).

Conventions:
- Basis ordering: m = -l, -l+1, ..., l-1, l
- Real spherical harmonics: Y_l^m with Wikipedia conventions
- Generators: Lx, Lz, Ly ordered for consistency with sphericart

Classes:
    Irrep: Single irreducible representation
    ProductIrrep: Tensor product decomposition of two irreps
    Repr: Collection of irreps into a unified representation
    ProductRepr: Tensor product of two representations

References:
    - Wigner, E. P. (1959). Group Theory and its Application to the
      Quantum Mechanics of Atomic Spectra.
    - Thomas, N., et al. (2018). "Tensor field networks"
"""
from __future__ import annotations

import itertools
from typing import Generator, Any

import torch
import torch.nn as nn


class Irrep:
    """A real irreducible representation of SO(3).

    An irreducible representation (irrep) of degree l is a (2l+1)-dimensional
    vector space on which SO(3) acts via Wigner D-matrices. Features
    transforming under this representation are called "spherical tensors
    of rank l" or "type-l features."

    Special cases:
        - l=0: Scalars (1D, invariant under rotation)
        - l=1: Vectors (3D, rotate like coordinates)
        - l=2: Traceless symmetric matrices (5D)

    Args:
        l: The degree (non-negative integer).
        mult: Multiplicity (number of independent copies).

    Attributes:
        l: The degree of the representation.
        mult: Multiplicity of the representation.

    Example:
        >>> irrep = Irrep(l=2)
        >>> irrep.dim()  # 2*2+1 = 5
        5
        >>> irrep.mvals()  # magnetic quantum numbers
        [-2, -1, 0, 1, 2]
    """

    # Class constants for dtype consistency
    REAL_DTYPE = torch.float32
    COMPLEX_DTYPE = torch.complex128

    def __init__(
        self: Irrep,
        l: int,
        mult: int = 1,
    ) -> None:
        if not isinstance(l, int) or l < 0:
            raise ValueError(f"Degree l must be a non-negative integer, got {l}")
        if not isinstance(mult, int) or mult < 1:
            raise ValueError(f"Multiplicity must be a positive integer, got {mult}")

        self.l = l
        self.mult = mult

    def __eq__(
        self: Irrep,
        other: Any,
    ) -> bool:
        """Check if representations have the same degree and multiplicity."""
        if not isinstance(other, Irrep):
            return False
        return self.l == other.l and self.mult == other.mult

    def __hash__(self: Irrep) -> int:
        """Return hash based on degree and multiplicity."""
        return hash((self.l, self.mult))

    def dim(self: Irrep) -> int:
        """Return the dimension of the representation (2l+1)."""
        return 2 * self.l + 1

    def mvals(self: Irrep) -> list[int]:
        """Return the magnetic quantum numbers for this irrep.

        Returns:
            List of m values from -l to l in increasing order.
        """
        return list(range(-self.l, self.l + 1))

    def offset(self: Irrep) -> int:
        """Return the offset of this representation from degree 0.

        This is the sum of dimensions of all irreps with lower degree.
        """
        return sum(Irrep(l).dim() for l in range(self.l))

    def raising(self: Irrep) -> torch.Tensor:
        """Compute the raising operator J_+ for this irreducible representation.

        The raising operator is defined by its action on basis states:
            J_+ |l,m> = sqrt(l(l+1) - m(m+1)) |l,m+1>

        Together with the lowering operator J_-, it generates the Lie algebra
        so(3) via: Lx = (J_+ + J_-)/2, Ly = -i(J_+ - J_-)/2.

        Returns:
            Shape (2l+1, 2l+1), the matrix representation of J_+ in the
            |l,-l>, |l,-l+1>, ..., |l,l> basis.
        """
        j = self.l
        m = torch.arange(-j, j)
        return torch.diag(-torch.sqrt(j * (j + 1) - m * (m + 1)), diagonal=-1)

    def lowering(self: Irrep) -> torch.Tensor:
        """Compute the lowering operator J_- for this irreducible representation.

        The lowering operator is defined by its action on basis states:
            J_- |l,m> = sqrt(l(l+1) - m(m-1)) |l,m-1>

        Returns:
            Shape (2l+1, 2l+1), the matrix representation of J_- in the
            |l,-l>, |l,-l+1>, ..., |l,l> basis.
        """
        j = self.l
        m = torch.arange(-j + 1, j + 1)
        return torch.diag(torch.sqrt(j * (j + 1) - m * (m - 1)), diagonal=1)

    def _generators(self: Irrep) -> torch.Tensor:
        """Return the generators of so(3) in this representation.

        Computes the three generators (Lx, Lz, Ly) of the Lie algebra so(3)
        in the real spherical harmonic basis.

        Returns:
            Shape (3, 2l+1, 2l+1) tensor of generator matrices.
        """
        # Get the raising and lowering operators
        raising = self.raising()
        lowering = self.lowering()

        # Compute the x and y generators
        genx = 0.5 * (raising + lowering)
        geny = -0.5j * (raising - lowering)

        # Get the z generator as a diagonal matrix
        mvals = torch.tensor(self.mvals(), dtype=self.COMPLEX_DTYPE)
        genz = 1j * torch.diag(mvals)

        # Stack the results (order: x, z, y for sphericart compatibility)
        gens = torch.stack([genx, genz, geny], dim=0)

        # Convert from complex to real representations
        Q = self.toreal()
        out = Q.t().conj() @ gens @ Q
        return out.real.to(self.REAL_DTYPE)

    def toreal(self: Irrep) -> torch.Tensor:
        """Get the conversion matrix from complex to real spherical harmonics.

        Returns:
            Unitary matrix Q such that Y_real = Q @ Y_complex.
        """
        SQRT2 = 2 ** -0.5

        q = torch.zeros(self.dim(), self.dim(), dtype=torch.complex128)

        # Fill in the negative degrees (m < 0)
        for m in range(-self.l, 0):
            q[self.l + m, self.l + abs(m)] = SQRT2
            q[self.l + m, self.l - abs(m)] = -1j * SQRT2

        # Fill in the positive degrees (m > 0)
        for m in range(1, self.l + 1):
            q[self.l + m, self.l + abs(m)] = (-1)**m * SQRT2
            q[self.l + m, self.l - abs(m)] = 1j * (-1)**m * SQRT2

        # Fill in the zero degree
        q[self.l, self.l] = 1

        # Scale by the correct complex factor
        q = (-1j)**self.l * q
        return q

    def __str__(self: Irrep) -> str:
        """Return a string representation."""
        return f"Irrep(l={self.l}, mult={self.mult})"


class ProductIrrep:
    """Tensor product decomposition of two irreducible representations.

    When taking the tensor product of two irreps with degrees l1 and l2,
    the result decomposes into a direct sum of irreps with degrees
    |l1-l2|, |l1-l2|+1, ..., l1+l2.

    This is the Clebsch-Gordan decomposition:
        D^{l1} x D^{l2} = D^{|l1-l2|} + D^{|l1-l2|+1} + ... + D^{l1+l2}

    Args:
        rep1: First irreducible representation.
        rep2: Second irreducible representation.

    Attributes:
        lmin: Minimum degree in the decomposition.
        lmax: Maximum degree in the decomposition.
        reps: List of irreps in the decomposition.
    """

    def __init__(
        self: ProductIrrep,
        rep1: Irrep,
        rep2: Irrep,
    ) -> None:
        self.rep1 = rep1
        self.rep2 = rep2

        self.lmin = abs(rep1.l - rep2.l)
        self.lmax = rep1.l + rep2.l

        self.reps = [Irrep(l) for l in range(self.lmin, self.lmax + 1)]

    def dim(self: ProductIrrep) -> int:
        """Return total dimension of the tensor product decomposition."""
        return sum(rep.dim() for rep in self.reps)

    def maxdim(self: ProductIrrep) -> int:
        """Return the maximum dimension among irreps in the decomposition."""
        return max(rep.dim() for rep in self.reps)

    def cumdims(self: ProductIrrep) -> list[int]:
        """Get cumulative dimensions for indexing into the representation."""
        return [
            sum(rep.dim() for rep in self.reps[:l])
            for l in range(self.nreps() + 1)
        ]

    def offset(self: ProductIrrep) -> int:
        """Return the offset of this representation from degree 0.

        This is the sum of dimensions of all irreps with degree < lmin.
        """
        return sum(Irrep(l).dim() for l in range(self.lmin))

    def nreps(self: ProductIrrep) -> int:
        """Return the number of irreps in the decomposition."""
        return len(self.reps)

    def __str__(self: ProductIrrep) -> str:
        """Return a string representation."""
        return f"ProductIrrep({self.rep1.l} x {self.rep2.l})"


class Repr(nn.Module):
    """A collection of irreducible representations.

    Combines multiple irreps into a single representation, enabling
    computation of Wigner D-matrices for the combined representation.

    The representation is specified by a list of degrees. Each degree l
    contributes a (2l+1)-dimensional subspace.

    Args:
        lvals: List of degrees to include. Defaults to [1] (vectors only).
        mult: Multiplicity for all irreps.

    Attributes:
        irreps: List of Irrep objects.
        lvals: List of degrees.
        generators: Pre-computed Lie algebra generators.

    Example:
        >>> # Representation with scalars, vectors, and rank-2 tensors
        >>> repr = Repr(lvals=[0, 1, 2])
        >>> repr.dim()  # 1 + 3 + 5 = 9
        9
        >>> # Get rotation matrix for 45 degrees around z-axis
        >>> axis = torch.tensor([0., 0., 1.])
        >>> angle = torch.tensor(3.14159 / 4)
        >>> D = repr.rot(axis, angle)
    """

    def __init__(
        self: Repr,
        lvals: list[int] | None = None,
        mult: int = 1,
    ) -> None:
        super().__init__()

        if lvals is None:
            lvals = [1]

        if not isinstance(lvals, (list, tuple)):
            raise TypeError(f"lvals must be a list or tuple, got {type(lvals).__name__}")
        if len(lvals) == 0:
            raise ValueError("lvals must contain at least one degree")
        if not isinstance(mult, int) or mult < 1:
            raise ValueError(f"Multiplicity must be a positive integer, got {mult}")

        # Irrep validation will catch invalid l values
        self.irreps = [Irrep(l, mult) for l in lvals]
        self.lvals = [irrep.l for irrep in self.irreps]
        self.mult = mult

        # Pre-compute cumulative dimensions for indexing
        self._cumdims = [
            sum(rep.dim() for rep in self.irreps[:i])
            for i in range(len(self.irreps) + 1)
        ]

        # Pre-compute indices mapping each dimension to its irrep index
        self._indices = [
            repnum
            for repnum, irrep in enumerate(self.irreps)
            for _ in range(irrep.dim())
        ]

        # Pre-compute the so(3) generators and reshape for efficient computation
        self.register_buffer(
            'generators',
            self._generators().view(3, -1)
        )
        self.perm = self._reorder_generators()

        # Register indices as buffer for efficient dot product computation
        self.register_buffer(
            '_indices_tensor',
            torch.tensor(self._indices, dtype=torch.long)
        )

    def nreps(self: Repr) -> int:
        """Return the number of irreducible representations."""
        return len(self.irreps)

    def __iter__(self: Repr) -> Generator[Irrep, None, None]:
        """Iterate over the irreducible representations."""
        yield from self.irreps

    def __eq__(self: Repr, other: Any) -> bool:
        """Check if representations have the same lvals and multiplicity."""
        if not isinstance(other, Repr):
            return False
        return self.lvals == other.lvals and self.mult == other.mult

    def __hash__(self: Repr) -> int:
        """Return hash based on lvals and multiplicity."""
        return hash((tuple(self.lvals), self.mult))

    def dim(self: Repr) -> int:
        """Get total dimension of the representation."""
        return sum(irrep.dim() for irrep in self)

    def lmax(self: Repr) -> int:
        """Get the largest degree among all irreps."""
        return max(irrep.l for irrep in self)

    def cumdims(self: Repr) -> list[int]:
        """Get cumulative dimensions for indexing into subspaces."""
        return self._cumdims

    def offsets(self: Repr) -> list[int]:
        """Return the offset of each irrep from degree 0."""
        return [rep.offset() for rep in self]

    def indices(self: Repr) -> list[int]:
        """Return irrep index for each dimension.

        Returns:
            List of length dim() where entry i gives the irrep index
            that dimension i belongs to.
        """
        return self._indices

    def verify(self: Repr, st: torch.Tensor) -> bool:
        """Check if a tensor has the correct shape for this representation.

        Args:
            st: Spherical tensor to verify.

        Returns:
            True if shape matches (..., mult, dim()).
        """
        correct_mult = st.size(-2) == self.mult
        correct_dim = st.size(-1) == self.dim()
        return correct_mult and correct_dim

    def dot(
        self: Repr,
        st1: torch.Tensor,
        st2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dot product between corresponding irreps.

        Takes the element-wise product of two spherical tensors and
        sums within each irrep subspace.

        Args:
            st1: First spherical tensor.
            st2: Second spherical tensor.

        Returns:
            Tensor with one value per irrep.
        """
        # Use pre-computed indices tensor (registered as buffer)
        ix = self._indices_tensor.to(st1.device)
        product = st1 * st2

        # Use native PyTorch scatter_add instead of torch_scatter
        num_irreps = self.nreps()
        result_shape = product.shape[:-1] + (num_irreps,)
        result = torch.zeros(result_shape, device=product.device, dtype=product.dtype)

        # Expand indices to match product shape for scatter
        ix_expanded = ix.expand(product.shape)
        result.scatter_add_(-1, ix_expanded, product)

        return result

    def find_scalar(self: Repr) -> tuple[int, list[int]]:
        """Find all degree-0 (scalar) representations and their locations.

        Returns:
            Tuple of (count, locations) where count is the number of
            scalar irreps and locations is a list of their starting indices.
        """
        nreps = 0
        locs = []
        cumdim = 0
        for repr in self.irreps:
            if repr.l == 0:
                nreps += 1
                locs.append(cumdim)
            cumdim += repr.dim()
        return nreps, locs

    def _generators(self: Repr) -> torch.Tensor:
        """Compute the so(3) generators for the full representation.

        Returns:
            Shape (3, dim, dim) tensor of generator matrices.
        """
        NUM_GENS = 3
        gens = torch.zeros(NUM_GENS, self.dim(), self.dim())

        # Fill in block-diagonal with generators from each irrep
        cumdim = 0
        for irrep in self.irreps:
            gens[
                ...,
                cumdim: cumdim + irrep.dim(),
                cumdim: cumdim + irrep.dim(),
            ] = irrep._generators()
            cumdim += irrep.dim()

        return gens

    def _reorder_generators(self: Repr) -> torch.Tensor:
        """Get permutation matrix for reordering rotation generators.

        Returns a permutation that makes rank-1 and rank-2 equivariant
        matrices contiguous, useful for certain equivariant operations.
        """
        p = torch.eye(self.dim())
        perm1 = []
        perm2 = []
        ix = 0

        for irrep in self.irreps:
            perm1.append(ix + irrep.l)
            for jx in range(irrep.l):
                perm2.append(ix + jx)
                perm2.append(ix + irrep.dim() - jx - 1)
            ix += irrep.dim()

        perm = perm1 + perm2
        return p[perm]

    def rot(
        self: Repr,
        axis: torch.Tensor,
        angle: torch.Tensor,
        perm: bool = False,
    ) -> torch.Tensor:
        """Compute the Wigner D-matrix for a rotation.

        Given a rotation specified by axis and angle, computes the
        representation matrix that transforms spherical tensors.

        Args:
            axis: Rotation axis of shape (..., 3), should be normalized.
            angle: Rotation angle in radians of shape (...).
            perm: If True, apply the generator reordering permutation.

        Returns:
            Wigner D-matrix of shape (..., dim, dim).

        Note:
            For invalid axes (e.g., zero vector), rotation matrices are
            set to zero except for degree-0 blocks which remain identity
            (scalars are invariant under all rotations).
        """
        # Weight the generators of so(3) by the axis values
        *b, _ = axis.size()
        gens = (axis @ self.generators).view(*b, self.dim(), self.dim())

        # Multiply by the angle and exponentiate to move to SO(3)
        rot = torch.linalg.matrix_exp(angle[..., None, None] * gens)

        if perm:
            rot = rot @ self.perm.t()

        # Zero out NaN values from invalid axes
        rot = torch.nan_to_num(rot, 0.0)

        # Restore identity for degree-0 (scalar) irreps
        # Scalars are invariant under rotation, including "invalid" rotations
        cdims = self.cumdims()
        for i, irrep in enumerate(self.irreps):
            if irrep.l == 0:
                rot[..., cdims[i], cdims[i]] = 1.0

        return rot

    def basis(self: Repr, x: torch.Tensor) -> torch.Tensor:
        """Compute equivariant basis elements for input points.

        Args:
            x: Points of shape (N, 3).

        Returns:
            Basis matrices of shape (N, dim, dim).
        """
        p = torch.tensor([0, 1, 0], device=x.device, dtype=x.dtype)

        # Find the axis to rotate about
        axis = torch.linalg.cross(x, p[None, :])
        axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True)

        # Find the angle to rotate by
        angle = -torch.arccos(x[:, 1] / torch.linalg.norm(x, dim=-1))

        # Compute the basis via the Wigner matrices
        return self.rot(axis, angle)

    def __str__(self: Repr) -> str:
        """Return a string representation."""
        degrees = ', '.join(str(rep.l) for rep in self.irreps)
        return f"Repr(lvals=[{degrees}])"


class ProductRepr:
    """Tensor product of two representations.

    Computes the tensor product structure and irreducible decomposition
    for the product of two Repr objects.

    Args:
        rep1: First representation.
        rep2: Second representation.

    Attributes:
        rep1: First input representation.
        rep2: Second input representation.
        reps: List of ProductIrrep objects for each pair of irreps.
    """

    def __init__(
        self: ProductRepr,
        rep1: Repr,
        rep2: Repr,
    ) -> None:
        self.rep1 = rep1
        self.rep2 = rep2

        self.reps = [
            ProductIrrep(irrep1, irrep2)
            for irrep1 in rep1
            for irrep2 in rep2
        ]
        # Store as list to allow multiple iterations (generators exhaust after one use)
        self.offsets = list(itertools.product(
            rep1.cumdims()[:-1],
            rep2.cumdims()[:-1],
        ))

    def dim(self: ProductRepr) -> int:
        """Get total dimension of the tensor product."""
        return sum(rep.dim() for rep in self.reps)

    def __eq__(self: ProductRepr, other: Any) -> bool:
        """Check if both representations are equal."""
        if not isinstance(other, ProductRepr):
            return False
        return self.rep1 == other.rep1 and self.rep2 == other.rep2

    def __hash__(self: ProductRepr) -> int:
        """Return hash based on component representations."""
        return hash((hash(self.rep1), hash(self.rep2)))

    def lmax(self: ProductRepr) -> int:
        """Get the largest degree in the decomposition."""
        return max(rep.lmax for rep in self.reps)

    def maxdim(self: ProductRepr) -> int:
        """Get dimension of the largest irrep in the decomposition."""
        return max(rep.dim() + rep.lmin ** 2 for rep in self.reps)

    def nreps(self: ProductRepr) -> int:
        """Return total number of irreps in the tensor product."""
        return sum(rep.nreps() for rep in self.reps)
