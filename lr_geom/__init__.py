"""Low-Rank Geometric Deep Learning (lr_geom).

A PyTorch library for SO(3)-equivariant neural networks using low-rank
tensor product decompositions. Designed for molecular modeling, point
cloud processing, and other 3D geometric learning tasks.

Key Concepts:
    - **SO(3) Equivariance**: Functions f where f(Rx) = Rf(x) for any rotation R
    - **Spherical Tensors**: Features transforming under irreducible representations
    - **Low-Rank Decomposition**: Efficient tensor product via basis matrices

Modules:
    alignment: Point cloud alignment (Kabsch algorithm, RMSD)
    representations: SO(3) representation theory (irreps, Wigner D-matrices)
    equivariant: Equivariant primitives (spherical harmonics, basis functions)
    layers: Equivariant neural network layers (convolution, attention)
    models: Pre-built geometric models

Optional Dependencies:
    - sphericart: Required for SphericalHarmonic
    - dgl: Required for graph-based layers (GraphAttention, EquivariantTransformer)

Example:
    >>> import lr_geom as lg
    >>>
    >>> # Create SO(3) representation
    >>> repr = lg.Repr(lvals=[0, 1, 2], mult=8)
    >>>
    >>> # Build equivariant layers
    >>> linear = lg.EquivariantLinear(repr, repr)
    >>> ln = lg.EquivariantLayerNorm(repr)
"""

__version__ = "0.1.0"

# Alignment and scoring
from .alignment import (
    rmsd,
    get_kabsch_rotation_matrix,
    kabsch_align,
    RMSD,
    graph_laplacian,
    gnm_correlations,
    gnm_variances,
)

# Representation theory
from .representations import (
    Irrep,
    ProductIrrep,
    Repr,
    ProductRepr,
)

# Equivariant primitives
from .equivariant import (
    FEATURE_DIM,
    REPR_DIM,
    RepNorm,
    SphericalHarmonic,
    RadialBasisFunctions,
    EquivariantBasis,
    EquivariantBases,
)

# Models
from .models import GNMA

# Equivariant layers
from .layers import (
    is_dgl_available,
    RadialWeight,
    EquivariantLinear,
    EquivariantGating,
    EquivariantTransition,
    EquivariantConvolution,
    EquivariantLayerNorm,
    GraphAttention,
    EquivariantAttention,
    EquivariantTransformerBlock,
    EquivariantTransformer,
)

__all__ = [
    # Version
    "__version__",
    # Constants
    "FEATURE_DIM",
    "REPR_DIM",
    # Alignment
    "rmsd",
    "get_kabsch_rotation_matrix",
    "kabsch_align",
    "RMSD",
    "graph_laplacian",
    "gnm_correlations",
    "gnm_variances",
    # Representations
    "Irrep",
    "ProductIrrep",
    "Repr",
    "ProductRepr",
    # Equivariant primitives
    "RepNorm",
    "SphericalHarmonic",
    "RadialBasisFunctions",
    "EquivariantBasis",
    "EquivariantBases",
    # Models
    "GNMA",
    # Layers
    "is_dgl_available",
    "RadialWeight",
    "EquivariantLinear",
    "EquivariantGating",
    "EquivariantTransition",
    "EquivariantConvolution",
    "EquivariantLayerNorm",
    "GraphAttention",
    "EquivariantAttention",
    "EquivariantTransformerBlock",
    "EquivariantTransformer",
]
