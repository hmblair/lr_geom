"""Low-Rank Geometric Deep Learning (lr_geom).

A PyTorch library for SO(3)-equivariant neural networks using low-rank
tensor product decompositions. Designed for molecular modeling, point
cloud processing, and other 3D geometric learning tasks.

Key Concepts:
    - **SO(3) Equivariance**: Functions f where f(Rx) = Rf(x) for any rotation R
    - **Spherical Tensors**: Features transforming under irreducible representations
    - **Low-Rank Decomposition**: Efficient tensor product via basis matrices

Modules:
    representations: SO(3) representation theory (irreps, Wigner D-matrices)
    equivariant: Equivariant primitives (spherical harmonics, basis functions)
    layers: Equivariant neural network layers (convolution, attention)
    models: Pre-built geometric models (GNM functions)
    vae: SO(3)-equivariant variational autoencoder
    config: Configuration system for experiments

Optional Dependencies:
    - sphericart: Required for SphericalHarmonic

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

# Models (includes GNM functions)
from .models import (
    GNMA,
    graph_laplacian,
    gnm_correlations,
    gnm_variances,
)

# VAE
from .vae import (
    EquivariantVAE,
    VariationalHead,
    reparameterize,
    kl_divergence,
)

# Equivariant layers
from .layers import (
    build_knn_graph,
    RadialWeight,
    EquivariantLinear,
    EquivariantGating,
    EquivariantTransition,
    EquivariantConvolution,
    EquivariantLayerNorm,
    Attention,
    EquivariantAttention,
    EquivariantTransformerBlock,
    EquivariantTransformer,
)

# Configuration
from .config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    ExperimentConfig,
    load_config,
    save_config,
)

__all__ = [
    # Version
    "__version__",
    # Constants
    "FEATURE_DIM",
    "REPR_DIM",
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
    # Models and GNM
    "GNMA",
    "graph_laplacian",
    "gnm_correlations",
    "gnm_variances",
    # VAE
    "EquivariantVAE",
    "VariationalHead",
    "reparameterize",
    "kl_divergence",
    # Layers
    "build_knn_graph",
    "RadialWeight",
    "EquivariantLinear",
    "EquivariantGating",
    "EquivariantTransition",
    "EquivariantConvolution",
    "EquivariantLayerNorm",
    "Attention",
    "EquivariantAttention",
    "EquivariantTransformerBlock",
    "EquivariantTransformer",
    # Configuration
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "ExperimentConfig",
    "load_config",
    "save_config",
]
