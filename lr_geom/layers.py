"""Equivariant neural network layers for graph-based architectures.

This module provides SE(3)-equivariant layers for building geometric
deep learning models on graphs. These layers maintain equivariance to
rotations and translations when processing 3D molecular or point cloud data.

The layers that require graph operations use DGL (Deep Graph Library).
DGL is an optional dependency - import will succeed but instantiating
graph-based layers without DGL will raise ImportError.

Classes:
    EquivariantLinear: Linear layer preserving spherical tensor structure
    EquivariantGating: Norm-based gating for spherical tensors
    EquivariantTransition: MLP transition layer for spherical tensors
    EquivariantConvolution: SE(3)-equivariant graph convolution
    EquivariantLayerNorm: Equivariant layer normalization
    GraphAttention: Multi-head attention on graphs (requires DGL)
    EquivariantAttention: Equivariant attention layer (requires DGL)
    EquivariantTransformerBlock: Single transformer block (requires DGL)
    EquivariantTransformer: Full equivariant transformer (requires DGL)
"""
from __future__ import annotations

from copy import copy, deepcopy
from typing import Any

import torch
import torch.nn as nn

from .representations import Repr, ProductRepr
from .equivariant import RepNorm, EquivariantBases, FEATURE_DIM
from .models import RadialWeight

# Lazy import for DGL
_dgl: Any = None
_dgl_available: bool | None = None


def _get_dgl() -> Any:
    """Lazily import DGL and cache the result."""
    global _dgl, _dgl_available

    if _dgl_available is None:
        try:
            import dgl
            _dgl = dgl
            _dgl_available = True
        except ImportError:
            _dgl_available = False

    return _dgl


def is_dgl_available() -> bool:
    """Check if DGL is installed and available."""
    _get_dgl()
    return _dgl_available or False


class EquivariantLinear(nn.Module):
    """Linear layer that preserves spherical tensor structure.

    Applies separate linear transformations to each irrep degree,
    preserving SO(3) equivariance. Only degree-0 (scalar) components
    can have bias terms.

    Args:
        repr: Input representation.
        out_repr: Output representation (must have same lvals as repr).
        dropout: Dropout probability.
        activation: Activation function (applied only to scalars).
        bias: Whether to include bias for scalar components.

    Raises:
        ValueError: If repr and out_repr have different lvals.

    Example:
        >>> repr_in = Repr(lvals=[0, 1, 2], mult=8)
        >>> repr_out = Repr(lvals=[0, 1, 2], mult=16)
        >>> layer = EquivariantLinear(repr_in, repr_out)
        >>> x = torch.randn(32, 8, 9)  # batch, mult, dim
        >>> y = layer(x)  # shape: (32, 16, 9)
    """

    def __init__(
        self: EquivariantLinear,
        repr: Repr,
        out_repr: Repr,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if repr.lvals != out_repr.lvals:
            raise ValueError(
                "EquivariantLinear cannot modify the degrees of a representation. "
                f"Got input lvals={repr.lvals}, output lvals={out_repr.lvals}"
            )

        self.repr = repr

        # Weight matrix for linear transformation
        self.weight = nn.Parameter(
            torch.randn(repr.nreps() * out_repr.mult, repr.mult)
        )

        # Indices for gathering correct degrees
        indices = torch.tensor(repr.indices(), dtype=torch.long)
        self.register_buffer('indices', indices)

        self.expanddims = (1, out_repr.mult, repr.dim())
        self.outdims = (repr.nreps(), out_repr.mult, repr.dim())

        # Bias only for scalar (degree-0) components
        nscalar, scalar_locs = repr.find_scalar()
        self.scalar_locs = scalar_locs

        if nscalar > 0 and bias:
            self.bias = nn.Parameter(
                torch.randn(out_repr.mult, nscalar),
                requires_grad=True,
            )
        else:
            self.bias = None

        self.dropout = nn.Dropout(dropout)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self: EquivariantLinear, f: torch.Tensor) -> torch.Tensor:
        """Apply equivariant linear transformation.

        Args:
            f: Spherical tensor of shape (..., mult, dim).

        Returns:
            Transformed tensor of shape (..., out_mult, dim).
        """
        GATHER_DIM = -3

        *b, _, _ = f.shape

        # Apply linear transformation
        out = (self.weight @ f).view(*b, *self.outdims)

        # Gather components for each degree
        ix = self.indices.expand(*b, *self.expanddims)
        out = out.gather(dim=GATHER_DIM, index=ix).squeeze(GATHER_DIM)

        # Add bias to scalar components
        if self.bias is not None:
            view = out[..., self.scalar_locs]
            out[..., self.scalar_locs] = self.activation(view + self.bias)

        return out


class EquivariantGating(nn.Module):
    """Norm-based gating for spherical tensors.

    Computes norms of each irrep component, processes them through
    a linear layer and sigmoid, then uses the result to gate the
    original tensor. This provides a learnable, equivariant nonlinearity.

    Args:
        repr: The representation of input tensors.
        dropout: Dropout probability.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2], mult=8)
        >>> gate = EquivariantGating(repr)
        >>> x = torch.randn(32, 8, 9)
        >>> y = gate(x)  # shape: (32, 8, 9)
    """

    def __init__(
        self: EquivariantGating,
        repr: Repr,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.repr = repr
        self.norm = RepNorm(repr)

        # Linear layer for processing norms
        self.linear = nn.Linear(
            repr.nreps() * repr.mult,
            repr.nreps() * repr.mult,
        )

        # Indices mapping norms back to full dimension
        self.register_buffer('ix', torch.tensor(repr.indices(), dtype=torch.long))

        self.outdims = (repr.mult, repr.nreps())
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self: EquivariantGating, st: torch.Tensor) -> torch.Tensor:
        """Apply gating to spherical tensor.

        Args:
            st: Spherical tensor of shape (..., mult, dim).

        Returns:
            Gated tensor of shape (..., mult, dim).
        """
        # Compute norms
        norms = self.norm(st)

        *b, _, _ = norms.size()
        norms = norms.flatten(-2, -1)

        # Process through linear layer
        norms = self.linear(norms).view(*b, *self.outdims)

        # Apply activation and dropout
        norms = self.activation(norms)
        norms = self.dropout(norms)

        # Gate the input
        return st * norms[..., self.ix]


class EquivariantTransition(nn.Module):
    """Transition layer for equivariant transformers.

    A feed-forward network that expands to a hidden representation
    and then projects back, with gating nonlinearity.

    Args:
        repr: Input/output representation.
        hidden_repr: Hidden representation (typically 4x multiplicity).

    Example:
        >>> repr = Repr(lvals=[0, 1], mult=8)
        >>> hidden = Repr(lvals=[0, 1], mult=32)
        >>> transition = EquivariantTransition(repr, hidden)
    """

    def __init__(
        self: EquivariantTransition,
        repr: Repr,
        hidden_repr: Repr,
    ) -> None:
        super().__init__()

        self.proj1 = EquivariantLinear(repr, hidden_repr, activation=None)
        self.gating = EquivariantGating(hidden_repr)
        self.proj2 = EquivariantLinear(hidden_repr, repr, activation=None)

    def forward(self: EquivariantTransition, x: torch.Tensor) -> torch.Tensor:
        """Apply transition layer.

        Args:
            x: Input tensor of shape (..., mult, dim).

        Returns:
            Output tensor of shape (..., mult, dim).
        """
        x = self.proj1(x)
        x = self.gating(x)
        return self.proj2(x)


class EquivariantConvolution(nn.Module):
    """Low-rank SE(3)-equivariant convolution.

    Performs equivariant message passing using a low-rank decomposition
    of the full tensor product. Uses radial weights computed from
    invariant edge features and equivariant basis matrices.

    Args:
        repr: ProductRepr specifying input/output representations.
        edge_dim: Dimension of invariant edge features.
        hidden_dim: Hidden dimension for radial weight network.
        dropout: Dropout probability.

    Example:
        >>> repr = ProductRepr(Repr([0, 1]), Repr([0, 1]))
        >>> conv = EquivariantConvolution(repr, edge_dim=16, hidden_dim=32)
    """

    def __init__(
        self: EquivariantConvolution,
        repr: ProductRepr,
        edge_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.rwlin = RadialWeight(
            edge_dim,
            hidden_dim,
            repr,
            repr.rep1.mult,
            repr.rep2.mult,
            dropout,
        )

    def forward(
        self: EquivariantConvolution,
        bases: tuple[torch.Tensor, torch.Tensor],
        edge_feats: torch.Tensor,
        f: torch.Tensor,
        src_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Apply equivariant convolution.

        Args:
            bases: Tuple of (basis1, basis2) from EquivariantBasis.
            edge_feats: Invariant edge features of shape (E, edge_dim).
            f: Node features of shape (N, mult, dim).
            src_idx: Source node indices for each edge, shape (E,).

        Returns:
            Convolved features of shape (E, out_mult, out_dim).
        """
        b1, b2 = bases
        n_edges = edge_feats.size(0)

        # Compute radial weights
        rw = self.rwlin(edge_feats)

        # Gather source features
        f_src = f[src_idx]

        # Apply convolution: f @ b1 -> rw @ ... -> ... @ b2
        tmp = (f_src @ b1).view(n_edges, -1, 1)
        tmp = (rw @ tmp).view(n_edges, -1, b2.size(1))
        return tmp @ b2


class EquivariantLayerNorm(nn.Module):
    """Equivariant layer normalization.

    Normalizes spherical tensors by their irrep norms while
    preserving equivariance. Applies standard LayerNorm to
    the norms across the multiplicity dimension.

    Args:
        repr: The representation of input tensors.
        epsilon: Small constant for numerical stability.

    Example:
        >>> repr = Repr(lvals=[0, 1, 2], mult=8)
        >>> ln = EquivariantLayerNorm(repr)
        >>> x = torch.randn(32, 8, 9)
        >>> y = ln(x)
    """

    EPSILON = 1e-8

    def __init__(
        self: EquivariantLayerNorm,
        repr: Repr,
        epsilon: float = EPSILON,
    ) -> None:
        super().__init__()

        self.norm = RepNorm(repr)
        self.lnorm = nn.LayerNorm(repr.mult)
        self.nonlinearity = nn.Identity()
        self.epsilon = epsilon

        self.register_buffer('ix', torch.tensor(repr.indices(), dtype=torch.long))

    def forward(self: EquivariantLayerNorm, f: torch.Tensor) -> torch.Tensor:
        """Apply equivariant layer normalization.

        Args:
            f: Spherical tensor of shape (..., mult, dim).

        Returns:
            Normalized tensor of shape (..., mult, dim).
        """
        # Compute norms
        norms = self.norm(f)
        *b, h, d = norms.size()

        # Apply LayerNorm across multiplicity
        lnorms = self.lnorm(norms.view(-1, d, h)).view(*b, h, d)
        lnorms = self.nonlinearity(lnorms)

        # Renormalize features
        norms_r = lnorms / (norms + self.epsilon)
        return f * norms_r[..., self.ix]


class GraphAttention(nn.Module):
    """Multi-head attention on graphs using DGL.

    Computes attention-weighted aggregation of values based on
    key-query dot products along graph edges.

    Args:
        hidden_size: Total hidden dimension.
        nheads: Number of attention heads.
        dropout: Dropout probability for attention weights.

    Raises:
        ImportError: If DGL is not installed.

    Example:
        >>> attn = GraphAttention(64, nheads=4)
        >>> # keys, queries: (E, 64), values: (E, 64)
        >>> output = attn(graph, keys, queries, values)  # (N, 64)
    """

    def __init__(
        self: GraphAttention,
        hidden_size: int,
        nheads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        dgl = _get_dgl()
        if dgl is None:
            raise ImportError(
                "DGL is required for GraphAttention. "
                "Install with: pip install dgl"
            )

        if hidden_size % nheads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by nheads ({nheads})"
            )

        self.hidden_size = hidden_size
        self.nheads = nheads
        self.tmpsize = (nheads, hidden_size // nheads)

        self.dropout = nn.Dropout1d(dropout)
        self.temp = hidden_size ** -0.5
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(
        self: GraphAttention,
        graph: Any,  # dgl.DGLGraph
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute graph attention.

        Args:
            graph: DGL graph.
            keys: Key vectors of shape (E, hidden_size).
            queries: Query vectors of shape (E, hidden_size).
            values: Value vectors of shape (E, hidden_size).
            mask: Optional edge mask of shape (E,).
            bias: Optional attention bias of shape (E, nheads).

        Returns:
            Aggregated values of shape (N, hidden_size).
        """
        dgl = _get_dgl()

        # Reshape for multi-head attention
        keys = keys.view(graph.num_edges(), *self.tmpsize)
        queries = queries.view(graph.num_edges(), *self.tmpsize)
        values = values.view(graph.num_edges(), *self.tmpsize)

        # Compute attention scores
        scores = (keys * queries).sum(-1) * self.temp
        scores = self.lrelu(scores)

        # Softmax over edges
        weights = dgl.ops.edge_softmax(graph, scores)
        weights = self.dropout(weights)

        # Apply bias and mask
        if bias is not None:
            weights = weights + bias
        if mask is not None:
            weights = weights * mask[:, None]

        # Compute weighted values and aggregate
        values = weights[..., None] * values
        output = dgl.ops.copy_e_sum(graph, values)

        return output.view(graph.num_nodes(), self.hidden_size)


class EquivariantAttention(nn.Module):
    """SE(3)-equivariant attention layer.

    Combines equivariant convolution with graph attention to
    compute equivariant message passing with attention.

    Args:
        repr: ProductRepr specifying representations.
        edge_dim: Dimension of invariant edge features.
        edge_hidden_dim: Hidden dimension for radial networks.
        nheads: Number of attention heads.
        dropout: Dropout for convolution.
        attn_dropout: Dropout for attention weights.

    Raises:
        ImportError: If DGL is not installed.
    """

    def __init__(
        self: EquivariantAttention,
        repr: ProductRepr,
        edge_dim: int,
        edge_hidden_dim: int,
        nheads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not is_dgl_available():
            raise ImportError(
                "DGL is required for EquivariantAttention. "
                "Install with: pip install dgl"
            )

        # Create representation for keys, queries, values (3x output mult)
        repr_h = deepcopy(repr)
        repr_h.rep2.mult = 3 * repr.rep2.mult

        self.conv = EquivariantConvolution(
            repr_h, edge_dim, edge_hidden_dim, dropout
        )

        self.attn = GraphAttention(
            repr.rep2.mult * repr.rep2.dim(),
            nheads,
            attn_dropout,
        )

        self.proj = EquivariantLinear(repr.rep2, repr.rep2, dropout, activation=None)
        self.outdims = (repr.rep2.mult, repr.rep2.dim())

    def forward(
        self: EquivariantAttention,
        graph: Any,
        basis: tuple[torch.Tensor, torch.Tensor],
        edge_feats: torch.Tensor,
        f: torch.Tensor,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply equivariant attention.

        Args:
            graph: DGL graph.
            basis: Equivariant basis matrices.
            edge_feats: Edge features of shape (E, edge_dim).
            f: Node features of shape (N, mult, dim).
            mask: Optional edge mask.
            bias: Optional attention bias.

        Returns:
            Updated node features of shape (N, mult, dim).
        """
        src, _ = graph.edges()

        # Equivariant convolution to get keys, queries, values
        conv = self.conv(basis, edge_feats, f, src)
        k, q, v = torch.chunk(conv.flatten(-2, -1), 3, dim=-1)

        # Graph attention
        attn_out = self.attn(graph, k, q, v, mask, bias)

        # Reshape and project
        attn_out = attn_out.view(graph.num_nodes(), *self.outdims)
        return self.proj(attn_out)


class EquivariantTransformerBlock(nn.Module):
    """Single block of an equivariant transformer.

    Combines equivariant attention with optional transition layer,
    layer normalization, and residual connections.

    Args:
        repr: ProductRepr for the block.
        edge_dim: Edge feature dimension.
        edge_hidden_dim: Hidden dimension for edge processing.
        nheads: Number of attention heads.
        dropout: Dropout probability.
        attn_dropout: Attention dropout probability.
        transition: Whether to include transition layer.

    Raises:
        ImportError: If DGL is not installed.
    """

    def __init__(
        self: EquivariantTransformerBlock,
        repr: ProductRepr,
        edge_dim: int,
        edge_hidden_dim: int,
        nheads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        transition: bool = False,
    ) -> None:
        super().__init__()

        if not is_dgl_available():
            raise ImportError(
                "DGL is required for EquivariantTransformerBlock. "
                "Install with: pip install dgl"
            )

        self.prepr = repr

        self.attn = EquivariantAttention(
            repr, edge_dim, edge_hidden_dim, nheads, dropout, attn_dropout
        )

        self.ln1 = EquivariantLayerNorm(repr.rep1)

        # Check if skip connection is possible
        deg_match = repr.rep1.lvals == repr.rep2.lvals
        mult_match = repr.rep1.mult == repr.rep2.mult
        self.skip = deg_match and mult_match

        # Optional transition layer
        if transition:
            self.ln2 = EquivariantLayerNorm(repr.rep2)
            hidden_repr = copy(repr.rep2)
            hidden_repr.mult = hidden_repr.mult * 4
            self.transition = EquivariantTransition(repr.rep2, hidden_repr)
        else:
            self.ln2 = None
            self.transition = None

    def forward(
        self: EquivariantTransformerBlock,
        graph: Any,
        basis: tuple[torch.Tensor, torch.Tensor],
        features: torch.Tensor,
        edge_feats: torch.Tensor,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply transformer block.

        Args:
            graph: DGL graph.
            basis: Equivariant basis matrices.
            features: Node features.
            edge_feats: Edge features.
            mask: Optional edge mask.
            bias: Optional attention bias.

        Returns:
            Updated node features.
        """
        # Pre-LN transformer variant
        if self.skip:
            features_tmp = features

        features = self.ln1(features)
        features = self.attn(graph, basis, edge_feats, features, mask, bias)

        if self.skip:
            features = features + features_tmp

        if self.transition is not None:
            if self.skip:
                features_tmp = features
            features = self.ln2(features)
            features = self.transition(features)
            if self.skip:
                features = features + features_tmp

        return features


class EquivariantTransformer(nn.Module):
    """SE(3)-equivariant transformer for geometric graphs.

    A full transformer architecture that processes geometric graphs
    while maintaining SE(3) equivariance. Uses equivariant attention
    and optional transition layers.

    Args:
        in_repr: Input representation.
        out_repr: Output representation.
        hidden_repr: Hidden representation for intermediate layers.
        hidden_layers: Number of hidden transformer blocks.
        edge_dim: Edge feature dimension.
        edge_hidden_dim: Hidden dimension for edge processing.
        nheads: Number of attention heads.
        dropout: Dropout probability.
        attn_dropout: Attention dropout probability.
        transition: Whether to include transition layers.

    Raises:
        ImportError: If DGL is not installed.

    Example:
        >>> in_repr = Repr(lvals=[0, 1], mult=4)
        >>> out_repr = Repr(lvals=[0, 1], mult=1)
        >>> hidden_repr = Repr(lvals=[0, 1], mult=16)
        >>> model = EquivariantTransformer(
        ...     in_repr, out_repr, hidden_repr,
        ...     hidden_layers=4,
        ...     edge_dim=16,
        ...     edge_hidden_dim=32,
        ... )
    """

    def __init__(
        self: EquivariantTransformer,
        in_repr: Repr,
        out_repr: Repr,
        hidden_repr: Repr,
        hidden_layers: int,
        edge_dim: int,
        edge_hidden_dim: int,
        nheads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        transition: bool = False,
    ) -> None:
        super().__init__()

        if not is_dgl_available():
            raise ImportError(
                "DGL is required for EquivariantTransformer. "
                "Install with: pip install dgl"
            )

        self.edge_dim = edge_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.nheads = nheads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.use_transition = transition

        self.in_repr = in_repr
        self.out_repr = out_repr
        self.hidden_repr = hidden_repr

        # Output projection
        out_repr_tmp = copy(out_repr)
        out_repr_tmp.mult = hidden_repr.mult
        self.proj = EquivariantLinear(out_repr_tmp, out_repr, activation=None, bias=True)

        # Build sequence of representations
        reprs = [in_repr] + [hidden_repr] * hidden_layers + [out_repr_tmp]

        # Create layers and track product representations
        layers = []
        preprs = []
        for i in range(len(reprs) - 1):
            repr1, repr2 = reprs[i], reprs[i + 1]
            prepr = ProductRepr(deepcopy(repr1), deepcopy(repr2))
            preprs.append(prepr)
            layers.append(self._construct_layer(prepr))

        self.layers = nn.ModuleList(layers)

        # Compute equivariant bases for all product representations
        self.bases = EquivariantBases(*preprs)

    def _construct_layer(self: EquivariantTransformer, prepr: ProductRepr) -> EquivariantTransformerBlock:
        """Construct a single transformer block."""
        return EquivariantTransformerBlock(
            prepr,
            self.edge_dim,
            self.edge_hidden_dim,
            self.nheads,
            self.dropout,
            self.attn_dropout,
            self.use_transition,
        )

    def forward(
        self: EquivariantTransformer,
        graph: Any,
        coordinates: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply equivariant transformer.

        Args:
            graph: DGL graph.
            coordinates: Node coordinates of shape (N, 3).
            node_features: Node features of shape (N, mult, dim).
            edge_features: Edge features of shape (E, edge_dim).
            mask: Optional edge mask.
            bias: Optional attention bias.

        Returns:
            Output features of shape (N, out_mult, out_dim).

        Raises:
            ValueError: If node_features shape doesn't match in_repr.
        """
        if not self.in_repr.verify(node_features):
            raise ValueError(
                f"Node features shape {node_features.size()} does not match "
                f"input representation (mult={self.in_repr.mult}, dim={self.in_repr.dim()})"
            )

        # Compute basis elements from edge displacements
        src, dst = graph.edges()
        displacements = coordinates[src] - coordinates[dst]
        bases = self.bases(displacements)

        # Pass through transformer layers
        for layer, basis in zip(self.layers, bases):
            node_features = layer(graph, basis, node_features, edge_features, mask, bias)

        # Output projection
        return self.proj(node_features)
