"""Equivariant neural network layers for geometric deep learning.

This module provides SE(3)-equivariant layers for building geometric
deep learning models. These layers maintain equivariance to rotations
and translations when processing 3D molecular or point cloud data.

Uses k-NN based sparse attention with O(N*k) complexity.

Classes:
    RadialWeight: Neural network for computing tensor product weights
    EquivariantLinear: Linear layer preserving spherical tensor structure
    EquivariantGating: Norm-based gating for spherical tensors
    EquivariantTransition: MLP transition layer for spherical tensors
    EquivariantConvolution: SE(3)-equivariant convolution
    EquivariantLayerNorm: Equivariant layer normalization
    Attention: k-NN based multi-head attention
    EquivariantAttention: SE(3)-equivariant sparse attention
    EquivariantTransformerBlock: Transformer block with sparse attention
    EquivariantTransformer: Full equivariant transformer
"""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .representations import Repr, ProductRepr
from .equivariant import RepNorm, EquivariantBases, RadialBasisFunctions


def _init_weights(module: nn.Module) -> None:
    """Initialize weights using Xavier uniform for linear layers."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_knn_graph(coordinates: torch.Tensor, k: int) -> torch.Tensor:
    """Build k-nearest neighbor graph from 3D coordinates.

    Args:
        coordinates: Node coordinates of shape (N, 3).
        k: Number of nearest neighbors per node.

    Returns:
        Neighbor indices of shape (N, k), where neighbor_idx[i] contains
        the k nearest neighbor indices for node i (excluding self).

    Example:
        >>> coords = torch.randn(100, 3)
        >>> neighbor_idx = build_knn_graph(coords, k=16)
        >>> neighbor_idx.shape
        torch.Size([100, 16])
    """
    N = coordinates.size(0)

    # Handle edge case where N <= k
    if N <= k + 1:
        # Use all other nodes as neighbors
        all_idx = torch.arange(N, device=coordinates.device)
        neighbor_idx = all_idx.unsqueeze(0).expand(N, -1)
        # Exclude self: create mask and gather non-self indices
        mask = neighbor_idx != torch.arange(N, device=coordinates.device).unsqueeze(1)
        # Pad if necessary
        result = torch.zeros(N, k, dtype=torch.long, device=coordinates.device)
        for i in range(N):
            others = neighbor_idx[i, mask[i]]
            result[i, :len(others)] = others
            if len(others) < k:
                # Repeat last neighbor to fill
                result[i, len(others):] = others[-1] if len(others) > 0 else 0
        return result

    # Compute pairwise squared distances
    dists = torch.cdist(coordinates, coordinates)  # (N, N)

    # Get k+1 nearest neighbors (including self at index 0)
    _, indices = dists.topk(k + 1, largest=False, dim=-1)  # (N, k+1)

    # Exclude self (first column after sorting by distance)
    return indices[:, 1:]  # (N, k)


class RadialWeight(nn.Module):
    """Compute tensor product weights from invariant edge features.

    A two-layer neural network that maps edge features to weights
    for tensor product contractions. Used in equivariant message
    passing networks.

    Args:
        edge_dim: Dimension of input edge features.
        hidden_dim: Hidden layer dimension.
        repr: ProductRepr specifying the tensor product structure.
        in_dim: Input multiplicity.
        out_dim: Output multiplicity.
        dropout: Dropout probability.

    Example:
        >>> repr = ProductRepr(Repr([1]), Repr([1]))
        >>> weight_net = RadialWeight(16, 32, repr, 8, 8)
        >>> edge_features = torch.randn(100, 16)
        >>> weights = weight_net(edge_features)
    """

    def __init__(
        self: RadialWeight,
        edge_dim: int,
        hidden_dim: int,
        repr: ProductRepr,
        in_dim: int,
        out_dim: int,
        dropout: float = 0,
    ) -> None:
        super().__init__()

        self.nl1 = repr.rep1.nreps()
        self.nl2 = repr.rep2.nreps()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.out_dim_flat = self.nl1 * self.nl2 * in_dim * out_dim

        self.layer1 = nn.Linear(edge_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, self.out_dim_flat)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.apply(_init_weights)

    def forward(self: RadialWeight, x: torch.Tensor) -> torch.Tensor:
        """Compute weights from edge features.

        Args:
            x: Edge features of shape (..., edge_dim).

        Returns:
            Weights of shape (..., nl2 * out_dim, nl1 * in_dim).
        """
        *b, _ = x.size()

        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)

        return self.layer2(x).view(
            *b, self.nl2 * self.out_dim,
            self.nl1 * self.in_dim,
        )


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
            torch.empty(repr.nreps() * out_repr.mult, repr.mult)
        )
        nn.init.xavier_uniform_(self.weight)

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
                torch.zeros(out_repr.mult, nscalar),
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
        out = self.dropout(out)

        # Gather components for each degree
        ix = self.indices.expand(*b, *self.expanddims)
        out = out.gather(dim=GATHER_DIM, index=ix).squeeze(GATHER_DIM)

        # Add bias and activation to scalar components
        if self.bias is not None:
            out = out.clone()
            out[..., self.scalar_locs] = self.activation(
                out[..., self.scalar_locs] + self.bias
            )

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

        self.apply(_init_weights)

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

        # Renormalize features
        norms_r = lnorms / (norms + self.epsilon)
        return f * norms_r[..., self.ix]


class Attention(nn.Module):
    """k-NN based multi-head attention with regular neighbor structure.

    Each node attends only to its k nearest neighbors, enabling
    efficient local attention with O(N*k) complexity instead of O(N^2).

    Args:
        hidden_size: Total hidden dimension.
        nheads: Number of attention heads.
        dropout: Dropout probability for attention weights.

    Example:
        >>> attn = Attention(64, nheads=4)
        >>> keys = torch.randn(100, 64)
        >>> queries = torch.randn(100, 64)
        >>> values = torch.randn(100, 64)
        >>> neighbor_idx = torch.randint(0, 100, (100, 16))
        >>> output = attn(keys, queries, values, neighbor_idx)  # (100, 64)
    """

    def __init__(
        self: Attention,
        hidden_size: int,
        nheads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_size % nheads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by nheads ({nheads})"
            )

        self.hidden_size = hidden_size
        self.nheads = nheads
        self.head_dim = hidden_size // nheads
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(
        self: Attention,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        neighbor_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute k-NN attention.

        Args:
            keys: Key vectors of shape (N, hidden_size).
            queries: Query vectors of shape (N, hidden_size).
            values: Value vectors of shape (N, hidden_size).
            neighbor_idx: Neighbor indices of shape (N, k).
            mask: Optional attention mask of shape (N, k).
                  True values are masked out.

        Returns:
            Attended values of shape (N, hidden_size).
        """
        N, k = neighbor_idx.shape

        # Reshape for multi-head: (N, nheads, head_dim)
        queries = queries.view(N, self.nheads, self.head_dim)
        keys = keys.view(N, self.nheads, self.head_dim)
        values = values.view(N, self.nheads, self.head_dim)

        # Gather neighbor keys and values: (N, k, nheads, head_dim)
        neighbor_keys = keys[neighbor_idx]
        neighbor_values = values[neighbor_idx]

        # Compute attention scores: (N, nheads, k)
        # queries: (N, nheads, head_dim) -> (N, nheads, 1, head_dim)
        # neighbor_keys: (N, k, nheads, head_dim) -> (N, nheads, k, head_dim)
        queries = queries.unsqueeze(2)  # (N, nheads, 1, head_dim)
        neighbor_keys = neighbor_keys.transpose(1, 2)  # (N, nheads, k, head_dim)
        neighbor_values = neighbor_values.transpose(1, 2)  # (N, nheads, k, head_dim)

        scores = (queries @ neighbor_keys.transpose(-2, -1)).squeeze(2) * self.scale
        # scores: (N, nheads, k)

        # Apply mask if provided
        if mask is not None:
            # mask: (N, k) -> (N, 1, k)
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        # Softmax over k neighbors
        weights = F.softmax(scores, dim=-1)  # (N, nheads, k)
        weights = self.dropout(weights)

        # Weighted sum of neighbor values: (N, nheads, head_dim)
        # weights: (N, nheads, k) -> (N, nheads, 1, k)
        # neighbor_values: (N, nheads, k, head_dim)
        output = (weights.unsqueeze(2) @ neighbor_values).squeeze(2)

        # Reshape back: (N, hidden_size)
        return output.view(N, self.hidden_size)


class EquivariantAttention(nn.Module):
    """SE(3)-equivariant k-NN attention.

    Computes Q, K, V as edge features via equivariant convolution. Attention
    weights are computed per-edge from Q · K, then used to weight edge-level
    values before aggregating to nodes.

    Flow:
        1. Q, K, V = conv(neighbor_features, edge_geometry) -> edge features
        2. scores = Q · K (per-edge dot product)
        3. attention = softmax(scores) over neighbors
        4. output = sum over neighbors of (attention * V)

    Args:
        repr: ProductRepr specifying representations.
        edge_dim: Dimension of invariant edge features.
        edge_hidden_dim: Hidden dimension for radial networks.
        nheads: Number of attention heads.
        dropout: Dropout for convolution.
        attn_dropout: Dropout for attention weights.
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

        self.repr = repr
        self.nheads = nheads

        out_mult = repr.rep2.mult
        out_dim = repr.rep2.dim()
        self.out_mult = out_mult
        self.out_dim = out_dim
        self.hidden_size = out_mult * out_dim

        if self.hidden_size % nheads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by nheads ({nheads})"
            )

        self.head_dim = self.hidden_size // nheads
        self.scale = self.head_dim ** -0.5

        # Q, K, V via equivariant convolution (all edge-level)
        repr_qkv = deepcopy(repr)
        repr_qkv.rep2.mult = 3 * out_mult  # Q, K, V concatenated
        self.conv_qkv = EquivariantConvolution(repr_qkv, edge_dim, edge_hidden_dim, dropout)

        # Output projection
        self.out_proj = EquivariantLinear(repr.rep2, repr.rep2, activation=None)

        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(
        self: EquivariantAttention,
        basis: tuple[torch.Tensor, torch.Tensor],
        edge_feats: torch.Tensor,
        f: torch.Tensor,
        neighbor_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply equivariant k-NN attention.

        Args:
            basis: Equivariant basis matrices of shape (N, k, ...).
            edge_feats: Edge features of shape (N, k, edge_dim).
            f: Node features of shape (N, mult, dim).
            neighbor_idx: Neighbor indices of shape (N, k).
            mask: Optional attention mask of shape (N, k).

        Returns:
            Updated node features of shape (N, mult, dim).
        """
        N, k = neighbor_idx.shape

        # Flatten for convolution
        src_idx = neighbor_idx.flatten()  # (N*k,)
        b1, b2 = basis
        b1_flat = b1.view(N * k, *b1.shape[2:])
        b2_flat = b2.view(N * k, *b2.shape[2:])
        edge_feats_flat = edge_feats.view(N * k, -1)

        # Compute Q, K, V as edge features via equivariant convolution
        qkv = self.conv_qkv((b1_flat, b2_flat), edge_feats_flat, f, src_idx)
        # qkv shape: (N*k, 3*out_mult, out_dim)
        qkv = qkv.view(N, k, 3 * self.out_mult, self.out_dim)

        # Split into Q, K, V: each (N, k, out_mult, out_dim)
        queries, keys, values = qkv.chunk(3, dim=2)

        # Reshape for multi-head attention: (N, k, nheads, head_dim)
        q_heads = queries.flatten(-2, -1).view(N, k, self.nheads, self.head_dim)
        k_heads = keys.flatten(-2, -1).view(N, k, self.nheads, self.head_dim)
        v_heads = values.flatten(-2, -1).view(N, k, self.nheads, self.head_dim)

        # Transpose to (N, nheads, k, head_dim)
        q_heads = q_heads.transpose(1, 2)
        k_heads = k_heads.transpose(1, 2)
        v_heads = v_heads.transpose(1, 2)

        # Edge attention scores: element-wise Q · K, sum over head_dim
        # (N, nheads, k, head_dim) * (N, nheads, k, head_dim) -> sum -> (N, nheads, k)
        scores = (q_heads * k_heads).sum(dim=-1) * self.scale

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), float('-inf'))

        # Softmax over neighbors for each node
        attn_weights = F.softmax(scores, dim=-1)  # (N, nheads, k)
        attn_weights = self.attn_dropout(attn_weights)

        # Weight edge values by attention and aggregate over edges
        # attn_weights: (N, nheads, k, 1)
        # v_heads: (N, nheads, k, head_dim)
        weighted = attn_weights.unsqueeze(-1) * v_heads  # (N, nheads, k, head_dim)
        output = weighted.sum(dim=2)  # (N, nheads, head_dim)

        # Reshape back to (N, out_mult, out_dim)
        output = output.view(N, self.out_mult, self.out_dim)

        return self.out_proj(output)


class EquivariantTransformerBlock(nn.Module):
    """Transformer block with k-NN attention.

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
        residual_scale: Scale factor for residual connections. Use < 1.0
            (e.g., 0.1-0.5) for deep networks to improve gradient flow.
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
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.prepr = repr
        self.residual_scale = residual_scale

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
            hidden_repr = deepcopy(repr.rep2)
            hidden_repr.mult = repr.rep2.mult * 4
            self.transition = EquivariantTransition(repr.rep2, hidden_repr)
        else:
            self.ln2 = None
            self.transition = None

    def forward(
        self: EquivariantTransformerBlock,
        basis: tuple[torch.Tensor, torch.Tensor],
        features: torch.Tensor,
        edge_feats: torch.Tensor,
        neighbor_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply transformer block.

        Args:
            basis: Equivariant basis matrices of shape (N, k, ...).
            features: Node features of shape (N, mult, dim).
            edge_feats: Edge features of shape (N, k, edge_dim).
            neighbor_idx: Neighbor indices of shape (N, k).
            mask: Optional attention mask of shape (N, k).

        Returns:
            Updated node features.
        """
        # Pre-LN transformer variant
        if self.skip:
            features_tmp = features

        features = self.ln1(features)
        features = self.attn(basis, edge_feats, features, neighbor_idx, mask)

        if self.skip:
            features = features_tmp + self.residual_scale * features

        if self.transition is not None:
            if self.skip:
                features_tmp = features
            features = self.ln2(features)
            features = self.transition(features)
            if self.skip:
                features = features_tmp + self.residual_scale * features

        return features


class EquivariantTransformer(nn.Module):
    """SE(3)-equivariant transformer for geometric point clouds.

    A full transformer architecture that processes 3D point clouds
    while maintaining SE(3) equivariance. Uses k-NN attention with
    O(N*k) complexity.

    Args:
        in_repr: Input representation.
        out_repr: Output representation.
        hidden_repr: Hidden representation for intermediate layers.
        hidden_layers: Number of hidden transformer blocks.
        edge_dim: Edge feature dimension.
        edge_hidden_dim: Hidden dimension for edge processing.
        k_neighbors: Number of nearest neighbors for attention.
        nheads: Number of attention heads.
        dropout: Dropout probability.
        attn_dropout: Attention dropout probability.
        transition: Whether to include transition layers.
        residual_scale: Scale factor for residual connections. Use < 1.0
            (e.g., 0.1-0.5) for deep networks to improve gradient flow.

    Example:
        >>> in_repr = Repr(lvals=[0, 1], mult=4)
        >>> out_repr = Repr(lvals=[0, 1], mult=1)
        >>> hidden_repr = Repr(lvals=[0, 1], mult=16)
        >>> model = EquivariantTransformer(
        ...     in_repr, out_repr, hidden_repr,
        ...     hidden_layers=4,
        ...     edge_dim=16,
        ...     edge_hidden_dim=32,
        ...     k_neighbors=16,
        ... )
        >>> output = model(coordinates, node_features)
    """

    def __init__(
        self: EquivariantTransformer,
        in_repr: Repr,
        out_repr: Repr,
        hidden_repr: Repr,
        hidden_layers: int,
        edge_dim: int,
        edge_hidden_dim: int,
        k_neighbors: int,
        nheads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        transition: bool = False,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.edge_dim = edge_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.nheads = nheads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.use_transition = transition
        self.residual_scale = residual_scale
        self.k_neighbors = k_neighbors

        self.in_repr = in_repr
        self.out_repr = out_repr
        self.hidden_repr = hidden_repr

        # Final layer norm and output projection
        out_repr_tmp = deepcopy(out_repr)
        out_repr_tmp.mult = hidden_repr.mult
        self.final_ln = EquivariantLayerNorm(out_repr_tmp)
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

        # Radial basis functions for edge features
        self.rbf = RadialBasisFunctions(edge_dim)

    def _construct_layer(
        self: EquivariantTransformer,
        prepr: ProductRepr,
    ) -> EquivariantTransformerBlock:
        """Construct a single transformer block."""
        return EquivariantTransformerBlock(
            prepr,
            self.edge_dim,
            self.edge_hidden_dim,
            self.nheads,
            self.dropout,
            self.attn_dropout,
            self.use_transition,
            self.residual_scale,
        )

    def forward(
        self: EquivariantTransformer,
        coordinates: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply equivariant transformer.

        Args:
            coordinates: Node coordinates of shape (N, 3).
            node_features: Node features of shape (N, mult, dim).
            edge_features: Optional edge features of shape (N, k, edge_dim).
                If None, uses distances.
            mask: Optional attention mask of shape (N, k).

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

        N = coordinates.size(0)

        # Build k-NN graph from coordinates
        neighbor_idx = build_knn_graph(coordinates, self.k_neighbors)
        k = neighbor_idx.size(1)

        # Compute displacements for (N, k) neighbor pairs
        neighbor_coords = coordinates[neighbor_idx]  # (N, k, 3)
        displacements = coordinates.unsqueeze(1) - neighbor_coords  # (N, k, 3)

        # Compute edge features from distances using RBFs if not provided
        if edge_features is None:
            distances = displacements.norm(dim=-1)  # (N, k)
            edge_features = self.rbf(distances)  # (N, k, edge_dim)

        # Compute all bases at once
        all_bases_flat = self.bases(displacements.view(N * k, 3))

        # Reshape bases for (N, k, ...) structure and pass through layers
        for layer, (b1, b2) in zip(self.layers, all_bases_flat):
            b1 = b1.view(N, k, *b1.shape[1:])
            b2 = b2.view(N, k, *b2.shape[1:])
            node_features = layer((b1, b2), node_features, edge_features, neighbor_idx, mask)

        # Final layer norm and output projection
        node_features = self.final_ln(node_features)
        return self.proj(node_features)
