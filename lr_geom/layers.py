"""Equivariant neural network layers for geometric deep learning.

This module provides SE(3)-equivariant layers for building geometric
deep learning models. These layers maintain equivariance to rotations
and translations when processing 3D molecular or point cloud data.

Supports two attention modes:
- Dense: All-to-all attention using PyTorch's scaled_dot_product_attention
- Sparse: k-NN based attention with regular (N, k) neighbor structure

Classes:
    RadialWeight: Neural network for computing tensor product weights
    EquivariantLinear: Linear layer preserving spherical tensor structure
    EquivariantGating: Norm-based gating for spherical tensors
    EquivariantTransition: MLP transition layer for spherical tensors
    EquivariantConvolution: SE(3)-equivariant convolution
    EquivariantLayerNorm: Equivariant layer normalization
    DenseAttention: All-to-all multi-head attention
    SparseAttention: k-NN based multi-head attention
    EquivariantDenseAttention: SE(3)-equivariant dense attention
    EquivariantSparseAttention: SE(3)-equivariant sparse attention
    EquivariantDenseTransformerBlock: Dense transformer block
    EquivariantSparseTransformerBlock: Sparse transformer block
    EquivariantTransformer: Full equivariant transformer
"""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .representations import Repr, ProductRepr
from .equivariant import RepNorm, EquivariantBases, FEATURE_DIM


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

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

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


class DenseAttention(nn.Module):
    """All-to-all multi-head attention using PyTorch SDPA.

    Computes attention over all pairs of nodes using PyTorch's
    efficient scaled_dot_product_attention implementation.

    Args:
        hidden_size: Total hidden dimension.
        nheads: Number of attention heads.
        dropout: Dropout probability for attention weights.

    Example:
        >>> attn = DenseAttention(64, nheads=4)
        >>> keys = torch.randn(100, 64)
        >>> queries = torch.randn(100, 64)
        >>> values = torch.randn(100, 64)
        >>> output = attn(keys, queries, values)  # (100, 64)
    """

    def __init__(
        self: DenseAttention,
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
        self.dropout_p = dropout

    def forward(
        self: DenseAttention,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dense attention.

        Args:
            keys: Key vectors of shape (N, hidden_size).
            queries: Query vectors of shape (N, hidden_size).
            values: Value vectors of shape (N, hidden_size).
            mask: Optional attention mask of shape (N, N).
                  True values are masked out.

        Returns:
            Attended values of shape (N, hidden_size).
        """
        N = keys.size(0)

        # Reshape for multi-head: (N, nheads, head_dim)
        keys = keys.view(N, self.nheads, self.head_dim)
        queries = queries.view(N, self.nheads, self.head_dim)
        values = values.view(N, self.nheads, self.head_dim)

        # Transpose for SDPA: (nheads, N, head_dim) -> (1, nheads, N, head_dim)
        keys = keys.transpose(0, 1).unsqueeze(0)
        queries = queries.transpose(0, 1).unsqueeze(0)
        values = values.transpose(0, 1).unsqueeze(0)

        # Prepare mask for SDPA if provided
        attn_mask = None
        if mask is not None:
            # SDPA expects mask where True = masked out
            attn_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)

        # Apply scaled dot product attention
        dropout_p = self.dropout_p if self.training else 0.0
        output = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )  # (1, nheads, N, head_dim)

        # Reshape back: (N, hidden_size)
        output = output.squeeze(0).transpose(0, 1).contiguous()
        return output.view(N, self.hidden_size)


class SparseAttention(nn.Module):
    """k-NN based multi-head attention with regular neighbor structure.

    Each node attends only to its k nearest neighbors, enabling
    efficient local attention with O(N*k) complexity instead of O(N^2).

    Args:
        hidden_size: Total hidden dimension.
        nheads: Number of attention heads.
        dropout: Dropout probability for attention weights.

    Example:
        >>> attn = SparseAttention(64, nheads=4)
        >>> keys = torch.randn(100, 64)
        >>> queries = torch.randn(100, 64)
        >>> values = torch.randn(100, 64)
        >>> neighbor_idx = torch.randint(0, 100, (100, 16))
        >>> output = attn(keys, queries, values, neighbor_idx)  # (100, 64)
    """

    def __init__(
        self: SparseAttention,
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
        self: SparseAttention,
        keys: torch.Tensor,
        queries: torch.Tensor,
        values: torch.Tensor,
        neighbor_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute sparse k-NN attention.

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


class EquivariantDenseAttention(nn.Module):
    """SE(3)-equivariant dense (all-to-all) attention.

    Combines equivariant convolution with dense attention to compute
    equivariant message passing over all pairs of nodes.

    Args:
        repr: ProductRepr specifying representations.
        edge_dim: Dimension of invariant edge features.
        edge_hidden_dim: Hidden dimension for radial networks.
        nheads: Number of attention heads.
        dropout: Dropout for convolution.
        attn_dropout: Dropout for attention weights.
    """

    def __init__(
        self: EquivariantDenseAttention,
        repr: ProductRepr,
        edge_dim: int,
        edge_hidden_dim: int,
        nheads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.repr = repr

        # Create representation for keys, queries, values (3x output mult)
        repr_h = deepcopy(repr)
        repr_h.rep2.mult = 3 * repr.rep2.mult

        self.conv = EquivariantConvolution(
            repr_h, edge_dim, edge_hidden_dim, dropout
        )

        self.attn = DenseAttention(
            repr.rep2.mult * repr.rep2.dim(),
            nheads,
            attn_dropout,
        )

        self.proj = EquivariantLinear(repr.rep2, repr.rep2, dropout, activation=None)
        self.outdims = (repr.rep2.mult, repr.rep2.dim())

    def forward(
        self: EquivariantDenseAttention,
        basis: tuple[torch.Tensor, torch.Tensor],
        edge_feats: torch.Tensor,
        f: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply equivariant dense attention.

        Args:
            basis: Equivariant basis matrices of shape (N, N, ...).
            edge_feats: Edge features of shape (N, N, edge_dim).
            f: Node features of shape (N, mult, dim).
            mask: Optional attention mask of shape (N, N).

        Returns:
            Updated node features of shape (N, mult, dim).
        """
        N = f.size(0)

        # Create source indices for all pairs: (N, N) -> (N*N,)
        src_idx = torch.arange(N, device=f.device).unsqueeze(0).expand(N, -1).flatten()

        # Flatten basis and edge_feats for convolution
        b1, b2 = basis
        b1_flat = b1.view(N * N, *b1.shape[2:])
        b2_flat = b2.view(N * N, *b2.shape[2:])
        edge_feats_flat = edge_feats.view(N * N, -1)

        # Equivariant convolution to get keys, queries, values
        conv = self.conv((b1_flat, b2_flat), edge_feats_flat, f, src_idx)
        # conv shape: (N*N, 3*mult, dim)
        conv = conv.view(N, N, 3 * self.outdims[0], self.outdims[1])  # (N, N, 3*mult, dim)

        # Extract k, q, v and aggregate
        # For dense attention, we need node-level k, q, v
        # Average over source dimension to get per-node representations
        conv_flat = conv.flatten(-2, -1)  # (N, N, 3*mult*dim)
        k, q, v = torch.chunk(conv_flat, 3, dim=-1)  # each (N, N, mult*dim)

        # Aggregate: take mean over source nodes for keys/values
        k_node = k.mean(dim=1)  # (N, hidden)
        q_node = q.mean(dim=0)  # (N, hidden) - mean over targets
        v_node = v.mean(dim=1)  # (N, hidden)

        # Apply dense attention
        attn_out = self.attn(k_node, q_node, v_node, mask)

        # Reshape and project
        attn_out = attn_out.view(N, *self.outdims)
        return self.proj(attn_out)


class EquivariantSparseAttention(nn.Module):
    """SE(3)-equivariant sparse (k-NN) attention.

    Combines equivariant convolution with sparse k-NN attention to
    compute equivariant message passing over local neighborhoods.

    Args:
        repr: ProductRepr specifying representations.
        edge_dim: Dimension of invariant edge features.
        edge_hidden_dim: Hidden dimension for radial networks.
        nheads: Number of attention heads.
        dropout: Dropout for convolution.
        attn_dropout: Dropout for attention weights.
    """

    def __init__(
        self: EquivariantSparseAttention,
        repr: ProductRepr,
        edge_dim: int,
        edge_hidden_dim: int,
        nheads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.repr = repr

        # Create representation for keys, queries, values (3x output mult)
        repr_h = deepcopy(repr)
        repr_h.rep2.mult = 3 * repr.rep2.mult

        self.conv = EquivariantConvolution(
            repr_h, edge_dim, edge_hidden_dim, dropout
        )

        self.attn = SparseAttention(
            repr.rep2.mult * repr.rep2.dim(),
            nheads,
            attn_dropout,
        )

        self.proj = EquivariantLinear(repr.rep2, repr.rep2, dropout, activation=None)
        self.outdims = (repr.rep2.mult, repr.rep2.dim())

    def forward(
        self: EquivariantSparseAttention,
        basis: tuple[torch.Tensor, torch.Tensor],
        edge_feats: torch.Tensor,
        f: torch.Tensor,
        neighbor_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply equivariant sparse attention.

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

        # Flatten neighbor dimension for convolution: (N*k,)
        src_idx = neighbor_idx.flatten()

        # Flatten basis and edge_feats
        b1, b2 = basis
        b1_flat = b1.view(N * k, *b1.shape[2:])
        b2_flat = b2.view(N * k, *b2.shape[2:])
        edge_feats_flat = edge_feats.view(N * k, -1)

        # Equivariant convolution to get keys, queries, values per edge
        conv = self.conv((b1_flat, b2_flat), edge_feats_flat, f, src_idx)
        # conv shape: (N*k, 3*mult, dim)
        conv = conv.view(N, k, 3 * self.outdims[0], self.outdims[1])  # (N, k, 3*mult, dim)

        # Split into k, q, v
        conv_flat = conv.flatten(-2, -1)  # (N, k, 3*mult*dim)
        k_edge, q_edge, v_edge = torch.chunk(conv_flat, 3, dim=-1)  # each (N, k, mult*dim)

        # For sparse attention:
        # - keys and values come from neighbors (N, k, hidden) -> gather to (N, hidden)
        # - queries are per-node
        # We need node-level representations for attention
        k_node = k_edge.mean(dim=1)  # (N, hidden)
        q_node = q_edge.mean(dim=1)  # (N, hidden)
        v_node = v_edge.mean(dim=1)  # (N, hidden)

        # Apply sparse attention
        attn_out = self.attn(k_node, q_node, v_node, neighbor_idx, mask)

        # Reshape and project
        attn_out = attn_out.view(N, *self.outdims)
        return self.proj(attn_out)


class EquivariantDenseTransformerBlock(nn.Module):
    """Dense transformer block with all-to-all attention.

    Combines equivariant dense attention with optional transition layer,
    layer normalization, and residual connections.

    Args:
        repr: ProductRepr for the block.
        edge_dim: Edge feature dimension.
        edge_hidden_dim: Hidden dimension for edge processing.
        nheads: Number of attention heads.
        dropout: Dropout probability.
        attn_dropout: Attention dropout probability.
        transition: Whether to include transition layer.
    """

    def __init__(
        self: EquivariantDenseTransformerBlock,
        repr: ProductRepr,
        edge_dim: int,
        edge_hidden_dim: int,
        nheads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        transition: bool = False,
    ) -> None:
        super().__init__()

        self.prepr = repr

        self.attn = EquivariantDenseAttention(
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
        self: EquivariantDenseTransformerBlock,
        basis: tuple[torch.Tensor, torch.Tensor],
        features: torch.Tensor,
        edge_feats: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply dense transformer block.

        Args:
            basis: Equivariant basis matrices of shape (N, N, ...).
            features: Node features of shape (N, mult, dim).
            edge_feats: Edge features of shape (N, N, edge_dim).
            mask: Optional attention mask of shape (N, N).

        Returns:
            Updated node features.
        """
        # Pre-LN transformer variant
        if self.skip:
            features_tmp = features

        features = self.ln1(features)
        features = self.attn(basis, edge_feats, features, mask)

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


class EquivariantSparseTransformerBlock(nn.Module):
    """Sparse transformer block with k-NN attention.

    Combines equivariant sparse attention with optional transition layer,
    layer normalization, and residual connections.

    Args:
        repr: ProductRepr for the block.
        edge_dim: Edge feature dimension.
        edge_hidden_dim: Hidden dimension for edge processing.
        nheads: Number of attention heads.
        dropout: Dropout probability.
        attn_dropout: Attention dropout probability.
        transition: Whether to include transition layer.
    """

    def __init__(
        self: EquivariantSparseTransformerBlock,
        repr: ProductRepr,
        edge_dim: int,
        edge_hidden_dim: int,
        nheads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        transition: bool = False,
    ) -> None:
        super().__init__()

        self.prepr = repr

        self.attn = EquivariantSparseAttention(
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
        self: EquivariantSparseTransformerBlock,
        basis: tuple[torch.Tensor, torch.Tensor],
        features: torch.Tensor,
        edge_feats: torch.Tensor,
        neighbor_idx: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply sparse transformer block.

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
    """SE(3)-equivariant transformer for geometric point clouds.

    A full transformer architecture that processes 3D point clouds
    while maintaining SE(3) equivariance. Supports two attention modes:
    - Dense (k_neighbors=None): All-to-all attention, O(N^2) complexity
    - Sparse (k_neighbors=int): k-NN attention, O(N*k) complexity

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
        k_neighbors: Number of nearest neighbors for sparse attention.
            If None, uses dense all-to-all attention.

    Example:
        >>> in_repr = Repr(lvals=[0, 1], mult=4)
        >>> out_repr = Repr(lvals=[0, 1], mult=1)
        >>> hidden_repr = Repr(lvals=[0, 1], mult=16)
        >>> # Sparse attention with k=16 neighbors
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
        nheads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        transition: bool = False,
        k_neighbors: int | None = None,
    ) -> None:
        super().__init__()

        self.edge_dim = edge_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.nheads = nheads
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.use_transition = transition
        self.k_neighbors = k_neighbors

        self.in_repr = in_repr
        self.out_repr = out_repr
        self.hidden_repr = hidden_repr

        # Output projection
        out_repr_tmp = deepcopy(out_repr)
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

    def _construct_layer(
        self: EquivariantTransformer,
        prepr: ProductRepr,
    ) -> EquivariantDenseTransformerBlock | EquivariantSparseTransformerBlock:
        """Construct a single transformer block."""
        if self.k_neighbors is None:
            return EquivariantDenseTransformerBlock(
                prepr,
                self.edge_dim,
                self.edge_hidden_dim,
                self.nheads,
                self.dropout,
                self.attn_dropout,
                self.use_transition,
            )
        else:
            return EquivariantSparseTransformerBlock(
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
        coordinates: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply equivariant transformer.

        Args:
            coordinates: Node coordinates of shape (N, 3).
            node_features: Node features of shape (N, mult, dim).
            edge_features: Optional edge features. For sparse mode, shape (N, k, edge_dim).
                For dense mode, shape (N, N, edge_dim). If None, uses distances.
            mask: Optional attention mask. For sparse mode, shape (N, k).
                For dense mode, shape (N, N).

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

        if self.k_neighbors is not None:
            # Sparse mode: k-NN attention
            neighbor_idx = build_knn_graph(coordinates, self.k_neighbors)
            k = neighbor_idx.size(1)

            # Compute displacements for (N, k) neighbor pairs
            neighbor_coords = coordinates[neighbor_idx]  # (N, k, 3)
            displacements = coordinates.unsqueeze(1) - neighbor_coords  # (N, k, 3)

            # Compute edge features from distances if not provided
            if edge_features is None:
                distances = displacements.norm(dim=-1, keepdim=True)  # (N, k, 1)
                edge_features = distances.expand(-1, -1, self.edge_dim)

            # Compute all bases at once
            all_bases_flat = self.bases(displacements.view(N * k, 3))

            # Reshape bases for (N, k, ...) structure and pass through layers
            for layer, (b1, b2) in zip(self.layers, all_bases_flat):
                b1 = b1.view(N, k, *b1.shape[1:])
                b2 = b2.view(N, k, *b2.shape[1:])
                node_features = layer((b1, b2), node_features, edge_features, neighbor_idx, mask)

        else:
            # Dense mode: all-to-all attention
            # Compute displacements for all (N, N) pairs
            displacements = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)  # (N, N, 3)

            # Compute edge features from distances if not provided
            if edge_features is None:
                distances = displacements.norm(dim=-1, keepdim=True)  # (N, N, 1)
                edge_features = distances.expand(-1, -1, self.edge_dim)

            # Compute all bases at once
            all_bases_flat = self.bases(displacements.view(N * N, 3))

            # Reshape bases for (N, N, ...) structure and pass through layers
            for layer, (b1, b2) in zip(self.layers, all_bases_flat):
                b1 = b1.view(N, N, *b1.shape[1:])
                b2 = b2.view(N, N, *b2.shape[1:])
                node_features = layer((b1, b2), node_features, edge_features, mask)

        # Output projection
        return self.proj(node_features)
