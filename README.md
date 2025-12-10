# lr_geom - Low-Rank Geometric Deep Learning

A PyTorch library for SO(3)-equivariant neural networks using low-rank tensor product decompositions.

## Installation

```bash
pip install -e .

# With optional dependencies
pip install -e ".[spherical]"  # Spherical harmonics (sphericart)
pip install -e ".[graphs]"     # Graph layers (DGL)
pip install -e ".[all]"        # All optional dependencies
```

## Quick Start

### SO(3) Representations

```python
import torch
import lr_geom as lg

# Single irreducible representation (degree l has dimension 2l+1)
irrep_scalar = lg.Irrep(l=0)  # 1D scalar
irrep_vector = lg.Irrep(l=1)  # 3D vector
irrep_matrix = lg.Irrep(l=2)  # 5D traceless symmetric matrix

# Combined representation with multiplicity
# This creates 8 copies of (scalar + vector + rank-2 tensor)
repr = lg.Repr(lvals=[0, 1, 2], mult=8)
print(repr.dim())    # 9 (1 + 3 + 5)
print(repr.nreps())  # 3 irreps

# Verify tensor shape matches representation
x = torch.randn(32, 8, 9)  # (batch, mult, repr_dim)
assert repr.verify(x)

# Compute Wigner D-matrix for rotating spherical tensors
axis = torch.tensor([0., 0., 1.])
angle = torch.tensor(torch.pi / 4)  # 45 degrees around z-axis
D = repr.rot(axis, angle)  # shape: (1, 9, 9)

# Apply rotation to spherical tensor
x_rotated = x @ D.squeeze(0).T
```

### Point Cloud Alignment

```python
import torch
import lr_geom as lg

# Generate two point clouds
N = 100
source = torch.randn(N, 3)

# Create target by rotating and translating source
R = torch.tensor([
    [0.866, -0.5, 0.],
    [0.5, 0.866, 0.],
    [0., 0., 1.]
])
target = source @ R.T + torch.randn(3)

# Compute RMSD (root mean square deviation)
distance = lg.rmsd(source, target)
print(f"Before alignment: {distance:.4f}")

# Align source to target using Kabsch algorithm
aligned, centered_target = lg.kabsch_align(source, target)
print(f"After alignment: {lg.rmsd(aligned, centered_target):.4f}")

# Use RMSD as a differentiable loss function
loss_fn = lg.RMSD()
predicted = torch.randn(N, 3, requires_grad=True)
loss = loss_fn(predicted, target)
loss.backward()
```

### Equivariant Neural Network Layers

```python
import torch
import lr_geom as lg

# Define input/output representations
repr_in = lg.Repr(lvals=[0, 1, 2], mult=8)   # 8 x (scalar + vector + matrix)
repr_out = lg.Repr(lvals=[0, 1, 2], mult=16)  # 16 x (scalar + vector + matrix)

# Create sample input: batch of 32 spherical tensors
x = torch.randn(32, 8, 9)  # (batch, mult_in, repr_dim)

# Equivariant linear layer
# Applies separate linear transforms per irrep degree, preserving equivariance
linear = lg.EquivariantLinear(repr_in, repr_out)
y = linear(x)  # (32, 16, 9)

# Equivariant gating (learnable nonlinearity)
# Computes norms of each irrep, processes through MLP, gates original tensor
gate = lg.EquivariantGating(repr_out)
y = gate(y)  # (32, 16, 9)

# Equivariant layer normalization
# Normalizes based on irrep norms while preserving equivariance
ln = lg.EquivariantLayerNorm(repr_out)
y = ln(y)  # (32, 16, 9)

# Equivariant transition block (like transformer FFN)
# Projects up, applies gating, projects back down
hidden_repr = lg.Repr(lvals=[0, 1, 2], mult=64)
transition = lg.EquivariantTransition(repr_out, hidden_repr)
y = transition(y)  # (32, 16, 9)
```

### Spherical Harmonics and Radial Features

```python
import torch
import lr_geom as lg

# Spherical harmonics encode directional information equivariantly
# Requires: pip install sphericart
sh = lg.SphericalHarmonic(lmax=3)

# Compute spherical harmonic features for 3D coordinates
coords = torch.randn(100, 3)
features = sh(coords)  # (100, 16) - (lmax+1)^2 features

# Pairwise spherical harmonics for relative positions
pairwise_sh = sh.pairwise(coords)  # (100, 100, 16)

# Radial basis functions for encoding distances
rbf = lg.RadialBasisFunctions(num_functions=16)
distances = torch.norm(coords, dim=-1)  # (100,)
radial_features = rbf(distances)  # (100, 16)

# Combine for edge features in a graph
src, dst = torch.randint(0, 100, (2, 500))  # 500 edges
edge_vectors = coords[dst] - coords[src]
edge_distances = torch.norm(edge_vectors, dim=-1)

edge_radial = rbf(edge_distances)        # (500, 16) - invariant
edge_angular = sh(edge_vectors)          # (500, 16) - equivariant
```

### Building Equivariant Convolutions

```python
import torch
import lr_geom as lg

# Define product representation for message passing
repr1 = lg.Repr(lvals=[0, 1], mult=8)   # sender features
repr2 = lg.Repr(lvals=[0, 1], mult=16)  # receiver features
prod_repr = lg.ProductRepr(repr1, repr2)

# Compute equivariant basis matrices from edge displacements
basis = lg.EquivariantBasis(prod_repr)
edge_vectors = torch.randn(500, 3)  # 500 edges
coeff1, coeff2 = basis(edge_vectors)
# coeff1: (500, 4, 2) - projects sender to intermediate
# coeff2: (500, 2, 4) - projects intermediate to receiver

# Equivariant convolution layer
conv = lg.EquivariantConvolution(
    repr=prod_repr,
    edge_dim=16,        # dimension of invariant edge features
    hidden_dim=32,      # hidden dim for radial weight network
    dropout=0.1,
)

# Forward pass
node_features = torch.randn(100, 8, 4)   # (nodes, mult, repr_dim)
edge_features = torch.randn(500, 16)      # invariant edge features
src_idx = torch.randint(0, 100, (500,))   # source node indices

messages = conv(
    bases=(coeff1, coeff2),
    edge_feats=edge_features,
    f=node_features,
    src_idx=src_idx,
)  # (500, 16, 4) - messages per edge
```

### SE(3)-Equivariant Transformer (requires DGL)

```python
import torch
import lr_geom as lg

if lg.is_dgl_available():
    import dgl

    # Create a random graph
    num_nodes, num_edges = 50, 200
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    graph = dgl.graph((src, dst))

    # Node coordinates and features
    coordinates = torch.randn(num_nodes, 3)
    node_features = torch.randn(num_nodes, 4, 4)  # (N, mult, repr_dim)
    edge_features = torch.randn(num_edges, 16)     # invariant edge features

    # Build equivariant transformer
    model = lg.EquivariantTransformer(
        in_repr=lg.Repr(lvals=[0, 1], mult=4),
        out_repr=lg.Repr(lvals=[0, 1], mult=1),
        hidden_repr=lg.Repr(lvals=[0, 1], mult=16),
        hidden_layers=4,
        edge_dim=16,
        edge_hidden_dim=32,
        nheads=4,
        dropout=0.1,
        attn_dropout=0.1,
        transition=True,  # include FFN blocks
    )

    # Forward pass
    output = model(
        graph=graph,
        coordinates=coordinates,
        node_features=node_features,
        edge_features=edge_features,
    )  # (50, 1, 4)

    # The output transforms equivariantly under rotations:
    # If we rotate coordinates and input features, output rotates the same way
```

### Custom Equivariant Model Example

```python
import torch
import torch.nn as nn
import lr_geom as lg


class EquivariantMLP(nn.Module):
    """Simple equivariant MLP for point cloud processing."""

    def __init__(self, in_mult: int, hidden_mult: int, out_mult: int):
        super().__init__()

        # All layers use same lvals to allow residual connections
        lvals = [0, 1, 2]

        repr_in = lg.Repr(lvals=lvals, mult=in_mult)
        repr_hidden = lg.Repr(lvals=lvals, mult=hidden_mult)
        repr_out = lg.Repr(lvals=lvals, mult=out_mult)

        self.layer1 = lg.EquivariantLinear(repr_in, repr_hidden)
        self.gate1 = lg.EquivariantGating(repr_hidden)
        self.ln1 = lg.EquivariantLayerNorm(repr_hidden)

        self.layer2 = lg.EquivariantLinear(repr_hidden, repr_hidden)
        self.gate2 = lg.EquivariantGating(repr_hidden)
        self.ln2 = lg.EquivariantLayerNorm(repr_hidden)

        self.output = lg.EquivariantLinear(repr_hidden, repr_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_mult, 9)

        h = self.layer1(x)
        h = self.gate1(h)
        h = self.ln1(h)

        h = self.layer2(h) + h  # residual
        h = self.gate2(h)
        h = self.ln2(h)

        return self.output(h)


# Usage
model = EquivariantMLP(in_mult=4, hidden_mult=16, out_mult=1)
x = torch.randn(32, 4, 9)
y = model(x)  # (32, 1, 9)
```

## Package Structure

```
lr_geom/
├── alignment.py       - Point cloud alignment (Kabsch, RMSD)
├── representations.py - SO(3) representation theory (Irrep, Repr, Wigner D-matrices)
├── equivariant.py     - Equivariant primitives (SphericalHarmonic, RadialBasisFunctions)
├── layers.py          - Equivariant layers (Linear, Attention, Transformer)
├── models.py          - Pre-built models (RadialWeight, GNMA)
└── nn.py              - Basic neural network utilities
```

## API Reference

### Representations

| Class | Description |
|-------|-------------|
| `Irrep(l)` | Single irreducible representation of degree l |
| `Repr(lvals, mult)` | Direct sum of irreps with multiplicity |
| `ProductRepr(rep1, rep2)` | Tensor product of two representations |

### Layers (no DGL required)

| Class | Description |
|-------|-------------|
| `EquivariantLinear` | Linear transformation preserving SO(3) structure |
| `EquivariantGating` | Norm-based gating nonlinearity |
| `EquivariantTransition` | Feed-forward with gating |
| `EquivariantLayerNorm` | Equivariant layer normalization |
| `EquivariantConvolution` | Low-rank equivariant convolution |
| `RepNorm` | Compute norms of irrep components |
| `EquivariantBasis` | Compute equivariant basis matrices |

### Layers (require DGL)

| Class | Description |
|-------|-------------|
| `GraphAttention` | Multi-head graph attention |
| `EquivariantAttention` | SE(3)-equivariant attention |
| `EquivariantTransformerBlock` | Single transformer block |
| `EquivariantTransformer` | Full equivariant transformer |

### Alignment

| Function/Class | Description |
|----------------|-------------|
| `rmsd(x, y)` | Root mean square deviation |
| `kabsch_align(x, y)` | Optimal rotation alignment |
| `RMSD` | RMSD as nn.Module loss |

## License

MIT License
