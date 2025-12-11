"""Profile VAE and layer components.

This script profiles the forward pass timing of each component in the
equivariant VAE architecture.

Usage:
    python examples/profile_vae.py
    python examples/profile_vae.py --n_atoms 200 --k_neighbors 64
    python examples/profile_vae.py --device cuda --warmup 5 --iterations 20
"""
from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

import lr_geom as lg
from lr_geom.vae import EquivariantVAE
from lr_geom.layers import (
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
    build_knn_graph,
)
from lr_geom.equivariant import (
    RadialBasisFunctions,
    EquivariantBasis,
    EquivariantBases,
    SphericalHarmonic,
    RepNorm,
)


@dataclass
class ProfileResult:
    """Result of profiling a component."""
    name: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    num_params: int = 0
    memory_mb: float = 0.0


@dataclass
class ProfilerConfig:
    """Configuration for profiling."""
    n_atoms: int = 100
    k_neighbors: int = 16
    embed_dim: int = 16
    hidden_mult: int = 32
    latent_mult: int = 8
    nheads: int = 8
    warmup: int = 3
    iterations: int = 10
    device: str = "cuda"
    attention_type: str = "node_wise"
    radial_weight_rank: int | None = None


class LayerProfiler:
    """Profiler for individual layers and full models."""

    def __init__(self, config: ProfilerConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.results: list[ProfileResult] = []

        # Create representations
        self.in_repr = lg.Repr([0], mult=config.embed_dim)
        self.hidden_repr = lg.Repr([0, 1, 2], mult=config.hidden_mult)
        self.latent_repr = lg.Repr([0, 1, 2], mult=config.latent_mult)
        self.out_repr = lg.Repr([1], mult=1)

        # Create sample data
        self._create_sample_data()

    def _create_sample_data(self):
        """Create sample input data for profiling."""
        n = self.config.n_atoms
        k = self.config.k_neighbors

        # Coordinates and features
        self.coords = torch.randn(n, 3, device=self.device)
        self.features_in = torch.randn(
            n, self.in_repr.mult, self.in_repr.dim(), device=self.device
        )
        self.features_hidden = torch.randn(
            n, self.hidden_repr.mult, self.hidden_repr.dim(), device=self.device
        )

        # Build k-NN graph
        self.neighbor_idx = build_knn_graph(self.coords, k)

        # Compute displacements and distances
        neighbors = self.coords[self.neighbor_idx]  # (N, k, 3)
        self.displacements = neighbors - self.coords.unsqueeze(1)  # (N, k, 3)
        self.distances = self.displacements.norm(dim=-1)  # (N, k)

    @contextmanager
    def _cuda_timer(self):
        """Context manager for CUDA-synchronized timing."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield lambda: (end.record(), torch.cuda.synchronize(), start.elapsed_time(end))
        else:
            start_time = time.perf_counter()
            yield lambda: (time.perf_counter() - start_time) * 1000  # ms

    def _profile_forward(
        self,
        name: str,
        model: nn.Module,
        forward_fn: Callable,
    ) -> ProfileResult:
        """Profile a forward pass."""
        model.eval()

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        # Warmup
        with torch.no_grad():
            for _ in range(self.config.warmup):
                forward_fn()
                if self.device.type == "cuda":
                    torch.cuda.synchronize()

        # Measure memory before
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Timed iterations
        times = []
        with torch.no_grad():
            for _ in range(self.config.iterations):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    forward_fn()
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end))
                else:
                    start = time.perf_counter()
                    forward_fn()
                    times.append((time.perf_counter() - start) * 1000)

        # Memory usage
        memory_mb = 0.0
        if self.device.type == "cuda":
            memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        times_tensor = torch.tensor(times)
        result = ProfileResult(
            name=name,
            mean_time_ms=times_tensor.mean().item(),
            std_time_ms=times_tensor.std().item(),
            min_time_ms=times_tensor.min().item(),
            max_time_ms=times_tensor.max().item(),
            num_params=num_params,
            memory_mb=memory_mb,
        )
        self.results.append(result)
        return result

    def profile_radial_basis_functions(self) -> ProfileResult:
        """Profile RadialBasisFunctions."""
        rbf = RadialBasisFunctions(
            num_functions=16,
            r_min=0.0,
            r_max=10.0,
        ).to(self.device)

        return self._profile_forward(
            "RadialBasisFunctions",
            rbf,
            lambda: rbf(self.distances),
        )

    def profile_spherical_harmonic(self) -> ProfileResult:
        """Profile SphericalHarmonic."""
        sh = SphericalHarmonic(lmax=2).to(self.device)

        # Normalize displacements
        unit_vecs = self.displacements / (self.distances.unsqueeze(-1) + 1e-8)

        return self._profile_forward(
            "SphericalHarmonic",
            sh,
            lambda: sh(unit_vecs.reshape(-1, 3)),
        )

    def profile_equivariant_basis(self) -> ProfileResult:
        """Profile EquivariantBasis."""
        prod_repr = lg.ProductRepr(self.hidden_repr, self.hidden_repr)
        basis = EquivariantBasis(prod_repr).to(self.device)

        # Need unit displacement vectors
        unit_vecs = self.displacements / (self.distances.unsqueeze(-1) + 1e-8)

        return self._profile_forward(
            "EquivariantBasis",
            basis,
            lambda: basis(unit_vecs.reshape(-1, 3)),
        )

    def profile_radial_weight(self) -> ProfileResult:
        """Profile RadialWeight."""
        prod_repr = lg.ProductRepr(self.hidden_repr, self.hidden_repr)
        edge_dim = 16

        rw = RadialWeight(
            edge_dim=edge_dim,
            hidden_dim=32,
            repr=prod_repr,
            in_dim=self.hidden_repr.mult,
            out_dim=self.hidden_repr.mult,
            rank=self.config.radial_weight_rank,
        ).to(self.device)

        # Create edge features
        rbf = RadialBasisFunctions(edge_dim).to(self.device)
        edge_feats = rbf(self.distances)  # (N, k, edge_dim)

        return self._profile_forward(
            f"RadialWeight (rank={self.config.radial_weight_rank or 'full'})",
            rw,
            lambda: rw(edge_feats),
        )

    def profile_rep_norm(self) -> ProfileResult:
        """Profile RepNorm."""
        norm = RepNorm(self.hidden_repr).to(self.device)

        return self._profile_forward(
            "RepNorm",
            norm,
            lambda: norm(self.features_hidden),
        )

    def profile_equivariant_linear(self) -> ProfileResult:
        """Profile EquivariantLinear."""
        linear = EquivariantLinear(
            self.hidden_repr,
            self.hidden_repr,
        ).to(self.device)

        return self._profile_forward(
            "EquivariantLinear",
            linear,
            lambda: linear(self.features_hidden),
        )

    def profile_equivariant_gating(self) -> ProfileResult:
        """Profile EquivariantGating."""
        gating = EquivariantGating(self.hidden_repr).to(self.device)

        return self._profile_forward(
            "EquivariantGating",
            gating,
            lambda: gating(self.features_hidden),
        )

    def profile_equivariant_layer_norm(self) -> ProfileResult:
        """Profile EquivariantLayerNorm."""
        ln = EquivariantLayerNorm(self.hidden_repr).to(self.device)

        return self._profile_forward(
            "EquivariantLayerNorm",
            ln,
            lambda: ln(self.features_hidden),
        )

    def profile_equivariant_transition(self) -> ProfileResult:
        """Profile EquivariantTransition."""
        transition = EquivariantTransition(
            self.hidden_repr,
            expansion_factor=2,
        ).to(self.device)

        return self._profile_forward(
            "EquivariantTransition",
            transition,
            lambda: transition(self.features_hidden),
        )

    def profile_attention(self) -> ProfileResult:
        """Profile Attention (scalar attention module)."""
        attn = Attention(
            dim=self.hidden_repr.mult,
            nheads=self.config.nheads,
        ).to(self.device)

        # Create Q, K, V tensors (N, k, mult)
        q = torch.randn(
            self.config.n_atoms, self.config.k_neighbors, self.hidden_repr.mult,
            device=self.device
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        return self._profile_forward(
            "Attention (scalar)",
            attn,
            lambda: attn(q, k, v),
        )

    def profile_equivariant_convolution(self) -> ProfileResult:
        """Profile EquivariantConvolution."""
        prod_repr = lg.ProductRepr(self.hidden_repr, self.hidden_repr)
        edge_dim = 16

        conv = EquivariantConvolution(
            prod_repr,
            edge_dim=edge_dim,
            edge_hidden_dim=32,
            in_dim=self.hidden_repr.mult,
            out_dim=self.hidden_repr.mult,
            radial_weight_rank=self.config.radial_weight_rank,
        ).to(self.device)

        # Create basis and edge features
        basis_mod = EquivariantBasis(prod_repr).to(self.device)
        rbf = RadialBasisFunctions(edge_dim).to(self.device)

        unit_vecs = self.displacements / (self.distances.unsqueeze(-1) + 1e-8)
        basis = basis_mod(unit_vecs.reshape(-1, 3)).reshape(
            self.config.n_atoms, self.config.k_neighbors, -1
        )
        edge_feats = rbf(self.distances)

        return self._profile_forward(
            f"EquivariantConvolution (rank={self.config.radial_weight_rank or 'full'})",
            conv,
            lambda: conv(basis, edge_feats, self.features_hidden, self.neighbor_idx),
        )

    def profile_equivariant_attention(self) -> ProfileResult:
        """Profile EquivariantAttention."""
        prod_repr = lg.ProductRepr(self.hidden_repr, self.hidden_repr)
        edge_dim = 16

        attn = EquivariantAttention(
            repr=prod_repr,
            edge_dim=edge_dim,
            edge_hidden_dim=32,
            nheads=self.config.nheads,
            attention_type=self.config.attention_type,
            radial_weight_rank=self.config.radial_weight_rank,
        ).to(self.device)

        # Create basis and edge features
        basis_mod = EquivariantBasis(prod_repr).to(self.device)
        rbf = RadialBasisFunctions(edge_dim).to(self.device)

        unit_vecs = self.displacements / (self.distances.unsqueeze(-1) + 1e-8)
        basis = basis_mod(unit_vecs.reshape(-1, 3)).reshape(
            self.config.n_atoms, self.config.k_neighbors, -1
        )
        edge_feats = rbf(self.distances)

        return self._profile_forward(
            f"EquivariantAttention ({self.config.attention_type})",
            attn,
            lambda: attn(basis, edge_feats, self.features_hidden, self.neighbor_idx),
        )

    def profile_transformer_block(self) -> ProfileResult:
        """Profile EquivariantTransformerBlock."""
        prod_repr = lg.ProductRepr(self.hidden_repr, self.hidden_repr)
        edge_dim = 16

        block = EquivariantTransformerBlock(
            repr=prod_repr,
            edge_dim=edge_dim,
            edge_hidden_dim=32,
            nheads=self.config.nheads,
            attention_type=self.config.attention_type,
            radial_weight_rank=self.config.radial_weight_rank,
        ).to(self.device)

        # Create basis and edge features
        basis_mod = EquivariantBasis(prod_repr).to(self.device)
        rbf = RadialBasisFunctions(edge_dim).to(self.device)

        unit_vecs = self.displacements / (self.distances.unsqueeze(-1) + 1e-8)
        basis = basis_mod(unit_vecs.reshape(-1, 3)).reshape(
            self.config.n_atoms, self.config.k_neighbors, -1
        )
        edge_feats = rbf(self.distances)

        return self._profile_forward(
            f"EquivariantTransformerBlock ({self.config.attention_type})",
            block,
            lambda: block(basis, edge_feats, self.features_hidden, self.neighbor_idx),
        )

    def profile_transformer(self) -> ProfileResult:
        """Profile EquivariantTransformer."""
        transformer = EquivariantTransformer(
            in_repr=self.in_repr,
            out_repr=self.hidden_repr,
            hidden_repr=self.hidden_repr,
            hidden_layers=2,
            k_neighbors=self.config.k_neighbors,
            nheads=self.config.nheads,
            attention_type=self.config.attention_type,
            radial_weight_rank=self.config.radial_weight_rank,
        ).to(self.device)

        return self._profile_forward(
            f"EquivariantTransformer (2 layers, {self.config.attention_type})",
            transformer,
            lambda: transformer(self.coords, self.features_in),
        )

    def profile_vae(self) -> ProfileResult:
        """Profile full EquivariantVAE."""
        vae = EquivariantVAE(
            in_repr=self.in_repr,
            latent_repr=self.latent_repr,
            out_repr=self.out_repr,
            hidden_repr=self.hidden_repr,
            encoder_layers=2,
            decoder_layers=2,
            k_neighbors=self.config.k_neighbors,
            nheads=self.config.nheads,
            attention_type=self.config.attention_type,
            radial_weight_rank=self.config.radial_weight_rank,
        ).to(self.device)

        return self._profile_forward(
            f"EquivariantVAE (2+2 layers, {self.config.attention_type})",
            vae,
            lambda: vae(self.coords, self.features_in),
        )

    def profile_all(self) -> list[ProfileResult]:
        """Run all profiling benchmarks."""
        self.results = []

        print(f"\nProfiling with config:")
        print(f"  n_atoms: {self.config.n_atoms}")
        print(f"  k_neighbors: {self.config.k_neighbors}")
        print(f"  hidden_mult: {self.config.hidden_mult}")
        print(f"  attention_type: {self.config.attention_type}")
        print(f"  radial_weight_rank: {self.config.radial_weight_rank or 'full'}")
        print(f"  device: {self.device}")
        print(f"  warmup: {self.config.warmup}, iterations: {self.config.iterations}")
        print()

        # Profile each component
        print("Profiling components...")

        # Primitives
        print("  [1/14] RadialBasisFunctions")
        self.profile_radial_basis_functions()
        print("  [2/14] SphericalHarmonic")
        self.profile_spherical_harmonic()
        print("  [3/14] EquivariantBasis")
        self.profile_equivariant_basis()
        print("  [4/14] RepNorm")
        self.profile_rep_norm()

        # Basic layers
        print("  [5/14] RadialWeight")
        self.profile_radial_weight()
        print("  [6/14] EquivariantLinear")
        self.profile_equivariant_linear()
        print("  [7/14] EquivariantGating")
        self.profile_equivariant_gating()
        print("  [8/14] EquivariantLayerNorm")
        self.profile_equivariant_layer_norm()
        print("  [9/14] EquivariantTransition")
        self.profile_equivariant_transition()

        # Attention
        print("  [10/14] Attention (scalar)")
        self.profile_attention()
        print("  [11/14] EquivariantConvolution")
        self.profile_equivariant_convolution()
        print("  [12/14] EquivariantAttention")
        self.profile_equivariant_attention()

        # Composite
        print("  [13/14] EquivariantTransformerBlock")
        self.profile_transformer_block()
        print("  [14/14] EquivariantTransformer + VAE")
        self.profile_transformer()
        self.profile_vae()

        return self.results

    def print_results(self):
        """Print profiling results as a table."""
        print()
        print("=" * 100)
        print("PROFILING RESULTS")
        print("=" * 100)
        print(
            f"{'Component':<50} {'Mean (ms)':<12} {'Std (ms)':<10} "
            f"{'Params':<12} {'Mem (MB)':<10}"
        )
        print("-" * 100)

        for r in self.results:
            params_str = f"{r.num_params:,}" if r.num_params > 0 else "-"
            mem_str = f"{r.memory_mb:.1f}" if r.memory_mb > 0 else "-"
            print(
                f"{r.name:<50} {r.mean_time_ms:<12.3f} {r.std_time_ms:<10.3f} "
                f"{params_str:<12} {mem_str:<10}"
            )

        print("=" * 100)


def compare_configurations(configs: list[ProfilerConfig]) -> None:
    """Compare profiling results across different configurations."""
    all_results = {}

    for config in configs:
        profiler = LayerProfiler(config)
        profiler.profile_all()

        key = f"k={config.k_neighbors}, rank={config.radial_weight_rank or 'full'}, attn={config.attention_type}"
        all_results[key] = {r.name: r for r in profiler.results}

    # Print comparison for key components
    print()
    print("=" * 120)
    print("CONFIGURATION COMPARISON (mean time in ms)")
    print("=" * 120)

    components = [
        "EquivariantAttention",
        "EquivariantTransformerBlock",
        "EquivariantVAE",
    ]

    # Find matching components
    for component_prefix in components:
        print(f"\n{component_prefix}:")
        print("-" * 80)

        for config_key, results in all_results.items():
            for name, result in results.items():
                if name.startswith(component_prefix):
                    print(f"  {config_key}: {result.mean_time_ms:.3f} ms (±{result.std_time_ms:.3f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile VAE components")
    parser.add_argument("--n_atoms", type=int, default=100)
    parser.add_argument("--k_neighbors", type=int, default=16)
    parser.add_argument("--hidden_mult", type=int, default=32)
    parser.add_argument("--latent_mult", type=int, default=8)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attention_type", type=str, default="node_wise", choices=["node_wise", "edge_wise"])
    parser.add_argument("--radial_weight_rank", type=int, default=None)
    parser.add_argument("--compare", action="store_true", help="Compare multiple configurations")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.compare:
        # Compare multiple configurations
        configs = [
            ProfilerConfig(
                n_atoms=args.n_atoms,
                k_neighbors=16,
                hidden_mult=args.hidden_mult,
                attention_type="node_wise",
                radial_weight_rank=None,
                device=args.device,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            ProfilerConfig(
                n_atoms=args.n_atoms,
                k_neighbors=16,
                hidden_mult=args.hidden_mult,
                attention_type="node_wise",
                radial_weight_rank=16,
                device=args.device,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            ProfilerConfig(
                n_atoms=args.n_atoms,
                k_neighbors=64,
                hidden_mult=args.hidden_mult,
                attention_type="node_wise",
                radial_weight_rank=None,
                device=args.device,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
            ProfilerConfig(
                n_atoms=args.n_atoms,
                k_neighbors=64,
                hidden_mult=args.hidden_mult,
                attention_type="edge_wise",
                radial_weight_rank=None,
                device=args.device,
                warmup=args.warmup,
                iterations=args.iterations,
            ),
        ]
        compare_configurations(configs)
    else:
        # Single configuration
        config = ProfilerConfig(
            n_atoms=args.n_atoms,
            k_neighbors=args.k_neighbors,
            hidden_mult=args.hidden_mult,
            latent_mult=args.latent_mult,
            nheads=args.nheads,
            warmup=args.warmup,
            iterations=args.iterations,
            device=args.device,
            attention_type=args.attention_type,
            radial_weight_rank=args.radial_weight_rank,
        )

        profiler = LayerProfiler(config)
        profiler.profile_all()
        profiler.print_results()


if __name__ == "__main__":
    main()
