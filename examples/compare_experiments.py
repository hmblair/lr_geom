"""Run multiple experiments and compare results.

Uses ExperimentManager for in-process execution with real-time progress.

Usage:
    python examples/compare_experiments.py \
        --data_dir /path/to/structures \
        --output_dir outputs/comparison \
        --epochs 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lr_geom.config import load_config
from lr_geom.training import ExperimentManager

# Experiment configurations
EXPERIMENTS = [
    {"name": "node_rank2", "k_neighbors": 64, "radial_weight_rank": 2, "attention_type": "node_wise"},
    {"name": "node_rank8", "k_neighbors": 64, "radial_weight_rank": 8, "attention_type": "node_wise"},
    {"name": "node_rank32", "k_neighbors": 64, "radial_weight_rank": 32, "attention_type": "node_wise"},
    {"name": "edge_rank2", "k_neighbors": 64, "radial_weight_rank": 2, "attention_type": "edge_wise"},
    {"name": "edge_rank8", "k_neighbors": 64, "radial_weight_rank": 8, "attention_type": "edge_wise"},
    {"name": "edge_rank32", "k_neighbors": 64, "radial_weight_rank": 32, "attention_type": "edge_wise"},
]


def main():
    parser = argparse.ArgumentParser(description="Run comparison experiments")
    parser.add_argument("--config", default="configs/comparison_base.yaml")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/comparison")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gpus", type=str, help="Comma-separated GPU IDs")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--residue_level", action="store_true")
    parser.add_argument("--kl_weight", type=float)
    parser.add_argument("--distance_weight", type=float)
    parser.add_argument("--sequential", action="store_true", help="Run experiments sequentially")
    args = parser.parse_args()

    # Parse GPUs
    gpus = None
    if args.gpus:
        gpus = [int(g) for g in args.gpus.split(",")]

    # Load and modify base config
    config = load_config(args.config)
    config.data.data_dir = args.data_dir
    config.training.epochs = args.epochs
    if args.residue_level:
        config.data.residue_level = True
    if args.kl_weight is not None:
        config.training.kl_weight = args.kl_weight
    if args.distance_weight is not None:
        config.training.distance_weight = args.distance_weight
    if args.no_compile:
        config.model.use_compile = False

    # Create manager
    manager = ExperimentManager(
        base_config=config,
        output_dir=args.output_dir,
        gpus=gpus,
    )

    # Add experiments
    experiments = EXPERIMENTS.copy()
    if args.residue_level:
        experiments = [{**e, "name": f"res_{e['name']}"} for e in experiments]

    for exp in experiments:
        manager.add(
            exp["name"],
            k_neighbors=exp["k_neighbors"],
            radial_weight_rank=exp["radial_weight_rank"],
            attention_type=exp["attention_type"],
        )

    # Print header
    print("=" * 60)
    print("Experiment Comparison")
    print("=" * 60)
    print(f"GPUs: {manager.gpus}")
    print(f"Experiments: {len(experiments)}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print()

    # Run experiments
    try:
        results = manager.run(parallel=not args.sequential)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print comparison
    manager.print_comparison()

    # Summary
    completed = sum(1 for r in results.values() if r.status == "completed")
    failed = sum(1 for r in results.values() if r.status == "failed")
    print(f"\nCompleted: {completed}, Failed: {failed}")


if __name__ == "__main__":
    main()
