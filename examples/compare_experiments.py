"""Multi-GPU experiment comparison runner.

This script runs multiple experiments in parallel across available GPUs
and outputs a comparison of results. Reports aligned RMSD in Angstroms
using ciffy.rmsd (Kabsch superposition).

Usage:
    python examples/compare_experiments.py \\
        --data_dir /path/to/structures \\
        --output_dir outputs/comparison \\
        --epochs 50

    # With specific GPUs
    python examples/compare_experiments.py \\
        --data_dir /path/to/structures \\
        --gpus 0,1,2,3

Runs 6 experiments in parallel:
- k=16, full_rank and rank=16
- k=64, full_rank and rank=16
- k=128, full_rank and rank=16
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# Experiment configurations to compare
EXPERIMENT_GRID = [
    {"name": "k16_fullrank", "k_neighbors": 16, "radial_weight_rank": None},
    {"name": "k16_rank16", "k_neighbors": 16, "radial_weight_rank": 16},
    {"name": "k64_fullrank", "k_neighbors": 64, "radial_weight_rank": None},
    {"name": "k64_rank16", "k_neighbors": 64, "radial_weight_rank": 16},
    {"name": "k128_fullrank", "k_neighbors": 128, "radial_weight_rank": None},
    {"name": "k128_rank16", "k_neighbors": 128, "radial_weight_rank": 16},
]


def get_available_gpus() -> list[int]:
    """Get list of available CUDA device indices.

    Returns:
        List of GPU indices, or empty list if CUDA not available.
    """
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def run_single_experiment(
    exp_config: dict[str, Any],
    gpu_id: int,
    base_config: str,
    output_dir: str,
    extra_args: list[str],
) -> dict[str, Any]:
    """Run a single experiment on specified GPU.

    Args:
        exp_config: Experiment configuration dict with name, k_neighbors, radial_weight_rank.
        gpu_id: GPU device index to use.
        base_config: Path to base YAML config file.
        output_dir: Directory for experiment outputs.
        extra_args: Additional CLI arguments to pass.

    Returns:
        Dictionary with experiment name, return code, and output.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_experiment.py"),
        "--config", base_config,
        "--name", exp_config["name"],
        "--k_neighbors", str(exp_config["k_neighbors"]),
        "--output_dir", output_dir,
    ]

    # Add radial_weight_rank if specified
    if exp_config["radial_weight_rank"] is not None:
        cmd.extend(["--radial_weight_rank", str(exp_config["radial_weight_rank"])])

    # Add extra arguments
    cmd.extend(extra_args)

    print(f"[GPU {gpu_id}] Starting {exp_config['name']}...")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600 * 24,  # 24 hour timeout
        )
        return {
            "name": exp_config["name"],
            "k_neighbors": exp_config["k_neighbors"],
            "radial_weight_rank": exp_config["radial_weight_rank"],
            "gpu_id": gpu_id,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "name": exp_config["name"],
            "k_neighbors": exp_config["k_neighbors"],
            "radial_weight_rank": exp_config["radial_weight_rank"],
            "gpu_id": gpu_id,
            "returncode": -1,
            "error": "Timeout expired",
        }
    except Exception as e:
        return {
            "name": exp_config["name"],
            "k_neighbors": exp_config["k_neighbors"],
            "radial_weight_rank": exp_config["radial_weight_rank"],
            "gpu_id": gpu_id,
            "returncode": -1,
            "error": str(e),
        }


def run_all_experiments(
    experiments: list[dict[str, Any]],
    gpus: list[int],
    base_config: str,
    output_dir: str,
    extra_args: list[str],
) -> list[dict[str, Any]]:
    """Run all experiments concurrently across GPUs.

    Args:
        experiments: List of experiment configurations.
        gpus: List of GPU indices to use.
        base_config: Path to base YAML config file.
        output_dir: Directory for experiment outputs.
        extra_args: Additional CLI arguments.

    Returns:
        List of result dictionaries.
    """
    results = []

    if not gpus:
        # CPU mode - run sequentially
        print("No GPUs available, running experiments sequentially on CPU...")
        for exp in experiments:
            result = run_single_experiment(
                exp, -1, base_config, output_dir, extra_args + ["--device", "cpu"]
            )
            status = "completed" if result["returncode"] == 0 else "FAILED"
            print(f"[CPU] {exp['name']} {status}")
            results.append(result)
        return results

    # GPU mode - run in parallel
    print(f"Running {len(experiments)} experiments across {len(gpus)} GPUs...")
    print()

    # Assign experiments to GPUs round-robin
    assignments = [
        (exp, gpus[i % len(gpus)])
        for i, exp in enumerate(experiments)
    ]

    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(
                run_single_experiment,
                exp,
                gpu,
                base_config,
                output_dir,
                extra_args,
            ): exp["name"]
            for exp, gpu in assignments
        }

        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                status = "completed" if result["returncode"] == 0 else "FAILED"
                print(f"[GPU {result['gpu_id']}] {name} {status}")
                results.append(result)
            except Exception as e:
                print(f"[ERROR] {name} failed with exception: {e}")
                results.append({
                    "name": name,
                    "returncode": -1,
                    "error": str(e),
                })

    return results


def compare_results(
    output_dir: str,
    experiments: list[dict[str, Any]],
) -> None:
    """Load results and print comparison table.

    Args:
        output_dir: Directory containing experiment outputs.
        experiments: List of experiment configurations.
    """
    output_path = Path(output_dir)
    results = []

    for exp in experiments:
        # Find the experiment output directory (most recent)
        exp_dirs = sorted(output_path.glob(f"{exp['name']}_*"))

        if not exp_dirs:
            print(f"Warning: No output found for {exp['name']}")
            continue

        exp_dir = exp_dirs[-1]  # Most recent
        results_file = exp_dir / "results.pt"

        if not results_file.exists():
            print(f"Warning: No results.pt found for {exp['name']}")
            continue

        try:
            data = torch.load(results_file, map_location="cpu")
            results.append({
                "name": exp["name"],
                "k": exp["k_neighbors"],
                "rank": exp["radial_weight_rank"] or "full",
                "best_val_loss": data.get("best_val_loss"),
                "best_val_rmsd": data.get("best_val_rmsd"),
                "test_rmsd": data.get("test", {}).get("rmsd"),
                "test_loss": data.get("test", {}).get("loss"),
                "best_epoch": data.get("best_epoch"),
                "avg_epoch_time": data.get("avg_epoch_time"),
            })
        except Exception as e:
            print(f"Warning: Could not load results for {exp['name']}: {e}")

    if not results:
        print("\nNo results to compare.")
        return

    # Sort by test RMSD (best first), fallback to val RMSD
    results.sort(key=lambda x: x.get("test_rmsd") or x.get("best_val_rmsd") or float("inf"))

    # Print comparison table
    print()
    print("=" * 90)
    print("EXPERIMENT COMPARISON (RMSD in Angstroms, aligned via Kabsch)")
    print("=" * 90)
    print(
        f"{'Name':<20} {'k':<6} {'Rank':<8} "
        f"{'Val RMSD':<12} {'Test RMSD':<12} {'Epoch':<8} {'Time/Ep':<10}"
    )
    print("-" * 90)

    for r in results:
        val_rmsd = f"{r['best_val_rmsd']:.2f}Å" if r.get("best_val_rmsd") else "N/A"
        test_rmsd = f"{r['test_rmsd']:.2f}Å" if r.get("test_rmsd") else "N/A"
        epoch = str(r.get("best_epoch", "N/A"))
        epoch_time = f"{r['avg_epoch_time']:.1f}s" if r.get("avg_epoch_time") else "N/A"
        print(
            f"{r['name']:<20} {r['k']:<6} {str(r['rank']):<8} "
            f"{val_rmsd:<12} {test_rmsd:<12} {epoch:<8} {epoch_time:<10}"
        )

    print("=" * 90)

    # Save summary to file
    summary_file = output_path / "comparison_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Experiment Comparison - {datetime.now().isoformat()}\n")
        f.write("RMSD values are aligned (Kabsch) and in Angstroms\n")
        f.write("=" * 78 + "\n\n")

        for r in results:
            f.write(f"{r['name']}:\n")
            f.write(f"  k_neighbors: {r['k']}\n")
            f.write(f"  radial_weight_rank: {r['rank']}\n")
            f.write(f"  best_val_rmsd: {r.get('best_val_rmsd')}\n")
            f.write(f"  test_rmsd: {r.get('test_rmsd')}\n")
            f.write(f"  best_val_loss: {r.get('best_val_loss')}\n")
            f.write(f"  test_loss: {r.get('test_loss')}\n")
            f.write(f"  best_epoch: {r.get('best_epoch')}\n")
            f.write(f"  avg_epoch_time: {r.get('avg_epoch_time')}\n")
            f.write("\n")

    print(f"\nSummary saved to: {summary_file}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comparison experiments across multiple GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/comparison_base.yaml",
        help="Base configuration file (default: configs/comparison_base.yaml)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing structure data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/comparison",
        help="Output directory for results (default: outputs/comparison)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (default: all available)",
    )
    parser.add_argument(
        "--num_structures",
        type=int,
        default=None,
        help="Maximum number of structures to load",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 78)
    print("Multi-GPU Experiment Comparison")
    print("=" * 78)
    print()

    # Determine GPUs to use
    if args.gpus:
        gpus = [int(g.strip()) for g in args.gpus.split(",")]
        print(f"Using specified GPUs: {gpus}")
    else:
        gpus = get_available_gpus()
        if gpus:
            print(f"Detected {len(gpus)} GPUs: {gpus}")
        else:
            print("No GPUs detected, will run on CPU")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build extra arguments
    extra_args = ["--epochs", str(args.epochs), "--data_dir", args.data_dir]
    if args.num_structures:
        extra_args.extend(["--num_structures", str(args.num_structures)])

    # Verify base config exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        print("Creating a default config...")
        # Will proceed anyway, run_experiment.py will use defaults
        print()

    # Run experiments
    print("Experiments to run:")
    for exp in EXPERIMENT_GRID:
        rank_str = exp["radial_weight_rank"] or "full"
        print(f"  - {exp['name']}: k={exp['k_neighbors']}, rank={rank_str}")
    print()

    results = run_all_experiments(
        EXPERIMENT_GRID,
        gpus,
        args.config,
        str(output_dir),
        extra_args,
    )

    # Check for failures
    failures = [r for r in results if r.get("returncode", 0) != 0]
    if failures:
        print()
        print("=" * 78)
        print(f"WARNING: {len(failures)} experiment(s) failed:")
        for f in failures:
            print(f"  - {f['name']}: {f.get('error', 'Unknown error')}")
            if f.get("stderr"):
                # Print last few lines of stderr
                stderr_lines = f["stderr"].strip().split("\n")[-5:]
                for line in stderr_lines:
                    print(f"      {line}")
        print("=" * 78)

    # Compare results
    compare_results(str(output_dir), EXPERIMENT_GRID)


if __name__ == "__main__":
    main()
