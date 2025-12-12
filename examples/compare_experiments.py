"""Run multiple experiments in parallel and compare results.

Usage:
    python examples/compare_experiments.py \
        --data_dir /path/to/structures \
        --output_dir outputs/comparison \
        --epochs 50
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import torch

# Experiment configurations
EXPERIMENTS = [
    {"name": "node_rank2", "k_neighbors": 64, "radial_weight_rank": 2, "attention_type": "node_wise"},
    {"name": "node_rank8", "k_neighbors": 64, "radial_weight_rank": 8, "attention_type": "node_wise"},
    {"name": "node_rank32", "k_neighbors": 64, "radial_weight_rank": 32, "attention_type": "node_wise"},
    {"name": "edge_rank2", "k_neighbors": 64, "radial_weight_rank": 2, "attention_type": "edge_wise"},
    {"name": "edge_rank8", "k_neighbors": 64, "radial_weight_rank": 8, "attention_type": "edge_wise"},
    {"name": "edge_rank32", "k_neighbors": 64, "radial_weight_rank": 32, "attention_type": "edge_wise"},
]


def get_gpus() -> list[int]:
    """Get available GPU indices."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def run_experiment(
    exp: dict,
    gpu: int,
    config_path: str,
    output_dir: str,
    extra_args: list[str],
) -> dict:
    """Run a single experiment."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_experiment.py"),
        "--config", config_path,
        "--name", exp["name"],
        "--k_neighbors", str(exp["k_neighbors"]),
        "--attention_type", exp["attention_type"],
        "--output_dir", output_dir,
    ]
    if exp["radial_weight_rank"]:
        cmd.extend(["--radial_weight_rank", str(exp["radial_weight_rank"])])
    cmd.extend(extra_args)

    print(f"[GPU {gpu}] Starting {exp['name']}")

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=3600 * 12
        )

        # Save log
        log_path = Path(output_dir) / f"{exp['name']}_log.txt"
        with open(log_path, "w") as f:
            f.write(f"=== {exp['name']} ===\n")
            f.write(f"GPU: {gpu}\n")
            f.write(f"Return code: {result.returncode}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)

        return {
            "name": exp["name"],
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"name": exp["name"], "returncode": -1, "error": "Timeout"}
    except Exception as e:
        return {"name": exp["name"], "returncode": -1, "error": str(e)}


def find_results(output_dir: Path, name: str) -> Path | None:
    """Find results.pt for an experiment."""
    dirs = sorted(output_dir.glob(f"{name}_*"))
    dirs = [d for d in dirs if d.is_dir()]
    if not dirs:
        return None
    results_path = dirs[-1] / "results.pt"
    return results_path if results_path.exists() else None


def print_comparison(output_dir: Path, experiments: list[dict]):
    """Print comparison table of results."""
    print()
    print("=" * 90)
    print("RESULTS (RMSD in Angstroms)")
    print("=" * 90)
    print(f"{'Name':<20} {'Attn':<6} {'Rank':<6} {'Val RMSD':<12} {'Test RMSD':<12} {'Epoch':<8}")
    print("-" * 90)

    results = []
    for exp in experiments:
        results_path = find_results(output_dir, exp["name"])
        if not results_path:
            print(f"{exp['name']:<20} {'—':<6} {'—':<6} {'MISSING':<12}")
            continue

        data = torch.load(results_path, map_location="cpu")
        results.append({
            "name": exp["name"],
            "attn": exp["attention_type"][:4],
            "rank": exp["radial_weight_rank"] or "full",
            "val_rmsd": data.get("best_val_rmsd"),
            "test_rmsd": data.get("test", {}).get("rmsd"),
            "epoch": data.get("best_epoch"),
        })

    # Sort by test RMSD
    results.sort(key=lambda x: x.get("test_rmsd") or float("inf"))

    for r in results:
        val = f"{r['val_rmsd']:.2f}Å" if r.get("val_rmsd") else "—"
        test = f"{r['test_rmsd']:.2f}Å" if r.get("test_rmsd") else "—"
        print(f"{r['name']:<20} {r['attn']:<6} {str(r['rank']):<6} {val:<12} {test:<12} {r.get('epoch', '—'):<8}")

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Run comparison experiments")
    parser.add_argument("--config", default="configs/comparison_base.yaml")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/comparison")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--gpus", type=str, help="Comma-separated GPU IDs")
    parser.add_argument("--num_structures", type=int)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--residue_level", action="store_true")
    parser.add_argument("--kl_weight", type=float)
    parser.add_argument("--distance_weight", type=float)
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else get_gpus()
    if not gpus:
        print("No GPUs available, using CPU")
        gpus = [-1]  # CPU mode

    # Build extra args
    extra_args = ["--epochs", str(args.epochs), "--data_dir", args.data_dir]
    if args.num_structures:
        extra_args.extend(["--num_structures", str(args.num_structures)])
    if args.no_compile:
        extra_args.append("--no-compile")
    if args.residue_level:
        extra_args.append("--residue_level")
    if args.kl_weight:
        extra_args.extend(["--kl_weight", str(args.kl_weight)])
    if args.distance_weight:
        extra_args.extend(["--distance_weight", str(args.distance_weight)])

    # Modify experiment names for residue level
    experiments = EXPERIMENTS.copy()
    if args.residue_level:
        experiments = [{**e, "name": f"res_{e['name']}"} for e in experiments]

    print("=" * 60)
    print("Experiment Comparison")
    print("=" * 60)
    print(f"GPUs: {gpus}")
    print(f"Experiments: {len(experiments)}")
    print(f"Output: {output_dir}")
    print()

    # Run experiments in parallel
    results = []
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(
                run_experiment,
                exp,
                gpus[i % len(gpus)],
                args.config,
                str(output_dir),
                extra_args,
            ): exp
            for i, exp in enumerate(experiments)
        }

        for future in as_completed(futures):
            exp = futures[future]
            result = future.result()
            results.append(result)

            if result["returncode"] == 0:
                print(f"[DONE] {exp['name']}")
            else:
                print(f"[FAIL] {exp['name']}: {result.get('error', 'see log')}")

    # Summary
    succeeded = sum(1 for r in results if r["returncode"] == 0)
    failed = len(results) - succeeded
    print()
    print(f"Completed: {succeeded}, Failed: {failed}")

    # Print comparison
    print_comparison(output_dir, experiments)


if __name__ == "__main__":
    main()
