"""Experiment management for running multiple experiments.

Provides an ExperimentManager class that runs experiments in a single process
with support for multi-GPU parallelism via threading.
"""
from __future__ import annotations

import copy
import queue
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from lr_geom.config import ExperimentConfig, save_config
from lr_geom.data import StructureDataset
from lr_geom.training.utils import set_seed


@dataclass
class ExperimentResult:
    """Results from a single experiment.

    Attributes:
        name: Experiment name.
        config: Configuration used for this experiment.
        history: Training history with "train" and "val" keys.
        best_epoch: Epoch with best validation loss.
        best_val_loss: Best validation loss achieved.
        best_val_rmsd: RMSD at best epoch.
        test_metrics: Test set metrics (if evaluated).
        status: Current status of the experiment.
        error: Error message if failed.
        output_dir: Directory where results were saved.
    """
    name: str
    config: ExperimentConfig | None = None
    history: dict[str, list[dict]] = field(default_factory=dict)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_rmsd: float = float("inf")
    test_metrics: dict[str, float] | None = None
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    error: str | None = None
    output_dir: Path | None = None


class ExperimentManager:
    """Manage multiple experiments with real-time progress tracking.

    Runs experiments in a single process using threading for multi-GPU
    parallelism. Provides independent progress bars per experiment and
    direct access to results without file system polling.

    Example:
        manager = ExperimentManager(
            base_config=load_config("configs/base.yaml"),
            output_dir="outputs/comparison",
            gpus=[0, 1],
        )
        manager.add("node_rank2", k_neighbors=64, radial_weight_rank=2)
        manager.add("node_rank8", k_neighbors=64, radial_weight_rank=8)

        results = manager.run(parallel=True)
        manager.print_comparison()
    """

    def __init__(
        self,
        base_config: ExperimentConfig,
        output_dir: str | Path,
        gpus: list[int] | None = None,
    ) -> None:
        """Initialize experiment manager.

        Args:
            base_config: Base configuration to use for all experiments.
            output_dir: Directory to save experiment outputs.
            gpus: List of GPU indices to use. None = auto-detect available GPUs.
        """
        self.base_config = base_config
        self.output_dir = Path(output_dir)
        self.gpus = gpus if gpus is not None else self._detect_gpus()

        self._experiments: dict[str, ExperimentConfig] = {}
        self._results: dict[str, ExperimentResult] = {}
        self._lock = Lock()
        self._gpu_queue: queue.Queue[int] = queue.Queue()
        self._dataset: StructureDataset | None = None

    def _detect_gpus(self) -> list[int]:
        """Detect available GPUs."""
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return [-1]  # CPU mode

    def add(self, name: str, **overrides: Any) -> None:
        """Add an experiment with config overrides.

        Args:
            name: Unique name for this experiment.
            **overrides: Config overrides. Keys can be flat (e.g., k_neighbors=64)
                or nested via dicts (e.g., model={"k_neighbors": 64}).
        """
        if name in self._experiments:
            raise ValueError(f"Experiment '{name}' already exists")

        config = self._apply_overrides(self.base_config, name, overrides)
        self._experiments[name] = config
        self._results[name] = ExperimentResult(name=name, config=config)

    def _apply_overrides(
        self,
        base: ExperimentConfig,
        name: str,
        overrides: dict[str, Any],
    ) -> ExperimentConfig:
        """Apply config overrides to create new config."""
        # Deep copy to avoid mutating base
        config = copy.deepcopy(base)
        config.name = name

        # Apply overrides to appropriate nested config
        for key, value in overrides.items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
            elif hasattr(config.training, key):
                setattr(config.training, key, value)
            elif hasattr(config.data, key):
                setattr(config.data, key, value)
            elif hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")

        return config

    def run(self, parallel: bool = True) -> dict[str, ExperimentResult]:
        """Run all experiments.

        Args:
            parallel: If True, run experiments in parallel across GPUs.

        Returns:
            Dictionary mapping experiment names to results.
        """
        if not self._experiments:
            raise ValueError("No experiments added. Use add() first.")

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.output_dir / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load shared dataset once
        print("Loading dataset...")
        self._dataset = self._load_dataset()

        # Initialize GPU queue
        for gpu in self.gpus:
            self._gpu_queue.put(gpu)

        # Create progress bars (one per experiment)
        # Reserve space for all progress bars
        print()  # Blank line before progress bars
        pbars: dict[str, tqdm] = {}
        for i, (name, cfg) in enumerate(self._experiments.items()):
            pbars[name] = tqdm(
                total=cfg.training.epochs,
                desc=f"{name:<20}",
                position=i,
                leave=True,
                ncols=100,
            )

        try:
            if parallel and len(self.gpus) > 1 and len(self._experiments) > 1:
                # Thread pool with one worker per GPU
                with ThreadPoolExecutor(max_workers=len(self.gpus)) as executor:
                    futures = {
                        executor.submit(self._run_one, name, pbars[name]): name
                        for name in self._experiments
                    }
                    for future in as_completed(futures):
                        name = futures[future]
                        try:
                            future.result()
                        except Exception as e:
                            self._mark_failed(name, str(e))
            else:
                # Sequential execution
                for name in self._experiments:
                    try:
                        self._run_one(name, pbars[name])
                    except Exception as e:
                        self._mark_failed(name, str(e))
        finally:
            # Close progress bars
            for pbar in pbars.values():
                pbar.close()

        # Print summary
        print("\n" * len(self._experiments))  # Clear space after progress bars
        return self._results

    def _mark_failed(self, name: str, error: str) -> None:
        """Mark an experiment as failed."""
        with self._lock:
            self._results[name].status = "failed"
            self._results[name].error = error

    def _run_one(self, name: str, pbar: tqdm) -> None:
        """Run single experiment with GPU from pool."""
        gpu = self._gpu_queue.get()
        try:
            device = torch.device(f"cuda:{gpu}" if gpu >= 0 else "cpu")
            result = self._train(name, device, pbar)
            with self._lock:
                self._results[name] = result
        except Exception as e:
            with self._lock:
                self._results[name].status = "failed"
                self._results[name].error = str(e)
            raise
        finally:
            self._gpu_queue.put(gpu)

    def _load_dataset(self) -> StructureDataset:
        """Load dataset from config."""
        config = self.base_config
        level = "residue" if config.data.residue_level else "atom"
        return StructureDataset.from_directory(
            config.data.data_dir,
            scale="chain",
            level=level,
            max_atoms=config.data.max_atoms,
        )

    def _train(
        self,
        name: str,
        device: torch.device,
        pbar: tqdm,
    ) -> ExperimentResult:
        """Training loop for a single experiment."""
        # Import here to avoid circular imports
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "examples"))
        from run_experiment import build_model, train_epoch, evaluate, get_kl_weight

        config = self._experiments[name]
        set_seed(config.seed)

        # Create experiment output directory
        exp_dir = self.output_dir / name
        exp_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, exp_dir / "config.yaml")

        # Split dataset and move to device
        train_data, val_data, test_data = self._dataset.split(
            train=config.data.train_split,
            val=config.data.val_split,
            seed=config.data.seed,
        )
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        test_data = test_data.to(device)

        # Build model
        embedding, vae = build_model(config, self._dataset.num_feature_types, device)

        # Setup optimizer and scheduler
        params = list(embedding.parameters()) + list(vae.parameters())
        optimizer = optim.AdamW(
            params,
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        scheduler = None
        if config.training.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training.epochs,
                eta_min=config.training.min_lr,
            )

        # Training loop
        history: dict[str, list[dict]] = {"train": [], "val": []}
        best_val_loss = float("inf")
        best_epoch = 0
        patience = 0

        result = ExperimentResult(
            name=name,
            config=config,
            status="running",
            output_dir=exp_dir,
        )

        for epoch in range(1, config.training.epochs + 1):
            kl_weight = get_kl_weight(epoch, config)

            # Train
            train_metrics = train_epoch(
                embedding, vae, train_data, optimizer,
                kl_weight, config.training.distance_weight,
                config.training.grad_clip,
            )
            history["train"].append(train_metrics)

            # Validate
            val_metrics = evaluate(
                embedding, vae, val_data,
                kl_weight, config.training.distance_weight,
            )
            history["val"].append(val_metrics)

            # Scheduler step
            if scheduler:
                scheduler.step()

            # Check for best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                patience = 0
                torch.save({
                    "embedding": embedding.state_dict(),
                    "vae": vae.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                    "val_rmsd": val_metrics["rmsd"],
                }, exp_dir / "best_model.pt")
            else:
                patience += 1

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                "rmsd": f"{val_metrics['rmsd']:.2f}Å",
                "best": f"{history['val'][best_epoch-1]['rmsd']:.2f}Å",
            })

            # Early stopping
            if config.training.early_stopping > 0 and patience >= config.training.early_stopping:
                break

        # Load best model for test evaluation
        checkpoint = torch.load(exp_dir / "best_model.pt", weights_only=True)
        embedding.load_state_dict(checkpoint["embedding"])
        vae.load_state_dict(checkpoint["vae"])

        # Test evaluation
        test_metrics = evaluate(
            embedding, vae, test_data,
            config.training.kl_weight, config.training.distance_weight,
        )

        # Save final results
        results_dict = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_rmsd": history["val"][best_epoch - 1]["rmsd"],
            "test": test_metrics,
            "history": history,
        }
        torch.save(results_dict, exp_dir / "results.pt")

        return ExperimentResult(
            name=name,
            config=config,
            history=history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_val_rmsd=history["val"][best_epoch - 1]["rmsd"],
            test_metrics=test_metrics,
            status="completed",
            output_dir=exp_dir,
        )

    @property
    def results(self) -> dict[str, ExperimentResult]:
        """Get all experiment results."""
        return self._results

    def print_comparison(self) -> None:
        """Print formatted comparison table of results."""
        print()
        print("=" * 90)
        print("RESULTS (RMSD in Angstroms)")
        print("=" * 90)
        print(f"{'Name':<20} {'Attn':<6} {'Rank':<6} {'Val RMSD':<12} {'Test RMSD':<12} {'Epoch':<8}")
        print("-" * 90)

        # Sort by test RMSD
        sorted_results = sorted(
            self._results.values(),
            key=lambda r: r.test_metrics.get("rmsd", float("inf")) if r.test_metrics else float("inf"),
        )

        for result in sorted_results:
            if result.status == "failed":
                print(f"{result.name:<20} {'—':<6} {'—':<6} {'FAILED':<12} {result.error or ''}")
                continue

            attn = result.config.model.attention_type[:4] if result.config else "—"
            rank = result.config.model.radial_weight_rank if result.config else "—"
            rank_str = str(rank) if rank is not None else "full"
            val_rmsd = f"{result.best_val_rmsd:.2f}Å" if result.best_val_rmsd < float("inf") else "—"
            test_rmsd = f"{result.test_metrics['rmsd']:.2f}Å" if result.test_metrics else "—"
            epoch = str(result.best_epoch) if result.best_epoch > 0 else "—"

            print(f"{result.name:<20} {attn:<6} {rank_str:<6} {val_rmsd:<12} {test_rmsd:<12} {epoch:<8}")

        print("=" * 90)
        print(f"Results saved to: {self.output_dir}")
