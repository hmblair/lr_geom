"""Run experiments from YAML configuration files.

This script provides a simple way to run reproducible training experiments
using configuration files. Uses the lr_geom.training module for a clean,
callback-based training loop.

Usage:
    python examples/run_experiment.py --config configs/baseline.yaml
    python examples/run_experiment.py --config configs/edge_wise.yaml --epochs 50
"""
from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import lr_geom as lg
from lr_geom.config import (
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    DataConfig,
    load_config,
    save_config,
    merge_config_with_args,
)
from lr_geom.data import StructureDataset, Structure
from lr_geom.vae import EquivariantVAE, kl_divergence
from lr_geom.training import set_seed, get_device


def get_kl_weight(
    epoch: int,
    config: TrainingConfig,
) -> float:
    """Get KL weight for current epoch based on annealing strategy.

    Args:
        epoch: Current epoch (1-indexed).
        config: Training configuration.

    Returns:
        KL weight for this epoch.
    """
    if config.kl_annealing == "none":
        return config.kl_weight

    elif config.kl_annealing == "linear":
        # Linear warmup from 0 to kl_weight over kl_warmup_epochs
        if epoch >= config.kl_warmup_epochs:
            return config.kl_weight
        return config.kl_weight * (epoch / config.kl_warmup_epochs)

    elif config.kl_annealing == "cyclical":
        # Cyclical annealing: repeat linear warmup every kl_cycle_epochs
        # This helps the model repeatedly "re-learn" to use the latent space
        cycle_position = epoch % config.kl_cycle_epochs
        if cycle_position == 0:
            cycle_position = config.kl_cycle_epochs
        # Linear ramp within each cycle
        ramp_epochs = config.kl_cycle_epochs // 2  # First half is ramp
        if cycle_position <= ramp_epochs:
            return config.kl_weight * (cycle_position / ramp_epochs)
        return config.kl_weight

    return config.kl_weight


def compute_kl_with_free_bits(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """Compute KL divergence with free bits to prevent posterior collapse.

    Free bits ensures a minimum KL per latent dimension, which encourages
    the model to use the latent space rather than ignoring it.

    Args:
        mu: Latent mean, shape (N, mult, repr_dim).
        logvar: Latent log-variance, shape (N, mult).
        free_bits: Minimum KL per dimension (lambda in the paper).

    Returns:
        Scalar KL divergence loss.
    """
    # Expand logvar to match mu dimensions: (N, mult) -> (N, mult, 1)
    logvar_expanded = logvar.unsqueeze(-1)

    # Standard KL: -0.5 * (1 + logvar - mu^2 - exp(logvar))
    # Computed per element then summed
    kl_per_dim = -0.5 * (1 + logvar_expanded - mu.pow(2) - logvar_expanded.exp())

    if free_bits > 0:
        # Apply free bits: max(kl, free_bits) per dimension
        # This ensures each dimension contributes at least free_bits to loss
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

    # Sum over repr dims and mult, mean over batch
    return kl_per_dim.sum(dim=(-1, -2)).mean()


def compute_distance_loss(
    true_polymer,
    pred_polymer,
    scale=None,
) -> torch.Tensor:
    """Compute pairwise distance matrix loss for local structure preservation.

    This loss encourages the model to preserve local pairwise distances,
    not just global fit (RMSD). Helps with bond lengths and local geometry.

    Args:
        true_polymer: Ground truth polymer.
        pred_polymer: Predicted polymer (with predicted coordinates).
        scale: Distance scale (e.g., ciffy.RESIDUE for residue-level distances).
            If None, uses atom-level distances.

    Returns:
        Scalar distance matrix loss (mean absolute error).
    """
    # Get pairwise distance matrices
    if scale is not None:
        D_true = true_polymer.pairwise_distance(scale)
        D_pred = pred_polymer.pairwise_distance(scale)
    else:
        D_true = true_polymer.pairwise_distance()
        D_pred = pred_polymer.pairwise_distance()

    # Mean absolute error on distance matrices
    return (D_pred - D_true).abs().mean()


class WarmupScheduler:
    """Learning rate scheduler with linear warmup.

    Wraps another scheduler and applies linear warmup for the first N epochs.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        base_scheduler: optim.lr_scheduler.LRScheduler | None = None,
    ):
        """Initialize warmup scheduler.

        Args:
            optimizer: The optimizer.
            warmup_epochs: Number of warmup epochs.
            base_scheduler: Scheduler to use after warmup (optional).
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self) -> None:
        """Update learning rate."""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            warmup_factor = self.current_epoch / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor
        elif self.base_scheduler is not None:
            # After warmup, use base scheduler
            self.base_scheduler.step()

    def get_last_lr(self) -> list[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


def build_model_from_config(
    config: ModelConfig,
    num_atom_types: int,
    device: torch.device,
) -> tuple[nn.Embedding, EquivariantVAE]:
    """Build model from configuration.

    Args:
        config: Model configuration.
        num_atom_types: Number of atom types for embedding.
        device: Device to place model on.

    Returns:
        Tuple of (embedding, vae) modules.
    """
    # Build representations
    lvals_hidden = list(range(config.lmax_hidden + 1))
    lvals_latent = list(range(config.lmax_latent + 1))

    in_repr = lg.Repr([0], mult=config.embed_dim)
    hidden_repr = lg.Repr(lvals_hidden, mult=config.hidden_mult)
    latent_repr = lg.Repr(lvals_latent, mult=config.latent_mult)
    out_repr = lg.Repr([1], mult=1)

    # Create embedding
    embedding = nn.Embedding(num_atom_types, config.embed_dim).to(device)

    # Create VAE
    vae = EquivariantVAE(
        in_repr=in_repr,
        latent_repr=latent_repr,
        out_repr=out_repr,
        hidden_repr=hidden_repr,
        encoder_layers=config.encoder_layers,
        decoder_layers=config.decoder_layers,
        k_neighbors=config.k_neighbors,
        nheads=config.nheads,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
        residual_scale=config.residual_scale,
        attention_type=config.attention_type,
        scale_type=config.scale_type,
        skip_type=config.skip_type,
        rbf_type=config.rbf_type,
        rbf_r_min=config.rbf_r_min,
        rbf_r_max=config.rbf_r_max,
        radial_weight_rank=config.radial_weight_rank,
    ).to(device)

    # Apply torch.compile for ~2x speedup (first forward pass will be slow)
    if config.use_compile:
        vae.compile()

    return embedding, vae


def train_epoch(
    embedding: nn.Embedding,
    vae: EquivariantVAE,
    dataset: StructureDataset,
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    kl_weight: float | None = None,
) -> dict:
    """Train for one epoch.

    Args:
        embedding: Atom embedding module.
        vae: VAE model.
        dataset: Training dataset.
        optimizer: Optimizer.
        config: Training configuration.
        kl_weight: Override KL weight for this epoch (for annealing).

    Returns:
        Dictionary of training metrics.
    """
    import ciffy
    import time

    embedding.train()
    vae.train()

    # Use provided kl_weight or fall back to config
    current_kl_weight = kl_weight if kl_weight is not None else config.kl_weight

    total_loss = 0.0
    total_rmsd = 0.0
    total_kl = 0.0
    total_dist = 0.0
    n_batches = 0

    epoch_start = time.time()

    # Shuffle structures
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    pbar = tqdm(indices, desc="Training", leave=False, ncols=80)
    for idx in pbar:
        s = dataset[idx]

        # Forward pass
        features = embedding(s.features).unsqueeze(-1)
        recon, mu, logvar = vae(s.coords, features)

        # Compute aligned RMSD loss using ciffy (differentiable)
        coords_pred = recon[:, 0, :]
        coords_pred_unnorm = coords_pred * s.coord_scale
        pred_polymer = s.polymer.with_coordinates(coords_pred_unnorm)
        rmsd_loss = ciffy.rmsd(s.polymer, pred_polymer)

        # KL loss with optional free bits
        kl_loss = compute_kl_with_free_bits(mu, logvar, config.free_bits)
        loss = rmsd_loss + current_kl_weight * kl_loss

        # Distance matrix loss for local structure preservation
        if config.distance_weight > 0:
            dist_loss = compute_distance_loss(s.polymer, pred_polymer)
            loss = loss + config.distance_weight * dist_loss
            total_dist += dist_loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(embedding.parameters()) + list(vae.parameters()),
                config.grad_clip,
            )

        optimizer.step()

        total_loss += loss.item()
        total_rmsd += rmsd_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1

        # Update progress bar with running average RMSD
        pbar.set_postfix({"RMSD": f"{total_rmsd / n_batches:.2f}Å"})

    epoch_time = time.time() - epoch_start

    metrics = {
        "loss": total_loss / n_batches,
        "rmsd": total_rmsd / n_batches,
        "kl": total_kl / n_batches,
        "time": epoch_time,
    }
    if config.distance_weight > 0:
        metrics["dist"] = total_dist / n_batches

    return metrics


def evaluate(
    embedding: nn.Embedding,
    vae: EquivariantVAE,
    dataset: StructureDataset,
    config: TrainingConfig,
    kl_weight: float | None = None,
) -> dict:
    """Evaluate model on structures.

    Args:
        embedding: Atom embedding module.
        vae: VAE model.
        dataset: Evaluation dataset.
        config: Training configuration.
        kl_weight: Override KL weight (for consistency with training).

    Returns:
        Dictionary of evaluation metrics.
    """
    import ciffy

    embedding.eval()
    vae.eval()

    # Use provided kl_weight or fall back to config
    current_kl_weight = kl_weight if kl_weight is not None else config.kl_weight

    total_loss = 0.0
    total_rmsd = 0.0
    total_kl = 0.0
    total_dist = 0.0

    with torch.no_grad():
        for s in dataset:
            features = embedding(s.features).unsqueeze(-1)
            recon, mu, logvar = vae(s.coords, features)

            # Compute aligned RMSD using ciffy
            coords_pred = recon[:, 0, :]
            coords_pred_unnorm = coords_pred * s.coord_scale
            pred_polymer = s.polymer.with_coordinates(coords_pred_unnorm)
            rmsd = ciffy.rmsd(s.polymer, pred_polymer)

            # KL loss with optional free bits
            kl_loss = compute_kl_with_free_bits(mu, logvar, config.free_bits)
            loss = rmsd + current_kl_weight * kl_loss

            # Distance matrix loss
            if config.distance_weight > 0:
                dist_loss = compute_distance_loss(s.polymer, pred_polymer)
                loss = loss + config.distance_weight * dist_loss
                total_dist += dist_loss.item()

            total_loss += loss.item()
            total_rmsd += rmsd.item()
            total_kl += kl_loss.item()

    n = len(dataset)
    metrics = {
        "loss": total_loss / n,
        "rmsd": total_rmsd / n,
        "kl": total_kl / n,
    }
    if config.distance_weight > 0:
        metrics["dist"] = total_dist / n

    return metrics


def save_reconstructions(
    embedding: nn.Embedding,
    vae: EquivariantVAE,
    dataset: StructureDataset,
    output_dir: Path,
    num_samples: int = 3,
) -> list[str]:
    """Save original, reconstructed, and sampled structures as .cif files.

    For each structure, saves:
    - Original structure
    - Reconstruction (encode -> decode)
    - Sample (random latent z from prior + conditioning -> decode)

    Args:
        embedding: Atom embedding module.
        vae: VAE model.
        dataset: Dataset of structures.
        output_dir: Directory to save .cif files.
        num_samples: Number of samples to save.

    Returns:
        List of saved structure IDs.
    """
    embedding.eval()
    vae.eval()

    recon_dir = output_dir / "reconstructions"
    recon_dir.mkdir(parents=True, exist_ok=True)

    saved_ids = []
    n_to_save = min(num_samples, len(dataset))

    with torch.no_grad():
        for i in range(n_to_save):
            s = dataset[i]

            # Get features (conditioning)
            features = embedding(s.features).unsqueeze(-1)

            # Reconstruction: encode -> decode
            recon, _, _ = vae(s.coords, features)
            coords_recon = recon[:, 0, :]
            coords_recon_unnorm = coords_recon * s.coord_scale

            # Sample: random z from prior + conditioning -> decode
            sampled = vae.sample(s.coords, features)
            coords_sampled = sampled[:, 0, :]
            coords_sampled_unnorm = coords_sampled * s.coord_scale

            # Create polymers
            recon_polymer = s.polymer.with_coordinates(coords_recon_unnorm)
            sampled_polymer = s.polymer.with_coordinates(coords_sampled_unnorm)

            # Save structures
            safe_id = s.id.replace("/", "_").replace(" ", "_")
            original_path = recon_dir / f"{safe_id}_original.cif"
            recon_path = recon_dir / f"{safe_id}_reconstructed.cif"
            sampled_path = recon_dir / f"{safe_id}_sampled.cif"

            try:
                s.polymer.write(str(original_path))
                recon_polymer.write(str(recon_path))
                sampled_polymer.write(str(sampled_path))
                saved_ids.append(s.id)
                print(f"  Saved reconstruction and sample for {s.id}")
            except Exception as e:
                print(f"  Warning: Could not save {s.id}: {e}")

    return saved_ids


def write_progress(progress_file: str | Path | None, data: dict) -> None:
    """Write progress to a temp file for monitoring.

    Args:
        progress_file: Path to progress file, or None to skip.
        data: Progress data to write.
    """
    if progress_file is None:
        return
    import json
    try:
        with open(progress_file, "w") as f:
            json.dump(data, f)
    except Exception:
        pass  # Ignore write errors


def run_experiment(config: ExperimentConfig, num_recon_samples: int = 3, dry_run: bool = False, progress_file: str | Path | None = None) -> dict:
    """Run a single experiment.

    Args:
        config: Experiment configuration.
        num_recon_samples: Number of test samples to save reconstructions for.
        dry_run: If True, validate setup without training (load data, build model, run one forward pass).
        progress_file: Path to write progress updates for external monitoring.

    Returns:
        Dictionary of results.
    """
    import time
    t0 = time.time()

    # Setup
    set_seed(config.seed)
    device = get_device(config.device)

    print(f"Device: {device}")
    print(f"Experiment: {config.name}")
    print(f"torch.compile: {'enabled' if config.model.use_compile else 'disabled'}")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / f"{config.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    save_config(config, output_dir / "config.yaml")
    print(f"Config saved to: {output_dir / 'config.yaml'}")

    # Load data
    level = "residue" if config.data.residue_level else "atom"
    dataset = StructureDataset.from_directory(
        config.data.data_dir,
        level=level,
        min_nodes=config.data.min_atoms,
        max_nodes=config.data.max_atoms,
        max_structures=config.data.num_structures,
    )

    print(f"  Data loaded in {time.time() - t0:.1f}s")

    if len(dataset) == 0:
        print("No structures loaded! Check data directory.")
        return {"error": "No data"}

    # Split data and move to device
    train_data, val_data, test_data = dataset.split(
        train=config.data.train_split,
        val=config.data.val_split,
        seed=config.data.seed,
    )
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print()

    # Build model
    t_model = time.time()
    embedding, vae = build_model_from_config(config.model, dataset.num_feature_types, device)
    print(f"  Model built in {time.time() - t_model:.1f}s")

    num_params = sum(p.numel() for p in embedding.parameters()) + sum(p.numel() for p in vae.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Attention type: {config.model.attention_type}")
    print(f"Scale type: {config.model.scale_type}")
    print(f"Skip type: {config.model.skip_type}")
    print(f"RBF type: {config.model.rbf_type}")
    print()

    # Dry run: validate setup without training
    if dry_run:
        print("DRY RUN: Validating setup (mirroring full experiment flow)...")
        print()
        embedding.eval()
        vae.eval()

        with torch.no_grad():
            # Run one forward pass with first training structure
            s = train_data[0]
            features = embedding(s.features).unsqueeze(-1)
            recon, mu, logvar = vae(s.coords, features)

            # Also test sampling
            sampled = vae.sample(s.coords, features)

            print(f"Forward pass: OK")
            print(f"  Input coords: {s.coords.shape}")
            print(f"  Features: {features.shape}")
            print(f"  Reconstruction: {recon.shape}")
            print(f"  Latent mu: {mu.shape}")
            print(f"  Sample: {sampled.shape}")
            print()

        # Evaluate on test set (with untrained model)
        if len(test_data) > 0:
            print("Evaluating on test set (untrained model)...")
            test_metrics = evaluate(embedding, vae, test_data, config.training)
            print(f"  Test RMSD: {test_metrics['rmsd']:.2f}Å")
            print(f"  Test Loss: {test_metrics['loss']:.4f}")
            print()

        # Save reconstructions and samples
        if len(test_data) > 0 and num_recon_samples > 0:
            print(f"Saving {num_recon_samples} reconstructions and samples...")
            saved_ids = save_reconstructions(
                embedding, vae, test_data, output_dir, num_recon_samples
            )
            print(f"  Saved {len(saved_ids)} structures to: {output_dir / 'reconstructions'}")
            print()

        print("=" * 60)
        print("DRY RUN COMPLETE: Setup validated successfully!")
        print("=" * 60)

        return {"dry_run": True, "status": "success"}

    # Setup optimizer
    params = list(embedding.parameters()) + list(vae.parameters())
    optimizer = optim.AdamW(
        params,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    # Learning rate scheduler with warmup
    base_scheduler = None
    if config.training.scheduler == "cosine":
        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs - config.training.warmup_epochs,
            eta_min=config.training.min_lr,
        )
    elif config.training.scheduler == "plateau":
        base_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=config.training.min_lr,
        )

    # Wrap with warmup scheduler
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=config.training.warmup_epochs,
        base_scheduler=base_scheduler if config.training.scheduler != "plateau" else None,
    )

    # Print training settings
    print(f"LR warmup: {config.training.warmup_epochs} epochs")
    print(f"KL annealing: {config.training.kl_annealing} (warmup: {config.training.kl_warmup_epochs} epochs)")
    if config.training.free_bits > 0:
        print(f"Free bits: {config.training.free_bits}")
    print()

    # Training loop
    history = {"train": [], "val": [], "kl_weight": [], "lr": []}
    best_val_loss = float("inf")
    best_val_rmsd = float("inf")
    patience_counter = 0

    print(f"Total setup time: {time.time() - t0:.1f}s")
    print()
    print("Training...")
    epoch_pbar = tqdm(range(1, config.training.epochs + 1), desc="Epochs", ncols=100)
    for epoch in epoch_pbar:
        # Get current KL weight based on annealing strategy
        current_kl_weight = get_kl_weight(epoch, config.training)

        # Train
        train_metrics = train_epoch(
            embedding, vae, train_data, optimizer, config.training,
            kl_weight=current_kl_weight,
        )
        history["train"].append(train_metrics)

        # Validate
        if len(val_data) > 0:
            val_metrics = evaluate(embedding, vae, val_data, config.training,
                                   kl_weight=current_kl_weight)
            history["val"].append(val_metrics)
            val_loss = val_metrics["loss"]
            val_rmsd = val_metrics["rmsd"]
        else:
            val_loss = train_metrics["loss"]
            val_rmsd = train_metrics["rmsd"]

        # Track KL weight and LR
        history["kl_weight"].append(current_kl_weight)
        history["lr"].append(scheduler.get_last_lr()[0])

        # Learning rate scheduling
        if config.training.scheduler == "plateau" and base_scheduler is not None:
            # Plateau scheduler needs val_loss
            if epoch > config.training.warmup_epochs:
                base_scheduler.step(val_loss)
        scheduler.step()

        # Early stopping
        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_rmsd = val_rmsd
            patience_counter = 0
            is_best = True
            # Save best model
            torch.save({
                "embedding": embedding.state_dict(),
                "vae": vae.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_rmsd": val_rmsd,
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1

        # Update progress bar
        postfix = {
            "train": f"{train_metrics['rmsd']:.2f}Å",
            "val": f"{val_rmsd:.2f}Å",
            "best": f"{best_val_rmsd:.2f}Å",
        }
        if is_best:
            postfix["*"] = "new best"
        epoch_pbar.set_postfix(postfix)

        # Write progress for external monitoring
        write_progress(progress_file, {
            "epoch": epoch,
            "total_epochs": config.training.epochs,
            "train_rmsd": train_metrics["rmsd"],
            "val_rmsd": val_rmsd,
            "best_val_rmsd": best_val_rmsd,
            "status": "training",
        })

        # Early stopping check
        if config.training.early_stopping > 0 and patience_counter >= config.training.early_stopping:
            print(f"Early stopping at epoch {epoch}")
            break

    # Final evaluation
    print()
    print("Final evaluation...")

    # Compute average epoch time
    epoch_times = [m["time"] for m in history["train"]]
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0

    # Load best model (with error handling)
    best_model_path = output_dir / "best_model.pt"
    if not best_model_path.exists():
        print(f"Warning: No best model saved (training may have failed early)")
        # Save current model as fallback
        torch.save({
            "embedding": embedding.state_dict(),
            "vae": vae.state_dict(),
            "epoch": epoch if 'epoch' in dir() else 0,
            "val_loss": best_val_loss,
            "val_rmsd": best_val_rmsd,
        }, best_model_path)

    checkpoint = torch.load(best_model_path)
    embedding.load_state_dict(checkpoint["embedding"])
    vae.load_state_dict(checkpoint["vae"])

    results = {
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["val_loss"],
        "best_val_rmsd": checkpoint.get("val_rmsd"),
        "avg_epoch_time": avg_epoch_time,
    }

    if len(test_data) > 0:
        test_metrics = evaluate(embedding, vae, test_data, config.training)
        results["test"] = test_metrics
        print(f"Test RMSD: {test_metrics['rmsd']:.2f}Å")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Avg epoch time: {avg_epoch_time:.1f}s")

        # Save reconstructions for visual inspection
        print()
        print("Saving reconstructions...")
        saved_ids = save_reconstructions(
            embedding, vae, test_data, output_dir, num_samples=num_recon_samples
        )
        if saved_ids:
            results["reconstruction_ids"] = saved_ids
            print(f"Saved {len(saved_ids)} reconstructions to: {output_dir / 'reconstructions'}")

    # Save results
    results["history"] = history
    torch.save(results, output_dir / "results.pt")

    # Write final progress
    write_progress(progress_file, {
        "epoch": results["best_epoch"],
        "total_epochs": config.training.epochs,
        "train_rmsd": results.get("test", {}).get("rmsd", best_val_rmsd),
        "val_rmsd": best_val_rmsd,
        "best_val_rmsd": best_val_rmsd,
        "test_rmsd": results.get("test", {}).get("rmsd"),
        "status": "completed",
    })

    print()
    print(f"Results saved to: {output_dir}")

    return results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run training experiment from config file"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )

    # Allow overriding config values from command line
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--kl_weight", type=float, help="KL weight")
    parser.add_argument("--kl_annealing", type=str, choices=["linear", "cyclical", "none"], help="KL annealing strategy")
    parser.add_argument("--kl_warmup_epochs", type=int, help="Epochs to anneal KL weight")
    parser.add_argument("--kl_cycle_epochs", type=int, help="Epochs per cycle for cyclical annealing")
    parser.add_argument("--free_bits", type=float, help="Minimum KL per dimension to prevent collapse")
    parser.add_argument("--distance_weight", type=float, help="Weight for pairwise distance matrix loss")
    parser.add_argument("--warmup_epochs", type=int, help="LR warmup epochs")
    parser.add_argument("--k_neighbors", type=int, help="Number of neighbors for k-NN graph")
    parser.add_argument("--attention_type", type=str, choices=["node_wise", "edge_wise"])
    parser.add_argument("--scale_type", type=str, choices=["sqrt_head_dim", "sqrt_dim", "learned", "none"])
    parser.add_argument("--skip_type", type=str, choices=["scaled", "gated", "none"])
    parser.add_argument("--rbf_type", type=str, choices=["gaussian", "bessel", "polynomial"])
    parser.add_argument("--radial_weight_rank", type=int, help="Rank for low-rank decomposition (None=full)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile (enabled by default)")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--num_structures", type=int, help="Max structures to load")
    parser.add_argument("--residue_level", action="store_true", help="Use residue centers instead of atoms")
    parser.add_argument("--device", type=str, help="Device (cuda, cpu, auto)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--num_recon_samples", type=int, default=3, help="Number of test samples to save reconstructions for")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without training (load data, build model, run one forward pass)")
    parser.add_argument("--progress_file", type=str, help="Path to write progress updates (for external monitoring)")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from: {args.config}")

    # Merge with command line overrides
    config = merge_config_with_args(config, args)

    # Run experiment
    dry_run = getattr(args, 'dry_run', False)
    progress_file = getattr(args, 'progress_file', None)
    run_experiment(config, num_recon_samples=args.num_recon_samples, dry_run=dry_run, progress_file=progress_file)


if __name__ == "__main__":
    main()
