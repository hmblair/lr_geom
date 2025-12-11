"""Run experiments from YAML configuration files.

This script provides a simple way to run reproducible training experiments
using configuration files.

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
from lr_geom.vae import EquivariantVAE, kl_divergence


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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

    return embedding, vae


def load_structures(
    config: DataConfig,
    device: torch.device,
) -> list[dict]:
    """Load structure data.

    Args:
        config: Data configuration.
        device: Device to place tensors on.

    Returns:
        List of structure dictionaries.
    """
    try:
        import ciffy
    except ImportError:
        raise ImportError(
            "ciffy is required for loading structure data. "
            "Install from: https://github.com/your/ciffy"
        )

    data_dir = Path(config.data_dir).expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    structures = []
    cif_files = sorted(data_dir.glob("*.cif"))

    print(f"Loading structures from {data_dir}")
    print(f"Found {len(cif_files)} CIF files")

    for cif_file in cif_files:
        try:
            polymer = ciffy.load(str(cif_file), backend="torch")
            n_atoms = polymer.size()

            # Filter by size
            if n_atoms < config.min_atoms or n_atoms > config.max_atoms:
                continue

            # Center and normalize
            polymer, _ = polymer.center()
            coords = polymer.coordinates.float()
            coord_scale = coords.std()
            coords_normalized = coords / coord_scale

            structures.append({
                "coords": coords_normalized.to(device),
                "coord_scale": coord_scale.to(device),
                "atoms": polymer.atoms.long().to(device).clamp(min=0),
                "id": polymer.id(),
                "polymer": polymer.to(device),  # Keep for ciffy.rmsd
            })

            if config.num_structures and len(structures) >= config.num_structures:
                break

        except Exception as e:
            print(f"  Warning: Could not load {cif_file.name}: {e}")
            continue

    print(f"Loaded {len(structures)} structures")
    return structures


def train_epoch(
    embedding: nn.Embedding,
    vae: EquivariantVAE,
    structures: list[dict],
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
) -> dict:
    """Train for one epoch.

    Args:
        embedding: Atom embedding module.
        vae: VAE model.
        structures: List of structure data.
        optimizer: Optimizer.
        config: Training configuration.
        device: Device.

    Returns:
        Dictionary of training metrics.
    """
    import ciffy

    embedding.train()
    vae.train()

    total_loss = 0.0
    total_rmsd = 0.0
    total_kl = 0.0
    n_batches = 0

    # Shuffle structures
    indices = list(range(len(structures)))
    random.shuffle(indices)

    for idx in indices:
        s = structures[idx]
        coords = s["coords"]
        atoms = s["atoms"]
        coord_scale = s["coord_scale"]
        polymer = s["polymer"]

        # Forward pass
        features = embedding(atoms).unsqueeze(-1)
        recon, mu, logvar = vae(coords, features)

        # Compute aligned RMSD loss using ciffy (differentiable)
        coords_pred = recon[:, 0, :]
        coords_pred_unnorm = coords_pred * coord_scale
        pred_polymer = polymer.with_coordinates(coords_pred_unnorm)
        rmsd_loss = ciffy.rmsd(polymer, pred_polymer)

        kl_loss = kl_divergence(mu, logvar).mean()
        loss = rmsd_loss + config.kl_weight * kl_loss

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

    return {
        "loss": total_loss / n_batches,
        "rmsd": total_rmsd / n_batches,
        "kl": total_kl / n_batches,
    }


def evaluate(
    embedding: nn.Embedding,
    vae: EquivariantVAE,
    structures: list[dict],
    config: TrainingConfig,
) -> dict:
    """Evaluate model on structures.

    Args:
        embedding: Atom embedding module.
        vae: VAE model.
        structures: List of structure data.
        config: Training configuration.

    Returns:
        Dictionary of evaluation metrics.
    """
    import ciffy

    embedding.eval()
    vae.eval()

    total_loss = 0.0
    total_rmsd = 0.0
    total_kl = 0.0

    with torch.no_grad():
        for s in structures:
            coords = s["coords"]
            atoms = s["atoms"]
            coord_scale = s["coord_scale"]
            polymer = s["polymer"]

            features = embedding(atoms).unsqueeze(-1)
            recon, mu, logvar = vae(coords, features)

            # Compute aligned RMSD using ciffy
            coords_pred = recon[:, 0, :]
            coords_pred_unnorm = coords_pred * coord_scale
            pred_polymer = polymer.with_coordinates(coords_pred_unnorm)
            rmsd = ciffy.rmsd(polymer, pred_polymer)

            kl_loss = kl_divergence(mu, logvar).mean()
            loss = rmsd + config.kl_weight * kl_loss

            total_loss += loss.item()
            total_rmsd += rmsd.item()
            total_kl += kl_loss.item()

    n = len(structures)
    return {
        "loss": total_loss / n,
        "rmsd": total_rmsd / n,
        "kl": total_kl / n,
    }


def run_experiment(config: ExperimentConfig) -> dict:
    """Run a single experiment.

    Args:
        config: Experiment configuration.

    Returns:
        Dictionary of results.
    """
    # Setup
    set_seed(config.seed)

    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    print(f"Device: {device}")
    print(f"Experiment: {config.name}")
    print()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / f"{config.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    save_config(config, output_dir / "config.yaml")
    print(f"Config saved to: {output_dir / 'config.yaml'}")

    # Load data
    try:
        import ciffy
        num_atom_types = ciffy.NUM_ATOMS
    except ImportError:
        num_atom_types = 128  # Default fallback

    structures = load_structures(config.data, device)

    if not structures:
        print("No structures loaded! Check data directory.")
        return {"error": "No data"}

    # Split data
    n = len(structures)
    n_train = int(n * config.data.train_split)
    n_val = int(n * config.data.val_split)

    random.seed(config.data.seed)
    indices = list(range(n))
    random.shuffle(indices)

    train_structures = [structures[i] for i in indices[:n_train]]
    val_structures = [structures[i] for i in indices[n_train:n_train + n_val]]
    test_structures = [structures[i] for i in indices[n_train + n_val:]]

    print(f"Train: {len(train_structures)}, Val: {len(val_structures)}, Test: {len(test_structures)}")
    print()

    # Build model
    embedding, vae = build_model_from_config(config.model, num_atom_types, device)

    num_params = sum(p.numel() for p in embedding.parameters()) + sum(p.numel() for p in vae.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Attention type: {config.model.attention_type}")
    print(f"Scale type: {config.model.scale_type}")
    print(f"Skip type: {config.model.skip_type}")
    print(f"RBF type: {config.model.rbf_type}")
    print()

    # Setup optimizer
    params = list(embedding.parameters()) + list(vae.parameters())
    optimizer = optim.AdamW(
        params,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )

    # Learning rate scheduler
    if config.training.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
            eta_min=config.training.min_lr,
        )
    elif config.training.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=config.training.min_lr,
        )
    else:
        scheduler = None

    # Training loop
    history = {"train": [], "val": []}
    best_val_loss = float("inf")
    best_val_rmsd = float("inf")
    patience_counter = 0

    print("Training...")
    for epoch in range(1, config.training.epochs + 1):
        # Train
        train_metrics = train_epoch(
            embedding, vae, train_structures, optimizer, config.training, device
        )
        history["train"].append(train_metrics)

        # Validate
        if val_structures:
            val_metrics = evaluate(embedding, vae, val_structures, config.training)
            history["val"].append(val_metrics)
            val_loss = val_metrics["loss"]
            val_rmsd = val_metrics["rmsd"]
        else:
            val_loss = train_metrics["loss"]
            val_rmsd = train_metrics["rmsd"]

        # Learning rate scheduling
        if scheduler is not None:
            if config.training.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_rmsd = val_rmsd
            patience_counter = 0
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

        # Log progress
        lr = optimizer.param_groups[0]["lr"]
        log_str = (
            f"Epoch {epoch:3d} | "
            f"Train RMSD: {train_metrics['rmsd']:.2f}Å | "
        )
        if val_structures:
            log_str += f"Val RMSD: {val_metrics['rmsd']:.2f}Å | "
        log_str += f"LR: {lr:.2e}"
        print(log_str)

        # Early stopping check
        if config.training.early_stopping > 0 and patience_counter >= config.training.early_stopping:
            print(f"Early stopping at epoch {epoch}")
            break

    # Final evaluation
    print()
    print("Final evaluation...")

    # Load best model
    checkpoint = torch.load(output_dir / "best_model.pt")
    embedding.load_state_dict(checkpoint["embedding"])
    vae.load_state_dict(checkpoint["vae"])

    results = {
        "best_epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["val_loss"],
        "best_val_rmsd": checkpoint.get("val_rmsd"),
    }

    if test_structures:
        test_metrics = evaluate(embedding, vae, test_structures, config.training)
        results["test"] = test_metrics
        print(f"Test RMSD: {test_metrics['rmsd']:.2f}Å")
        print(f"Test Loss: {test_metrics['loss']:.4f}")

    # Save results
    results["history"] = history
    torch.save(results, output_dir / "results.pt")

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
    parser.add_argument("--k_neighbors", type=int, help="Number of neighbors for k-NN graph")
    parser.add_argument("--attention_type", type=str, choices=["node_wise", "edge_wise"])
    parser.add_argument("--scale_type", type=str, choices=["sqrt_head_dim", "sqrt_dim", "learned", "none"])
    parser.add_argument("--skip_type", type=str, choices=["scaled", "gated", "none"])
    parser.add_argument("--rbf_type", type=str, choices=["gaussian", "bessel", "polynomial"])
    parser.add_argument("--radial_weight_rank", type=int, help="Rank for low-rank decomposition (None=full)")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--num_structures", type=int, help="Max structures to load")
    parser.add_argument("--device", type=str, help="Device (cuda, cpu, auto)")
    parser.add_argument("--seed", type=int, help="Random seed")

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
    run_experiment(config)


if __name__ == "__main__":
    main()
