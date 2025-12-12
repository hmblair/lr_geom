"""Run structure VAE experiments.

Simple, straightforward training script.

Usage:
    python examples/run_experiment.py --config configs/baseline.yaml
    python examples/run_experiment.py --config configs/baseline.yaml --epochs 50
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import lr_geom as lg
from lr_geom.config import (
    ExperimentConfig,
    load_config,
    save_config,
    merge_config_with_args,
)
from lr_geom.data import StructureDataset
from lr_geom.training import set_seed, get_device


def compute_loss(
    embedding: nn.Embedding,
    vae: lg.EquivariantVAE,
    structure,
    kl_weight: float,
    distance_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Compute total loss for a single structure.

    Returns:
        (total_loss, metrics_dict)
    """
    import ciffy

    # Forward pass
    features = embedding(structure.features).unsqueeze(-1)
    recon, mu, logvar = vae(structure.coords, features)

    # Reconstruction: aligned RMSD
    coords_pred = recon[:, 0, :] * structure.coord_scale
    pred_polymer = structure.polymer.with_coordinates(coords_pred)
    rmsd = ciffy.rmsd(structure.polymer, pred_polymer)

    # KL divergence
    logvar_exp = logvar.unsqueeze(-1)
    kl_per_dim = -0.5 * (1 + logvar_exp - mu.pow(2) - logvar_exp.exp())
    kl = kl_per_dim.sum(dim=(-1, -2)).mean()

    # Total loss
    loss = rmsd + kl_weight * kl

    metrics = {"rmsd": rmsd.item(), "kl": kl.item()}

    # Optional distance matrix loss
    if distance_weight > 0:
        D_true = structure.polymer.pairwise_distance()
        D_pred = pred_polymer.pairwise_distance()
        dist_loss = (D_pred - D_true).abs().mean()
        loss = loss + distance_weight * dist_loss
        metrics["dist"] = dist_loss.item()

    metrics["loss"] = loss.item()
    return loss, metrics


def train_epoch(
    embedding: nn.Embedding,
    vae: lg.EquivariantVAE,
    dataset: StructureDataset,
    optimizer: optim.Optimizer,
    kl_weight: float,
    distance_weight: float,
    grad_clip: float,
    progress: bool = True,
) -> dict:
    """Train for one epoch."""
    import ciffy

    embedding.train()
    vae.train()

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    totals = {"loss": 0, "rmsd": 0, "kl": 0, "dist": 0}
    n = 0

    iterator = tqdm(indices, desc="Train", leave=False, ncols=80) if progress else indices
    for idx in iterator:
        structure = dataset[idx]

        # Skip non-RNA chains and empty structures
        if not structure.polymer.istype(ciffy.RNA):
            continue
        if structure.coords.shape[0] < 3:
            continue

        loss, metrics = compute_loss(
            embedding, vae, structure, kl_weight, distance_weight
        )

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(embedding.parameters()) + list(vae.parameters()),
                grad_clip,
            )
        optimizer.step()

        for k, v in metrics.items():
            totals[k] += v
        n += 1
        if progress:
            iterator.set_postfix({"rmsd": f"{totals['rmsd']/n:.2f}Å"})

    if n == 0:
        return {"loss": 0, "rmsd": 0, "kl": 0}
    return {k: v / n for k, v in totals.items() if v > 0}


@torch.no_grad()
def evaluate(
    embedding: nn.Embedding,
    vae: lg.EquivariantVAE,
    dataset: StructureDataset,
    kl_weight: float,
    distance_weight: float,
) -> dict:
    """Evaluate on dataset."""
    import ciffy

    embedding.eval()
    vae.eval()

    totals = {"loss": 0, "rmsd": 0, "kl": 0, "dist": 0}
    n = 0
    for s in dataset:
        # Skip non-RNA chains and empty structures
        if not s.polymer.istype(ciffy.RNA):
            continue
        if s.coords.shape[0] < 3:
            continue

        _, metrics = compute_loss(embedding, vae, s, kl_weight, distance_weight)
        for k, v in metrics.items():
            totals[k] += v
        n += 1

    if n == 0:
        return {"loss": 0, "rmsd": 0, "kl": 0}
    return {k: v / n for k, v in totals.items() if v > 0}


def build_model(config, num_feature_types: int, device: torch.device):
    """Build embedding and VAE from config."""
    lvals_hidden = list(range(config.model.lmax_hidden + 1))
    lvals_latent = list(range(config.model.lmax_latent + 1))

    in_repr = lg.Repr([0], mult=config.model.embed_dim)
    hidden_repr = lg.Repr(lvals_hidden, mult=config.model.hidden_mult)
    latent_repr = lg.Repr(lvals_latent, mult=config.model.latent_mult)
    out_repr = lg.Repr([1], mult=1)

    embedding = nn.Embedding(num_feature_types, config.model.embed_dim).to(device)

    vae = lg.EquivariantVAE(
        in_repr=in_repr,
        latent_repr=latent_repr,
        out_repr=out_repr,
        hidden_repr=hidden_repr,
        encoder_layers=config.model.encoder_layers,
        decoder_layers=config.model.decoder_layers,
        k_neighbors=config.model.k_neighbors,
        nheads=config.model.nheads,
        dropout=config.model.dropout,
        attn_dropout=config.model.attn_dropout,
        residual_scale=config.model.residual_scale,
        attention_type=config.model.attention_type,
        scale_type=config.model.scale_type,
        skip_type=config.model.skip_type,
        rbf_type=config.model.rbf_type,
        rbf_r_min=config.model.rbf_r_min,
        rbf_r_max=config.model.rbf_r_max,
        radial_weight_rank=config.model.radial_weight_rank,
    ).to(device)

    if config.model.use_compile:
        vae.compile()

    return embedding, vae


def get_kl_weight(epoch: int, config) -> float:
    """Get KL weight with optional annealing."""
    if config.training.kl_annealing == "none":
        return config.training.kl_weight

    warmup = config.training.kl_warmup_epochs
    if epoch < warmup:
        return config.training.kl_weight * (epoch / warmup)
    return config.training.kl_weight


def run_experiment(config: ExperimentConfig) -> dict:
    """Run a single experiment."""
    set_seed(config.seed)
    device = get_device(config.device)

    print(f"Experiment: {config.name}")
    print(f"Device: {device}")
    print(f"Compile: {config.model.use_compile}")
    print()

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / f"{config.name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    # Load data
    t0 = time.time()
    level = "residue" if config.data.residue_level else "atom"
    dataset = StructureDataset.from_directory(
        config.data.data_dir,
        scale="chain",  # iterate over chains
        level=level,
        max_atoms=config.data.max_atoms,
        num_structures=config.data.num_structures,
    )
    print(f"Data scanned in {time.time() - t0:.1f}s")

    if len(dataset) == 0:
        print("ERROR: No structures loaded!")
        sys.exit(1)

    # Split and move to device
    train_data, val_data, test_data = dataset.split(
        train=config.data.train_split,
        val=config.data.val_split,
        seed=config.data.seed,
    )
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Build model
    t0 = time.time()
    embedding, vae = build_model(config, dataset.num_feature_types, device)
    print(f"Model built in {time.time() - t0:.1f}s")

    n_params = sum(p.numel() for p in embedding.parameters()) + \
               sum(p.numel() for p in vae.parameters())
    print(f"Parameters: {n_params:,}")
    print()

    # Optimizer and scheduler
    params = list(embedding.parameters()) + list(vae.parameters())
    optimizer = optim.AdamW(params, lr=config.training.lr, weight_decay=config.training.weight_decay)

    scheduler = None
    if config.training.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.epochs, eta_min=config.training.min_lr
        )

    # Training loop
    history = {"train": [], "val": []}
    best_val_loss = float("inf")
    best_epoch = 0
    patience = 0

    print("Training...")
    for epoch in range(1, config.training.epochs + 1):
        kl_weight = get_kl_weight(epoch, config)

        # Train
        train_metrics = train_epoch(
            embedding, vae, train_data, optimizer,
            kl_weight, config.training.distance_weight, config.training.grad_clip,
        )
        history["train"].append(train_metrics)

        # Validate
        val_metrics = evaluate(
            embedding, vae, val_data, kl_weight, config.training.distance_weight
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
            }, output_dir / "best_model.pt")
        else:
            patience += 1

        # Print progress
        print(
            f"Epoch {epoch:3d}: "
            f"train={train_metrics['rmsd']:.2f}Å "
            f"val={val_metrics['rmsd']:.2f}Å "
            f"best={history['val'][best_epoch-1]['rmsd']:.2f}Å"
        )

        # Early stopping
        if config.training.early_stopping > 0 and patience >= config.training.early_stopping:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model and evaluate on test
    print()
    print("Final evaluation...")
    checkpoint = torch.load(output_dir / "best_model.pt")
    embedding.load_state_dict(checkpoint["embedding"])
    vae.load_state_dict(checkpoint["vae"])

    test_metrics = evaluate(
        embedding, vae, test_data,
        config.training.kl_weight, config.training.distance_weight
    )
    print(f"Test RMSD: {test_metrics['rmsd']:.2f}Å")

    # Save results
    results = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_rmsd": history["val"][best_epoch - 1]["rmsd"],
        "test": test_metrics,
        "history": history,
    }
    torch.save(results, output_dir / "results.pt")
    print(f"Results saved to: {output_dir}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run structure VAE experiment")
    parser.add_argument("--config", type=str, required=True, help="Config file")

    # Overrides
    parser.add_argument("--name", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--kl_weight", type=float)
    parser.add_argument("--distance_weight", type=float)
    parser.add_argument("--k_neighbors", type=int)
    parser.add_argument("--attention_type", type=str, choices=["node_wise", "edge_wise"])
    parser.add_argument("--radial_weight_rank", type=int)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--num_structures", type=int)
    parser.add_argument("--residue_level", action="store_true")
    parser.add_argument("--device", type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    config = merge_config_with_args(config, args)

    try:
        run_experiment(config)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
