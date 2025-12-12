"""
Train an SO(3)-equivariant VAE on molecular structures from CIF files.

Uses ciffy for loading CIF files and lr_geom for the equivariant VAE.
Atom embeddings (l=0 scalars) are used as input features.

Usage:
    python train_structure_vae.py /path/to/cif/directory

Example:
    python train_structure_vae.py ./structures --epochs 100 --batch-size 4

Features:
    - KL annealing: Gradually increase KL weight over warmup epochs
    - Learning rate warmup: Linear warmup before cosine decay
    - RMSD metric: Reports RMSD in Angstroms (more interpretable than MSE)
    - Early stopping: Stop if validation loss doesn't improve
    - Training history: Saves loss history to JSON file
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

import lr_geom as lg
from lr_geom.vae import EquivariantVAE, kl_divergence
from lr_geom.training import set_seed, get_device

# ciffy for loading CIF files
sys.path.insert(0, str(Path.home() / "academic/software/ciffy"))
import ciffy


class AtomEmbedding(nn.Module):
    """Atom embedding layer producing l=0 (scalar) features.

    Maps atom type indices to learned scalar embeddings that can be
    used as input to equivariant networks.

    Args:
        num_atoms: Vocabulary size (number of atom types).
        embed_dim: Embedding dimension (multiplicity for l=0 repr).
    """

    def __init__(self, num_atoms: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_atoms, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, atom_indices: torch.Tensor) -> torch.Tensor:
        """Embed atom indices.

        Args:
            atom_indices: (N,) tensor of atom type indices.

        Returns:
            (N, embed_dim, 1) tensor - l=0 features with shape (N, mult, repr_dim).
        """
        # (N,) -> (N, embed_dim)
        emb = self.embedding(atom_indices)
        # Add repr dimension for l=0: (N, embed_dim, 1)
        return emb.unsqueeze(-1)


class StructureVAE(nn.Module):
    """SO(3)-equivariant VAE for molecular structures.

    Encodes molecular structures using atom embeddings and coordinates,
    and decodes to reconstruct the coordinates.

    Args:
        num_atoms: Number of atom types in vocabulary.
        embed_dim: Atom embedding dimension.
        latent_mult: Multiplicity for latent representation.
        hidden_mult: Multiplicity for hidden layers.
        encoder_layers: Number of encoder transformer layers.
        decoder_layers: Number of decoder transformer layers.
        k_neighbors: Number of neighbors for k-NN attention.
        nheads: Number of attention heads.
        dropout: Dropout rate.
        residual_scale: Scale factor for residual connections. Use < 1.0 for deep nets.
    """

    def __init__(
        self,
        num_atoms: int,
        embed_dim: int = 16,
        latent_mult: int = 8,
        hidden_mult: int = 32,
        encoder_layers: int = 4,
        decoder_layers: int = 4,
        k_neighbors: int = 64,
        nheads: int = 8,
        dropout: float = 0.1,
        residual_scale: float = 0.5,
    ):
        super().__init__()

        # Atom embedding (l=0 scalars)
        self.atom_embedding = AtomEmbedding(num_atoms, embed_dim)

        # Representations
        # Input: l=0 only (scalar embeddings)
        self.in_repr = lg.Repr(lvals=[0], mult=embed_dim)
        # Latent: l=0 and l=1 for richer representation
        self.latent_repr = lg.Repr(lvals=[0, 1, 2], mult=latent_mult)
        # Output: l=1 only (3D vectors for coordinates)
        self.out_repr = lg.Repr(lvals=[1], mult=1)
        # Hidden: l=0 and l=1
        self.hidden_repr = lg.Repr(lvals=[0, 1, 2], mult=hidden_mult)

        # Equivariant VAE
        self.vae = EquivariantVAE(
            in_repr=self.in_repr,
            latent_repr=self.latent_repr,
            out_repr=self.out_repr,
            hidden_repr=self.hidden_repr,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            k_neighbors=k_neighbors,
            nheads=nheads,
            dropout=dropout,
            residual_scale=residual_scale,
        )

    def forward(
        self,
        coords: torch.Tensor,
        atom_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            coords: (N, 3) atom coordinates.
            atom_indices: (N,) atom type indices.

        Returns:
            Tuple of (reconstruction, mu, logvar).
            reconstruction: (N, 1, 4) - l=0 scalar + l=1 vector.
            mu: (N, latent_mult, latent_dim) latent mean.
            logvar: (N, latent_mult) log variance.
        """
        # Get atom embeddings: (N, embed_dim, 1)
        features = self.atom_embedding(atom_indices)

        # Forward through VAE
        return self.vae(coords, features)

    def get_coord_reconstruction(self, recon: torch.Tensor) -> torch.Tensor:
        """Extract coordinate reconstruction from output.

        Args:
            recon: (N, 1, 3) output from VAE (l=1 only).

        Returns:
            (N, 3) reconstructed coordinates.
        """
        return recon[:, 0, :]


class CIFDataset(Dataset):
    """Dataset for loading molecular structures from CIF files.

    Args:
        cif_dir: Directory containing .cif files.
        max_atoms: Maximum number of atoms (structures with more are skipped).
        limit: Maximum number of structures to use (None = use all).
    """

    def __init__(self, cif_dir: str | Path, max_atoms: int = 1000, limit: int | None = None):
        self.cif_dir = Path(cif_dir)
        self.max_atoms = max_atoms

        # Find all CIF files
        self.cif_files = sorted(self.cif_dir.glob("*.cif"))
        if not self.cif_files:
            raise ValueError(f"No .cif files found in {cif_dir}")

        # Limit number of structures if requested
        if limit is not None and limit > 0:
            self.cif_files = self.cif_files[:limit]

        print(f"Found {len(self.cif_files)} CIF files in {cif_dir}")

    def __len__(self) -> int:
        return len(self.cif_files)

    def __getitem__(self, idx: int) -> dict | None:
        """Load a structure.

        Returns:
            Dict with 'coords', 'atoms', 'id', 'coord_scale', 'polymer' or None if loading fails.
        """
        cif_file = self.cif_files[idx]

        try:
            polymer = ciffy.load(str(cif_file), backend="torch")

            # Skip if too many atoms
            if polymer.size() > self.max_atoms:
                return None

            # Skip if no atoms
            if polymer.size() == 0:
                return None

            # Center coordinates
            polymer, _ = polymer.center()

            # Normalize coordinates to unit variance for stable training
            coords = polymer.coordinates.float()
            coord_scale = coords.std().clamp(min=1e-6)
            coords_normalized = coords / coord_scale

            return {
                "coords": coords_normalized,
                "atoms": polymer.atoms.long(),
                "id": polymer.id(),
                "coord_scale": coord_scale,  # For un-normalizing outputs
                "polymer": polymer,  # Keep for ciffy.rmsd (uses original coords)
            }
        except Exception as e:
            print(f"Warning: Failed to load {cif_file}: {e}")
            return None


def collate_structures(batch: list[dict | None]) -> list[dict]:
    """Collate function that filters None values.

    Since structures have variable sizes, we don't batch them together.
    Instead, return a list of valid structures.
    """
    return [item for item in batch if item is not None]


def get_kl_weight(epoch: int, warmup_epochs: int, target_weight: float) -> float:
    """Get KL weight with linear annealing.

    Gradually increases KL weight from 0 to target over warmup epochs.
    This helps the model learn reconstruction first before enforcing
    the latent space structure.

    Args:
        epoch: Current epoch (1-indexed).
        warmup_epochs: Number of epochs to anneal over.
        target_weight: Final KL weight after warmup.

    Returns:
        Current KL weight.
    """
    if warmup_epochs <= 0:
        return target_weight
    if epoch >= warmup_epochs:
        return target_weight
    return target_weight * (epoch / warmup_epochs)


def compute_rmsd(coords_pred: torch.Tensor, coords_true: torch.Tensor) -> float:
    """Compute RMSD between predicted and true coordinates.

    Args:
        coords_pred: (N, 3) predicted coordinates.
        coords_true: (N, 3) true coordinates.

    Returns:
        RMSD in same units as input (typically Angstroms).
    """
    msd = ((coords_pred - coords_true) ** 2).sum(dim=-1).mean()
    return math.sqrt(msd.item())


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine decay.

    Args:
        optimizer: The optimizer.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total number of training epochs.
        min_lr_ratio: Minimum learning rate as fraction of initial (default 0.01).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float = 0.01,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        """Update learning rate for the next epoch."""
        self.current_epoch += 1
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = self._get_lr(base_lr)

    def _get_lr(self, base_lr: float) -> float:
        """Compute learning rate for current epoch."""
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            return base_lr * (self.current_epoch / max(1, self.warmup_epochs))
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay)

    def get_last_lr(self) -> list[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]


def train_epoch(
    model: StructureVAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kl_weight: float = 0.01,
    epoch: int = 0,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: The VAE model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        device: Device to use.
        kl_weight: Weight for KL divergence term.
        epoch: Current epoch number (for progress bar).
        grad_clip: Maximum gradient norm for clipping.

    Returns:
        Dict with average losses and metrics.
    """
    model.train()

    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_loss = 0.0
    total_rmsd = 0.0
    num_structures = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:3d}", leave=False)
    for batch in pbar:
        if not batch:
            continue

        for structure in batch:
            coords = structure["coords"].to(device)
            atoms = structure["atoms"].to(device)
            coord_scale = structure["coord_scale"].to(device)

            # Handle unknown atom types (index -1)
            atoms = atoms.clamp(min=0)

            optimizer.zero_grad()

            # Forward pass
            recon, mu, logvar = model(coords, atoms)

            # Extract coordinate reconstruction and un-normalize
            coords_recon_normalized = model.get_coord_reconstruction(recon)
            coords_recon = coords_recon_normalized * coord_scale

            # Reconstruction loss - MSE on normalized coordinates (simpler, guaranteed differentiable)
            # This works because equivariant model + centered data = same coordinate frame
            recon_loss = ((coords_recon_normalized - coords) ** 2).sum(dim=-1).mean()
            # Scale to Angstroms for interpretability
            recon_loss = recon_loss * (coord_scale ** 2)

            # KL divergence
            kl_loss = kl_divergence(mu, logvar)

            # Total loss
            loss = recon_loss + kl_weight * kl_loss

            # Backward
            loss.backward()

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            # Track metrics (RMSD in original Angstrom scale)
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_loss += loss.item()
            # Un-normalize target coords for RMSD comparison
            coords_original = coords * coord_scale
            total_rmsd += compute_rmsd(coords_recon.detach(), coords_original)
            num_structures += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{total_loss / num_structures:.4f}",
                "rmsd": f"{total_rmsd / num_structures:.2f}Å",
            })

    if num_structures == 0:
        return {"recon_loss": 0.0, "kl_loss": 0.0, "total_loss": 0.0, "rmsd": 0.0}

    return {
        "recon_loss": total_recon_loss / num_structures,
        "kl_loss": total_kl_loss / num_structures,
        "total_loss": total_loss / num_structures,
        "rmsd": total_rmsd / num_structures,
    }


@torch.no_grad()
def validate(
    model: StructureVAE,
    dataloader: DataLoader,
    device: torch.device,
    kl_weight: float = 0.01,
) -> dict[str, float]:
    """Validate the model.

    Args:
        model: The VAE model.
        dataloader: Validation data loader.
        device: Device to use.
        kl_weight: Weight for KL divergence term.

    Returns:
        Dict with average losses and metrics.
    """
    model.eval()

    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_loss = 0.0
    total_rmsd = 0.0
    num_structures = 0

    for batch in dataloader:
        if not batch:
            continue

        for structure in batch:
            coords = structure["coords"].to(device)
            atoms = structure["atoms"].to(device)
            coord_scale = structure["coord_scale"].to(device)
            atoms = atoms.clamp(min=0)

            recon, mu, logvar = model(coords, atoms)
            coords_recon_normalized = model.get_coord_reconstruction(recon)
            coords_recon = coords_recon_normalized * coord_scale

            # Reconstruction loss - MSE on normalized coordinates
            recon_loss = ((coords_recon_normalized - coords) ** 2).sum(dim=-1).mean()
            recon_loss = recon_loss * (coord_scale ** 2)
            kl_loss = kl_divergence(mu, logvar)
            loss = recon_loss + kl_weight * kl_loss

            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_loss += loss.item()
            coords_original = coords * coord_scale
            total_rmsd += compute_rmsd(coords_recon, coords_original)
            num_structures += 1

    if num_structures == 0:
        return {"recon_loss": 0.0, "kl_loss": 0.0, "total_loss": 0.0, "rmsd": 0.0}

    return {
        "recon_loss": total_recon_loss / num_structures,
        "kl_loss": total_kl_loss / num_structures,
        "total_loss": total_loss / num_structures,
        "rmsd": total_rmsd / num_structures,
    }


@torch.no_grad()
def save_example_reconstruction(
    model: StructureVAE,
    dataset: CIFDataset,
    device: torch.device,
    output_dir: Path,
    epoch: int,
    sample_idx: int = 0,
) -> None:
    """Save an example structure, its reconstruction, and a sample.

    Saves three structures:
    - Original structure (only on first epoch)
    - Reconstruction (encode -> decode)
    - Sample (random z from prior + conditioning -> decode)

    Args:
        model: The VAE model.
        dataset: Dataset to sample from.
        device: Device to use.
        output_dir: Directory to save CIF files.
        epoch: Current epoch number.
        sample_idx: Index of structure to use.
    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the structure directly using ciffy (to get full Polymer object)
    cif_file = dataset.cif_files[sample_idx % len(dataset.cif_files)]
    try:
        polymer = ciffy.load(str(cif_file), backend="torch")
        polymer, _ = polymer.center()

        coords = polymer.coordinates.float()
        coord_scale = coords.std().clamp(min=1e-6)
        coords_normalized = (coords / coord_scale).to(device)
        atoms = polymer.atoms.long().to(device).clamp(min=0)

        # Get atom embeddings (conditioning)
        features = model.atom_embedding(atoms)

        # Get reconstruction (in normalized space, then un-normalize)
        recon, mu, logvar = model(coords_normalized, atoms)
        coords_recon_normalized = model.get_coord_reconstruction(recon)
        coords_recon = coords_recon_normalized * coord_scale.to(device)

        # Get sample from prior (random z + conditioning)
        with torch.no_grad():
            sampled = model.vae.sample(coords_normalized, features)
        coords_sampled_normalized = model.get_coord_reconstruction(sampled)
        coords_sampled = coords_sampled_normalized * coord_scale.to(device)

        # Save original (only on first epoch)
        if epoch == 1:
            original_path = output_dir / f"{polymer.id()}_original.cif"
            polymer.write(str(original_path))

        # Save reconstruction
        recon_polymer = polymer.with_coordinates(coords_recon.cpu())
        recon_path = output_dir / f"{polymer.id()}_epoch{epoch:03d}_recon.cif"
        recon_polymer.write(str(recon_path))

        # Save sample
        sampled_polymer = polymer.with_coordinates(coords_sampled.cpu())
        sampled_path = output_dir / f"{polymer.id()}_epoch{epoch:03d}_sample.cif"
        sampled_polymer.write(str(sampled_path))

    except Exception as e:
        print(f"Warning: Failed to save reconstruction/sample: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Train SO(3)-equivariant VAE on molecular structures"
    )
    parser.add_argument(
        "cif_dir",
        type=str,
        help="Directory containing .cif files",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=0.01,
        help="KL divergence weight (default: 0.01)",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=16,
        help="Atom embedding dimension (default: 16)",
    )
    parser.add_argument(
        "--latent-mult",
        type=int,
        default=8,
        help="Latent space multiplicity (default: 8)",
    )
    parser.add_argument(
        "--hidden-mult",
        type=int,
        default=32,
        help="Hidden layer multiplicity (default: 32)",
    )
    parser.add_argument(
        "--encoder-layers",
        type=int,
        default=4,
        help="Number of encoder layers (default: 4)",
    )
    parser.add_argument(
        "--decoder-layers",
        type=int,
        default=4,
        help="Number of decoder layers (default: 4)",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=16,
        help="Number of k-NN neighbors (default: 16)",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help="Residual connection scale factor. Use <1.0 for deep nets (default: 1.0)",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=1000,
        help="Maximum atoms per structure (default: 1000)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split fraction (default: 0.1)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="structure_vae.pt",
        help="Path to save model (default: structure_vae.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reconstructions",
        help="Directory for saving example reconstructions (default: reconstructions)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        help="Number of warmup epochs for LR and KL annealing (default: 10)",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=20,
        help="Stop if val loss doesn't improve for N epochs. 0 to disable (default: 20)",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm. 0 to disable (default: 1.0)",
    )
    parser.add_argument(
        "--history-path",
        type=str,
        default="training_history.json",
        help="Path to save training history (default: training_history.json)",
    )
    parser.add_argument(
        "--num-structures",
        type=int,
        default=None,
        help="Limit number of structures (default: use all). Use 1 for overfitting test.",
    )

    args = parser.parse_args()

    print(f"Training SO(3)-equivariant Structure VAE")
    print(f"  CIF directory: {args.cif_dir}")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  KL weight: {args.kl_weight} (warmup over {args.warmup_epochs} epochs)")
    print(f"  Residual scale: {args.residual_scale}")
    print(f"  Warmup epochs: {args.warmup_epochs}")
    print(f"  Early stopping: {args.early_stopping if args.early_stopping > 0 else 'disabled'}")
    print(f"  Gradient clipping: {args.grad_clip if args.grad_clip > 0 else 'disabled'}")
    print(f"  Num structures: {args.num_structures if args.num_structures else 'all'}")
    print()

    device = get_device(args.device)

    # Load dataset
    dataset = CIFDataset(args.cif_dir, max_atoms=args.max_atoms, limit=args.num_structures)

    # Split into train/val
    # For single-structure overfitting, use the same structure for train and val
    if len(dataset) == 1:
        train_dataset = dataset
        val_dataset = dataset
        num_train = 1
        num_val = 1
        print("Single structure mode: using same structure for train and val")
    else:
        num_val = max(1, int(len(dataset) * args.val_split))
        num_train = len(dataset) - num_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [num_train, num_val]
        )

    print(f"Train structures: {num_train}")
    print(f"Val structures: {num_val}")
    print()

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_structures,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_structures,
        num_workers=0,
    )

    # Create model
    model = StructureVAE(
        num_atoms=ciffy.NUM_ATOMS,
        embed_dim=args.embed_dim,
        latent_mult=args.latent_mult,
        hidden_mult=args.hidden_mult,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        k_neighbors=args.k_neighbors,
        residual_scale=args.residual_scale,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate scheduler with warmup
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr_ratio=0.01,
    )

    # Training state
    best_val_loss = float("inf")
    best_val_rmsd = float("inf")
    epochs_without_improvement = 0
    history = {
        "train_loss": [],
        "train_recon_loss": [],
        "train_kl_loss": [],
        "train_rmsd": [],
        "val_loss": [],
        "val_recon_loss": [],
        "val_kl_loss": [],
        "val_rmsd": [],
        "lr": [],
        "kl_weight": [],
    }

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Get current KL weight (with annealing)
        current_kl_weight = get_kl_weight(epoch, args.warmup_epochs, args.kl_weight)

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            kl_weight=current_kl_weight,
            epoch=epoch,
            grad_clip=args.grad_clip,
        )

        # Validate
        val_metrics = validate(model, val_loader, device, kl_weight=current_kl_weight)

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Record history
        history["train_loss"].append(train_metrics["total_loss"])
        history["train_recon_loss"].append(train_metrics["recon_loss"])
        history["train_kl_loss"].append(train_metrics["kl_loss"])
        history["train_rmsd"].append(train_metrics["rmsd"])
        history["val_loss"].append(val_metrics["total_loss"])
        history["val_recon_loss"].append(val_metrics["recon_loss"])
        history["val_kl_loss"].append(val_metrics["kl_loss"])
        history["val_rmsd"].append(val_metrics["rmsd"])
        history["lr"].append(current_lr)
        history["kl_weight"].append(current_kl_weight)

        # Print progress
        print(
            f"Epoch {epoch:3d} | "
            f"Train: {train_metrics['total_loss']:.4f} (RMSD: {train_metrics['rmsd']:.2f}Å) | "
            f"Val: {val_metrics['total_loss']:.4f} (RMSD: {val_metrics['rmsd']:.2f}Å) | "
            f"LR: {current_lr:.2e} | KL_w: {current_kl_weight:.4f}"
        )

        # Check for improvement
        improved = False
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_val_rmsd = val_metrics["rmsd"]
            improved = True
            epochs_without_improvement = 0

            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_rmsd": best_val_rmsd,
                "args": vars(args),
            }, args.save_path)
            print(f"  -> Saved best model (RMSD: {best_val_rmsd:.2f}Å)")
        else:
            epochs_without_improvement += 1

        # Save training history
        with open(args.history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Save example reconstruction periodically
        if epoch % 5 == 0 or epoch == 1 or improved:
            save_example_reconstruction(
                model, dataset, device, Path(args.output_dir), epoch
            )

        # Early stopping
        if args.early_stopping > 0 and epochs_without_improvement >= args.early_stopping:
            print(f"\nEarly stopping: no improvement for {args.early_stopping} epochs")
            break

    print()
    print(f"Training complete.")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Best validation RMSD: {best_val_rmsd:.2f}Å")
    print(f"  History saved to: {args.history_path}")


if __name__ == "__main__":
    main()
