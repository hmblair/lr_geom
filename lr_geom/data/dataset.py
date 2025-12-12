"""Structure dataset using ciffy's PolymerDataset.

Wraps ciffy.nn.PolymerDataset to provide normalized coordinates
and feature indices for geometric deep learning.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Any
import random

import torch
from torch.utils.data import Dataset


@dataclass
class Structure:
    """A molecular structure ready for geometric deep learning.

    Attributes:
        coords: Normalized coordinates, shape (N, 3).
        features: Feature indices for embedding lookup, shape (N,).
        coord_scale: Scale factor to recover original coordinates.
        polymer: Original ciffy Polymer object.
        id: Structure identifier.
    """
    coords: torch.Tensor
    features: torch.Tensor
    coord_scale: torch.Tensor
    polymer: Any
    id: str

    def to(self, device: torch.device) -> Structure:
        """Move tensors to device."""
        return Structure(
            coords=self.coords.to(device),
            features=self.features.to(device),
            coord_scale=self.coord_scale.to(device),
            polymer=self.polymer.to(device),
            id=self.id,
        )


class StructureDataset(Dataset):
    """Dataset for molecular structures using ciffy.

    Uses ciffy.nn.PolymerDataset for efficient loading with optional
    chain-level iteration.

    Example:
        dataset = StructureDataset.from_directory(
            "/path/to/cifs",
            scale="chain",  # iterate over chains
            level="residue",  # residue-level features
            max_atoms=1000,
        )
        train, val, test = dataset.split()
        for structure in train:
            coords, features = structure.coords, structure.features
    """

    def __init__(
        self,
        ciffy_dataset,
        level: Literal["atom", "residue"] = "atom",
        device: torch.device | None = None,
    ):
        """Initialize with a ciffy PolymerDataset.

        Args:
            ciffy_dataset: A ciffy.nn.PolymerDataset instance.
            level: Feature level ("atom" or "residue").
            device: Device to place tensors on.
        """
        self._dataset = ciffy_dataset
        self.level = level
        self._device = device

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        scale: Literal["molecule", "chain"] = "chain",
        level: Literal["atom", "residue"] = "atom",
        max_atoms: int | None = None,
        num_structures: int | None = None,
        device: torch.device | None = None,
    ) -> StructureDataset:
        """Create dataset from directory of CIF files.

        Args:
            path: Directory containing .cif files.
            scale: Iteration scale:
                - "molecule": iterate over full structures
                - "chain": iterate over individual chains (recommended)
            level: Feature level:
                - "atom": use atom type indices
                - "residue": use residue type indices
            max_atoms: Maximum atoms per item (filtered out if exceeded).
            num_structures: Maximum number of structures to load (None for all).
            device: Device for tensors.

        Returns:
            StructureDataset instance.
        """
        import ciffy
        from ciffy.nn import PolymerDataset

        scale_enum = ciffy.CHAIN if scale == "chain" else ciffy.MOLECULE
        path = Path(path)

        # If num_structures is set, create temp dir with symlinks to limit scan time
        scan_path = path
        temp_dir = None
        if num_structures is not None:
            cif_files = sorted(path.glob("*.cif"))[:num_structures]
            if cif_files:
                temp_dir = tempfile.mkdtemp(prefix="lr_geom_")
                scan_path = Path(temp_dir)
                for f in cif_files:
                    (scan_path / f.name).symlink_to(f)
                print(f"Scanning {len(cif_files)} files (limited from {path})")
            else:
                print(f"No .cif files found in {path}")
        else:
            print(f"Scanning {path} (scale={scale}, level={level}, max_atoms={max_atoms})")

        ciffy_dataset = PolymerDataset(
            directory=scan_path,
            scale=scale_enum,
            max_atoms=max_atoms,
            backend="torch",
        )
        print(f"Found {len(ciffy_dataset)} items")

        # Clean up temp dir (symlinks only, originals safe)
        if temp_dir is not None:
            import shutil
            shutil.rmtree(temp_dir)

        return cls(ciffy_dataset, level=level, device=device)

    @property
    def num_feature_types(self) -> int:
        """Number of feature types for embedding table."""
        import ciffy
        return ciffy.NUM_RESIDUES if self.level == "residue" else ciffy.NUM_ATOMS

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Structure:
        """Get structure at index."""
        import ciffy

        polymer = self._dataset[idx]

        # Center coordinates at appropriate level
        if self.level == "residue":
            polymer, _ = polymer.center(ciffy.RESIDUE)
            features = polymer.sequence
            max_idx = ciffy.NUM_RESIDUES - 1
        else:
            polymer, _ = polymer.center()
            features = polymer.atoms
            max_idx = ciffy.NUM_ATOMS - 1

        # Normalize coordinates
        coords = polymer.coordinates.float()
        if coords.shape[0] >= 2:
            coord_scale = coords.std()
            if coord_scale > 0:
                coords = coords / coord_scale
            else:
                coord_scale = torch.tensor(1.0)
        else:
            coord_scale = torch.tensor(1.0)

        structure = Structure(
            coords=coords,
            features=features.long().clamp(0, max_idx),
            coord_scale=coord_scale,
            polymer=polymer,
            id=polymer.id(),
        )

        if self._device is not None:
            structure = structure.to(self._device)

        return structure

    def __iter__(self) -> Iterator[Structure]:
        for i in range(len(self)):
            yield self[i]

    def to(self, device: torch.device) -> StructureDataset:
        """Set device for loaded structures."""
        return StructureDataset(self._dataset, self.level, device)

    def split(
        self,
        train: float = 0.8,
        val: float = 0.1,
        seed: int = 42,
    ) -> tuple[StructureDataset, StructureDataset, StructureDataset]:
        """Split into train/val/test sets.

        Note: This creates subset views that share the underlying ciffy dataset.
        """
        n = len(self._dataset)
        indices = list(range(n))
        random.Random(seed).shuffle(indices)

        n_train = int(n * train)
        n_val = int(n * val)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        return (
            _SubsetDataset(self, train_idx),
            _SubsetDataset(self, val_idx),
            _SubsetDataset(self, test_idx),
        )


class _SubsetDataset(Dataset):
    """Subset view of a StructureDataset."""

    def __init__(self, parent: StructureDataset, indices: list[int]):
        self._parent = parent
        self._indices = indices
        self.level = parent.level
        self._device = parent._device

    @property
    def num_feature_types(self) -> int:
        return self._parent.num_feature_types

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Structure:
        return self._parent[self._indices[idx]]

    def __iter__(self) -> Iterator[Structure]:
        for i in range(len(self)):
            yield self[i]

    def to(self, device: torch.device) -> _SubsetDataset:
        subset = _SubsetDataset(self._parent.to(device), self._indices)
        return subset

    def split(
        self,
        train: float = 0.8,
        val: float = 0.1,
        seed: int = 42,
    ) -> tuple[_SubsetDataset, _SubsetDataset, _SubsetDataset]:
        """Split into train/val/test sets."""
        n = len(self._indices)
        local_indices = list(range(n))
        random.Random(seed).shuffle(local_indices)

        n_train = int(n * train)
        n_val = int(n * val)

        train_local = local_indices[:n_train]
        val_local = local_indices[n_train:n_train + n_val]
        test_local = local_indices[n_train + n_val:]

        # Map back to parent indices
        train_idx = [self._indices[i] for i in train_local]
        val_idx = [self._indices[i] for i in val_local]
        test_idx = [self._indices[i] for i in test_local]

        return (
            _SubsetDataset(self._parent, train_idx),
            _SubsetDataset(self._parent, val_idx),
            _SubsetDataset(self._parent, test_idx),
        )
