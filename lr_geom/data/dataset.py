"""Structure dataset for geometric deep learning.

Simple, straightforward data loading. No lazy loading complexity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Any

import torch
from torch.utils.data import Dataset


@dataclass
class Structure:
    """A molecular structure for geometric deep learning.

    Attributes:
        coords: Normalized coordinates, shape (N, 3).
        features: Node feature indices for embedding lookup, shape (N,).
        coord_scale: Scale factor to recover original coordinates.
        polymer: Original ciffy polymer object.
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
    """Simple dataset for molecular structures.

    Example:
        dataset = StructureDataset.from_directory("/path/to/cifs")
        train, val, test = dataset.split()
        train = train.to(device)
        for structure in train:
            ...
    """

    def __init__(
        self,
        structures: list[Structure],
        level: Literal["atom", "residue"] = "atom",
    ):
        self.structures = structures
        self.level = level
        self._device = None

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        level: Literal["atom", "residue"] = "atom",
        max_structures: int | None = None,
        verbose: bool = True,
    ) -> StructureDataset:
        """Load structures from a directory of CIF files.

        Args:
            path: Directory containing .cif files.
            level: "atom" or "residue" level representation.
            max_structures: Maximum number to load (None for all).
            verbose: Print progress.

        Returns:
            StructureDataset with loaded structures.
        """
        try:
            import ciffy
        except ImportError:
            raise ImportError(
                "ciffy is required. Install from: https://github.com/hmblair/ciffy"
            )

        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        cif_files = sorted(path.glob("*.cif"))
        if max_structures:
            cif_files = cif_files[:max_structures]

        if verbose:
            print(f"Loading {len(cif_files)} structures from {path} ({level}-level)")

        structures = []
        for cif_file in cif_files:
            try:
                s = cls._load_one(cif_file, level)
                structures.append(s)
            except Exception as e:
                if verbose:
                    print(f"  Skipping {cif_file.name}: {e}")

        if verbose:
            print(f"Loaded {len(structures)} structures")

        return cls(structures, level=level)

    @staticmethod
    def _load_one(path: Path, level: str) -> Structure:
        """Load a single structure from CIF file."""
        import ciffy

        polymer = ciffy.load(str(path), backend="torch")

        if level == "residue":
            polymer, _ = polymer.center(ciffy.RESIDUE)
            features = polymer.sequence
            max_idx = ciffy.NUM_RESIDUES - 1
        else:
            polymer, _ = polymer.center()
            features = polymer.atoms
            max_idx = ciffy.NUM_ATOMS - 1

        coords = polymer.coordinates.float()
        coord_scale = coords.std()
        coords_norm = coords / coord_scale

        return Structure(
            coords=coords_norm,
            features=features.long().clamp(0, max_idx),
            coord_scale=coord_scale,
            polymer=polymer,
            id=polymer.id(),
        )

    @property
    def num_feature_types(self) -> int:
        """Number of feature types for embedding table."""
        try:
            import ciffy
            return ciffy.NUM_RESIDUES if self.level == "residue" else ciffy.NUM_ATOMS
        except ImportError:
            return 32 if self.level == "residue" else 128

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(self, idx: int) -> Structure:
        return self.structures[idx]

    def __iter__(self) -> Iterator[Structure]:
        return iter(self.structures)

    def to(self, device: torch.device) -> StructureDataset:
        """Move all structures to device."""
        new_dataset = StructureDataset(
            [s.to(device) for s in self.structures],
            level=self.level,
        )
        new_dataset._device = device
        return new_dataset

    def split(
        self,
        train: float = 0.8,
        val: float = 0.1,
        seed: int = 42,
    ) -> tuple[StructureDataset, StructureDataset, StructureDataset]:
        """Split into train/val/test sets."""
        import random

        n = len(self.structures)
        indices = list(range(n))
        random.Random(seed).shuffle(indices)

        n_train = int(n * train)
        n_val = int(n * val)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        return (
            StructureDataset([self.structures[i] for i in train_idx], self.level),
            StructureDataset([self.structures[i] for i in val_idx], self.level),
            StructureDataset([self.structures[i] for i in test_idx], self.level),
        )
