"""Structure dataset for geometric deep learning.

This module provides data loading abstractions that separate domain-specific
structure loading (atoms, residues, ciffy) from generic training code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Any

import torch
from torch.utils.data import Dataset


@dataclass
class Structure:
    """A structure ready for geometric deep learning.

    This is a domain-agnostic container that training code can use without
    knowing about atoms, residues, or ciffy internals.

    Attributes:
        coords: Normalized coordinates, shape (N, 3).
        features: Node feature indices for embedding lookup, shape (N,).
        coord_scale: Scale factor to recover original coordinates.
        polymer: Original polymer object for domain-specific operations (e.g., RMSD).
        id: Structure identifier.
    """

    coords: torch.Tensor
    features: torch.Tensor
    coord_scale: torch.Tensor
    polymer: Any  # ciffy.Polymer, but we don't require ciffy at import time
    id: str

    def to(self, device: torch.device) -> Structure:
        """Move tensors to device.

        Args:
            device: Target device.

        Returns:
            New Structure with tensors on device.
        """
        return Structure(
            coords=self.coords.to(device),
            features=self.features.to(device),
            coord_scale=self.coord_scale.to(device),
            polymer=self.polymer.to(device),
            id=self.id,
        )


class StructureDataset(Dataset):
    """PyTorch Dataset for molecular structures.

    Loads structures from CIF files with optional filtering and
    supports atom-level or residue-level representation.

    This class handles all domain-specific loading logic, so training
    code only needs to work with generic (coords, features) pairs.

    Example:
        # Atom-level dataset
        dataset = StructureDataset.from_directory(
            "/path/to/cifs",
            level="atom",
            min_nodes=20,
            max_nodes=500,
        )

        # Residue-level dataset
        dataset = StructureDataset.from_directory(
            "/path/to/cifs",
            level="residue",
        )

        # Split into train/val/test
        train, val, test = dataset.split(train=0.8, val=0.1)

        # Move to GPU
        train = train.to(device)

        # Use in training loop
        for structure in train:
            features = embedding(structure.features)
            output = model(structure.coords, features)
    """

    def __init__(
        self,
        structures: list[Structure],
        level: Literal["atom", "residue"] = "atom",
        num_feature_types: int | None = None,
    ):
        """Initialize dataset with pre-loaded structures.

        Args:
            structures: List of Structure objects.
            level: Granularity level ("atom" or "residue").
            num_feature_types: Number of feature types (for embedding table).
                If None, will be inferred from ciffy constants.
        """
        self.structures = structures
        self.level = level
        self._num_feature_types = num_feature_types

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        level: Literal["atom", "residue"] = "atom",
        min_nodes: int = 0,
        max_nodes: int | None = None,
        max_structures: int | None = None,
        verbose: bool = True,
    ) -> StructureDataset:
        """Load structures from a directory of CIF files.

        Args:
            path: Directory containing .cif files.
            level: Granularity ("atom" or "residue").
            min_nodes: Minimum nodes per structure.
            max_nodes: Maximum nodes per structure (None for no limit).
            max_structures: Maximum structures to load (None for all).
            verbose: Print loading progress.

        Returns:
            StructureDataset instance.

        Raises:
            ImportError: If ciffy is not installed.
            FileNotFoundError: If directory doesn't exist.
        """
        try:
            import ciffy
        except ImportError:
            raise ImportError(
                "ciffy is required for loading structure data. "
                "Install from: https://github.com/hmblair/ciffy"
            )

        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Data directory not found: {path}")

        structures = []
        cif_files = sorted(path.glob("*.cif"))

        if verbose:
            print(f"Loading structures from {path} ({level}-level)")
            print(f"Found {len(cif_files)} CIF files")

        for cif_file in cif_files:
            try:
                structure = cls._load_structure(cif_file, level)

                # Filter by size
                n_nodes = structure.coords.shape[0]
                if n_nodes < min_nodes:
                    continue
                if max_nodes is not None and n_nodes > max_nodes:
                    continue

                structures.append(structure)

                if max_structures and len(structures) >= max_structures:
                    break

            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not load {cif_file.name}: {e}")
                continue

        if verbose:
            print(f"Loaded {len(structures)} structures")

        # Get num_feature_types from ciffy
        num_feature_types = (
            ciffy.NUM_RESIDUES if level == "residue" else ciffy.NUM_ATOMS
        )

        return cls(structures, level=level, num_feature_types=num_feature_types)

    @classmethod
    def from_files(
        cls,
        paths: list[str | Path],
        level: Literal["atom", "residue"] = "atom",
        verbose: bool = True,
    ) -> StructureDataset:
        """Load structures from specific CIF files.

        Args:
            paths: List of paths to .cif files.
            level: Granularity ("atom" or "residue").
            verbose: Print loading progress.

        Returns:
            StructureDataset instance.

        Raises:
            ImportError: If ciffy is not installed.
        """
        try:
            import ciffy
        except ImportError:
            raise ImportError(
                "ciffy is required for loading structure data. "
                "Install from: https://github.com/hmblair/ciffy"
            )

        structures = []
        for path in paths:
            try:
                structures.append(cls._load_structure(path, level))
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not load {path}: {e}")
                continue

        num_feature_types = (
            ciffy.NUM_RESIDUES if level == "residue" else ciffy.NUM_ATOMS
        )

        return cls(structures, level=level, num_feature_types=num_feature_types)

    @staticmethod
    def _load_structure(
        path: str | Path,
        level: Literal["atom", "residue"],
    ) -> Structure:
        """Load a single structure from CIF file.

        Args:
            path: Path to .cif file.
            level: Granularity ("atom" or "residue").

        Returns:
            Structure instance.
        """
        import ciffy

        polymer = ciffy.load(str(path), backend="torch")

        # Center at appropriate level
        if level == "residue":
            polymer, _ = polymer.center(ciffy.RESIDUE)
            features = polymer.sequence
            max_idx = ciffy.NUM_RESIDUES - 1
        else:
            polymer, _ = polymer.center()
            features = polymer.atoms
            max_idx = ciffy.NUM_ATOMS - 1

        # Normalize coordinates
        coords = polymer.coordinates.float()
        coord_scale = coords.std()
        coords_normalized = coords / coord_scale

        return Structure(
            coords=coords_normalized,
            features=features.long().clamp(min=0, max=max_idx),
            coord_scale=coord_scale,
            polymer=polymer,
            id=polymer.id(),
        )

    @property
    def num_feature_types(self) -> int:
        """Number of distinct feature types (for embedding table size)."""
        if self._num_feature_types is not None:
            return self._num_feature_types

        # Fallback: infer from ciffy if available
        try:
            import ciffy

            return ciffy.NUM_RESIDUES if self.level == "residue" else ciffy.NUM_ATOMS
        except ImportError:
            # Default fallback
            return 128 if self.level == "atom" else 32

    def __len__(self) -> int:
        return len(self.structures)

    def __getitem__(self, idx: int) -> Structure:
        return self.structures[idx]

    def __iter__(self) -> Iterator[Structure]:
        return iter(self.structures)

    def to(self, device: torch.device) -> StructureDataset:
        """Move all structures to device.

        Args:
            device: Target device.

        Returns:
            New StructureDataset with structures on device.
        """
        return StructureDataset(
            [s.to(device) for s in self.structures],
            level=self.level,
            num_feature_types=self._num_feature_types,
        )

    def split(
        self,
        train: float = 0.8,
        val: float = 0.1,
        seed: int = 42,
    ) -> tuple[StructureDataset, StructureDataset, StructureDataset]:
        """Split dataset into train/val/test sets.

        Args:
            train: Fraction for training.
            val: Fraction for validation.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train, val, test) datasets.

        Raises:
            ValueError: If train + val > 1.0.
        """
        if train + val > 1.0:
            raise ValueError(f"train + val must be <= 1.0, got {train} + {val}")

        import random

        n = len(self.structures)
        indices = list(range(n))
        random.Random(seed).shuffle(indices)

        n_train = int(n * train)
        n_val = int(n * val)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        return (
            StructureDataset(
                [self.structures[i] for i in train_idx],
                self.level,
                self._num_feature_types,
            ),
            StructureDataset(
                [self.structures[i] for i in val_idx],
                self.level,
                self._num_feature_types,
            ),
            StructureDataset(
                [self.structures[i] for i in test_idx],
                self.level,
                self._num_feature_types,
            ),
        )
