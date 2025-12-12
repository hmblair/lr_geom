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


class StructureDataset(Dataset):
    """PyTorch Dataset for molecular structures with lazy loading.

    Structures are loaded on-demand from CIF files, not all at once.
    This provides fast startup and lower memory usage.

    This class handles all domain-specific loading logic, so training
    code only needs to work with generic (coords, features) pairs.

    Example:
        # Atom-level dataset (lazy loading)
        dataset = StructureDataset.from_directory(
            "/path/to/cifs",
            level="atom",
            min_nodes=20,
            max_nodes=500,
        )

        # Split into train/val/test
        train, val, test = dataset.split(train=0.8, val=0.1)

        # Move to GPU (structures loaded on access)
        train = train.to(device)

        # Use in training loop - structures loaded here
        for structure in train:
            features = embedding(structure.features)
            output = model(structure.coords, features)
    """

    def __init__(
        self,
        file_paths: list[Path],
        level: Literal["atom", "residue"] = "atom",
        num_feature_types: int | None = None,
        device: torch.device | None = None,
    ):
        """Initialize dataset with file paths for lazy loading.

        Args:
            file_paths: List of paths to .cif files.
            level: Granularity level ("atom" or "residue").
            num_feature_types: Number of feature types (for embedding table).
            device: Device to move structures to when loaded.
        """
        self._file_paths = file_paths
        self.level = level
        self._num_feature_types = num_feature_types
        self._device = device
        # Cache for loaded structures
        self._cache: dict[int, Structure] = {}

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
        """Create dataset from a directory of CIF files.

        Performs a quick scan to filter by size, but doesn't keep
        structures in memory. Actual loading happens on access.

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

        cif_files = sorted(path.glob("*.cif"))

        if verbose:
            print(f"Scanning structures from {path} ({level}-level)")
            print(f"Found {len(cif_files)} CIF files")

        # Quick scan to filter by size (doesn't keep structures in memory)
        valid_paths = []
        needs_filtering = min_nodes > 0 or max_nodes is not None

        if needs_filtering:
            for cif_file in cif_files:
                try:
                    # Quick load just to check size
                    polymer = ciffy.load(str(cif_file), backend="torch")
                    if level == "residue":
                        n_nodes = polymer.num_residues
                    else:
                        n_nodes = polymer.num_atoms

                    if n_nodes < min_nodes:
                        continue
                    if max_nodes is not None and n_nodes > max_nodes:
                        continue

                    valid_paths.append(cif_file)

                    if max_structures and len(valid_paths) >= max_structures:
                        break

                except Exception as e:
                    if verbose:
                        print(f"  Warning: Could not scan {cif_file.name}: {e}")
                    continue
        else:
            # No filtering needed, just use all files
            if max_structures:
                valid_paths = cif_files[:max_structures]
            else:
                valid_paths = cif_files

        if verbose:
            print(f"Found {len(valid_paths)} valid structures")

        # Get num_feature_types from ciffy
        num_feature_types = (
            ciffy.NUM_RESIDUES if level == "residue" else ciffy.NUM_ATOMS
        )

        return cls(valid_paths, level=level, num_feature_types=num_feature_types)

    @classmethod
    def from_files(
        cls,
        paths: list[str | Path],
        level: Literal["atom", "residue"] = "atom",
    ) -> StructureDataset:
        """Create dataset from specific CIF files.

        Args:
            paths: List of paths to .cif files.
            level: Granularity ("atom" or "residue").

        Returns:
            StructureDataset instance.
        """
        try:
            import ciffy
        except ImportError:
            raise ImportError(
                "ciffy is required for loading structure data. "
                "Install from: https://github.com/hmblair/ciffy"
            )

        num_feature_types = (
            ciffy.NUM_RESIDUES if level == "residue" else ciffy.NUM_ATOMS
        )

        return cls(
            [Path(p) for p in paths],
            level=level,
            num_feature_types=num_feature_types,
        )

    @classmethod
    def from_structures(
        cls,
        structures: list[Structure],
        level: Literal["atom", "residue"] = "atom",
        num_feature_types: int | None = None,
    ) -> StructureDataset:
        """Create dataset from pre-loaded structures.

        Args:
            structures: List of Structure objects.
            level: Granularity level.
            num_feature_types: Number of feature types.

        Returns:
            StructureDataset instance with pre-cached structures.
        """
        dataset = cls(
            file_paths=[],
            level=level,
            num_feature_types=num_feature_types,
        )
        # Pre-populate cache
        dataset._cache = {i: s for i, s in enumerate(structures)}
        dataset._file_paths = [None] * len(structures)  # Placeholder
        return dataset

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
            return 128 if self.level == "atom" else 32

    def __len__(self) -> int:
        return len(self._file_paths)

    def __getitem__(self, idx: int) -> Structure:
        """Load and return structure at index (with caching)."""
        if idx in self._cache:
            return self._cache[idx]

        # Load structure
        structure = _load_structure(self._file_paths[idx], self.level)

        # Move to device if specified
        if self._device is not None:
            structure = structure.to(self._device)

        # Cache for future access
        self._cache[idx] = structure
        return structure

    def __iter__(self) -> Iterator[Structure]:
        for i in range(len(self)):
            yield self[i]

    def to(self, device: torch.device) -> StructureDataset:
        """Set device for loaded structures.

        Structures already in cache are moved immediately.
        Future loads will be placed on this device.

        Args:
            device: Target device.

        Returns:
            New StructureDataset with device set.
        """
        # Create new dataset with same paths but new device
        new_dataset = StructureDataset(
            file_paths=self._file_paths,
            level=self.level,
            num_feature_types=self._num_feature_types,
            device=device,
        )
        # Move cached structures to device
        new_dataset._cache = {
            i: s.to(device) for i, s in self._cache.items()
        }
        return new_dataset

    def preload(self, verbose: bool = True) -> StructureDataset:
        """Eagerly load all structures into cache.

        Args:
            verbose: Show progress bar.

        Returns:
            Self for chaining.
        """
        from tqdm import tqdm

        iterator = range(len(self))
        if verbose:
            iterator = tqdm(iterator, desc="Preloading", ncols=80)

        for i in iterator:
            _ = self[i]  # Triggers load and cache

        return self

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

        n = len(self._file_paths)
        indices = list(range(n))
        random.Random(seed).shuffle(indices)

        n_train = int(n * train)
        n_val = int(n * val)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        def make_subset(idx_list: list[int]) -> StructureDataset:
            subset = StructureDataset(
                file_paths=[self._file_paths[i] for i in idx_list],
                level=self.level,
                num_feature_types=self._num_feature_types,
                device=self._device,
            )
            # Copy any cached structures
            for new_idx, old_idx in enumerate(idx_list):
                if old_idx in self._cache:
                    subset._cache[new_idx] = self._cache[old_idx]
            return subset

        return make_subset(train_idx), make_subset(val_idx), make_subset(test_idx)
