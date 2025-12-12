"""Data loading utilities for geometric deep learning.

This module provides a clean abstraction for loading molecular structures
into a format suitable for training geometric neural networks.

Classes:
    Structure: A single structure with coords, features, and metadata.
    StructureDataset: Dataset of structures with train/val/test splitting.

Example:
    from lr_geom.data import StructureDataset

    # Load atom-level structures
    dataset = StructureDataset.from_directory(
        "~/data/pdb130",
        level="atom",
        min_nodes=20,
        max_nodes=500,
    )

    # Load residue-level structures
    dataset = StructureDataset.from_directory(
        "~/data/pdb130",
        level="residue",
    )

    # Split and use
    train, val, test = dataset.split()
    for structure in train:
        coords, features = structure.coords, structure.features
"""
from lr_geom.data.dataset import Structure, StructureDataset

__all__ = ["Structure", "StructureDataset"]
