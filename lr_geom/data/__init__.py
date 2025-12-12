"""Data loading utilities for geometric deep learning.

This module provides a clean abstraction for loading molecular structures
into a format suitable for training geometric neural networks.

Classes:
    Structure: A single structure with coords, polymer, and metadata.
    StructureDataset: Dataset of structures with train/val/test splitting.

Example:
    from lr_geom.data import StructureDataset
    from ciffy.nn import PolymerEmbedding

    # Load RNA structures at residue level
    dataset = StructureDataset.from_directory(
        "~/data/pdb130",
        level="residue",
        molecule_type="rna",
    )

    # Split and use
    train, val, test = dataset.split()
    embedding = PolymerEmbedding(scale=ciffy.RESIDUE, residue_dim=64)
    for structure in train:
        features = embedding(structure.polymer)
"""
from lr_geom.data.dataset import Structure, StructureDataset

__all__ = ["Structure", "StructureDataset"]
