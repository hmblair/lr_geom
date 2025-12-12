"""Callback system for training.

Callbacks provide hooks into the training loop for customization.
"""
from __future__ import annotations

from lr_geom.training.callbacks.base import Callback, CallbackList
from lr_geom.training.callbacks.checkpointing import ModelCheckpoint
from lr_geom.training.callbacks.early_stopping import EarlyStopping
from lr_geom.training.callbacks.progress import ProgressBar

__all__ = [
    "Callback",
    "CallbackList",
    "ModelCheckpoint",
    "EarlyStopping",
    "ProgressBar",
]
