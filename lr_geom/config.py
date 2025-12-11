"""Configuration system for lr_geom experiments.

This module provides dataclass-based configuration with YAML serialization
for reproducible experiments.

Classes:
    ModelConfig: Neural network architecture configuration
    TrainingConfig: Training hyperparameters
    ExperimentConfig: Complete experiment specification

Functions:
    load_config: Load configuration from YAML file
    save_config: Save configuration to YAML file
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Neural network architecture configuration.

    Attributes:
        embed_dim: Dimension of atom embedding.
        latent_mult: Multiplicity for latent representation.
        hidden_mult: Multiplicity for hidden layers.
        encoder_layers: Number of encoder transformer blocks.
        decoder_layers: Number of decoder transformer blocks.
        k_neighbors: Number of neighbors for k-NN graph.
        nheads: Number of attention heads.
        dropout: Dropout rate for feedforward layers.
        attn_dropout: Dropout rate for attention weights.
        residual_scale: Scaling factor for residual connections.
        attention_type: Type of attention ("node_wise" or "edge_wise").
        scale_type: Attention scaling ("sqrt_head_dim", "sqrt_dim", "learned", "none").
        skip_type: Skip connection type ("scaled", "gated", "none").
        rbf_type: Radial basis function type ("gaussian", "bessel", "polynomial").
        rbf_num_functions: Number of radial basis functions.
        rbf_r_min: Minimum radius for RBF initialization.
        rbf_r_max: Maximum radius for RBF initialization.
        lmax_hidden: Maximum angular momentum for hidden representation.
        lmax_latent: Maximum angular momentum for latent representation.
    """

    # Architecture dimensions
    embed_dim: int = 16
    latent_mult: int = 8
    hidden_mult: int = 32
    encoder_layers: int = 4
    decoder_layers: int = 4
    k_neighbors: int = 16
    nheads: int = 8

    # Regularization
    dropout: float = 0.1
    attn_dropout: float = 0.0
    residual_scale: float = 0.5

    # Attention configuration
    attention_type: str = "node_wise"
    scale_type: str = "sqrt_head_dim"

    # Skip connection configuration
    skip_type: str = "scaled"

    # RBF configuration
    rbf_type: str = "gaussian"
    rbf_num_functions: int = 16
    rbf_r_min: float = 0.0
    rbf_r_max: float = 10.0

    # Low-rank decomposition
    radial_weight_rank: int | None = None  # None = full rank

    # Compilation (disabled by default due to RepNorm dynamic slicing issues)
    use_compile: bool = False

    # Representation configuration
    lmax_hidden: int = 1
    lmax_latent: int = 1

    def __post_init__(self) -> None:
        """Validate configuration values."""
        valid_attention = {"node_wise", "edge_wise"}
        if self.attention_type not in valid_attention:
            raise ValueError(
                f"attention_type must be one of {valid_attention}, "
                f"got '{self.attention_type}'"
            )

        valid_scale = {"sqrt_head_dim", "sqrt_dim", "learned", "none"}
        if self.scale_type not in valid_scale:
            raise ValueError(
                f"scale_type must be one of {valid_scale}, "
                f"got '{self.scale_type}'"
            )

        valid_skip = {"scaled", "gated", "none"}
        if self.skip_type not in valid_skip:
            raise ValueError(
                f"skip_type must be one of {valid_skip}, "
                f"got '{self.skip_type}'"
            )

        valid_rbf = {"gaussian", "bessel", "polynomial"}
        if self.rbf_type not in valid_rbf:
            raise ValueError(
                f"rbf_type must be one of {valid_rbf}, "
                f"got '{self.rbf_type}'"
            )

        if self.radial_weight_rank is not None and self.radial_weight_rank < 1:
            raise ValueError(
                f"radial_weight_rank must be positive, got {self.radial_weight_rank}"
            )


@dataclass
class TrainingConfig:
    """Training hyperparameters.

    Attributes:
        epochs: Maximum number of training epochs.
        batch_size: Number of structures per batch.
        lr: Learning rate.
        weight_decay: L2 regularization strength.
        kl_weight: Weight for KL divergence term in VAE loss.
        warmup_epochs: Number of epochs for learning rate warmup.
        early_stopping: Patience for early stopping (0 to disable).
        grad_clip: Maximum gradient norm (0 to disable).
        scheduler: Learning rate scheduler ("cosine", "plateau", "none").
        min_lr: Minimum learning rate for schedulers.
    """

    epochs: int = 100
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 0.0
    kl_weight: float = 0.01
    warmup_epochs: int = 10
    early_stopping: int = 20
    grad_clip: float = 1.0
    scheduler: str = "cosine"
    min_lr: float = 1e-6

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.epochs < 1:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")

        valid_scheduler = {"cosine", "plateau", "none"}
        if self.scheduler not in valid_scheduler:
            raise ValueError(
                f"scheduler must be one of {valid_scheduler}, "
                f"got '{self.scheduler}'"
            )


@dataclass
class DataConfig:
    """Data loading configuration.

    Attributes:
        data_dir: Directory containing structure files.
        num_structures: Maximum number of structures to load (None for all).
        min_atoms: Minimum number of atoms per structure.
        max_atoms: Maximum number of atoms per structure.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation.
        seed: Random seed for data splitting.
    """

    data_dir: str = ""
    num_structures: int | None = None
    min_atoms: int = 20
    max_atoms: int = 700
    train_split: float = 0.8
    val_split: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.train_split + self.val_split > 1.0:
            raise ValueError(
                f"train_split + val_split must be <= 1.0, "
                f"got {self.train_split} + {self.val_split}"
            )


@dataclass
class ExperimentConfig:
    """Complete experiment specification.

    Attributes:
        name: Experiment name for logging and output directories.
        model: Model architecture configuration.
        training: Training hyperparameters.
        data: Data loading configuration.
        output_dir: Directory for saving results.
        seed: Global random seed.
        device: Device to use ("cuda", "cpu", or "auto").
    """

    name: str = "experiment"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = "outputs"
    seed: int = 42
    device: str = "auto"


def _nested_dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert nested dataclass to dictionary."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _nested_dataclass_to_dict(v) for k, v in asdict(obj).items()}
    return obj


def _dict_to_nested_dataclass(data: dict[str, Any], cls: type) -> Any:
    """Convert dictionary to nested dataclass."""
    if not hasattr(cls, "__dataclass_fields__"):
        return data

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]
            # Handle nested dataclasses
            if isinstance(value, dict) and hasattr(field_type, "__dataclass_fields__"):
                kwargs[key] = _dict_to_nested_dataclass(value, field_type)
            else:
                kwargs[key] = value

    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from YAML file.

    Args:
        path: Path to YAML configuration file.

    Returns:
        ExperimentConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    # Handle nested structure
    config_dict = {}
    config_dict["name"] = data.get("name", "experiment")
    config_dict["output_dir"] = data.get("output_dir", "outputs")
    config_dict["seed"] = data.get("seed", 42)
    config_dict["device"] = data.get("device", "auto")

    # Nested configs
    if "model" in data:
        config_dict["model"] = ModelConfig(**data["model"])
    if "training" in data:
        config_dict["training"] = TrainingConfig(**data["training"])
    if "data" in data:
        config_dict["data"] = DataConfig(**data["data"])

    return ExperimentConfig(**config_dict)


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Save experiment configuration to YAML file.

    Args:
        config: ExperimentConfig instance to save.
        path: Path for output YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = _nested_dataclass_to_dict(config)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def config_to_args(config: ExperimentConfig) -> argparse.Namespace:
    """Convert ExperimentConfig to argparse.Namespace for compatibility.

    This allows using config files with existing training scripts that
    expect command-line arguments.

    Args:
        config: ExperimentConfig instance.

    Returns:
        argparse.Namespace with flattened configuration.
    """
    args = argparse.Namespace()

    # Flatten model config
    for key, value in asdict(config.model).items():
        setattr(args, key, value)

    # Flatten training config
    for key, value in asdict(config.training).items():
        setattr(args, key, value)

    # Flatten data config
    for key, value in asdict(config.data).items():
        setattr(args, key, value)

    # Top-level config
    args.name = config.name
    args.output_dir = config.output_dir
    args.seed = config.seed
    args.device = config.device

    return args


def merge_config_with_args(
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> ExperimentConfig:
    """Merge config file with command-line arguments.

    Command-line arguments take precedence over config file values.
    Only non-None argument values override config values.

    Args:
        config: Base configuration from file.
        args: Command-line arguments.

    Returns:
        Merged ExperimentConfig.
    """
    config_dict = _nested_dataclass_to_dict(config)

    # Override with non-None args
    args_dict = vars(args)

    # Handle special case: --no-compile flag maps to use_compile=False
    if args_dict.get("no_compile"):
        config_dict["model"]["use_compile"] = False

    for key, value in args_dict.items():
        if value is not None and key != "no_compile":
            # Try to find the right nested location
            if key in asdict(config.model):
                config_dict["model"][key] = value
            elif key in asdict(config.training):
                config_dict["training"][key] = value
            elif key in asdict(config.data):
                config_dict["data"][key] = value
            elif key in config_dict:
                config_dict[key] = value

    # Reconstruct config
    return ExperimentConfig(
        name=config_dict["name"],
        model=ModelConfig(**config_dict["model"]),
        training=TrainingConfig(**config_dict["training"]),
        data=DataConfig(**config_dict["data"]),
        output_dir=config_dict["output_dir"],
        seed=config_dict["seed"],
        device=config_dict["device"],
    )
