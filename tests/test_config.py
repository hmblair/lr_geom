"""Tests for configuration system."""

import tempfile
from pathlib import Path

import pytest
import torch

from lr_geom.config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    ExperimentConfig,
    load_config,
    save_config,
    config_to_args,
    merge_config_with_args,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ModelConfig()
        assert config.embed_dim == 16
        assert config.attention_type == "node_wise"
        assert config.scale_type == "sqrt_head_dim"
        assert config.skip_type == "scaled"
        assert config.rbf_type == "gaussian"

    def test_custom_values(self):
        """Test setting custom values."""
        config = ModelConfig(
            embed_dim=32,
            attention_type="edge_wise",
            scale_type="learned",
            skip_type="gated",
            rbf_type="bessel",
        )
        assert config.embed_dim == 32
        assert config.attention_type == "edge_wise"
        assert config.scale_type == "learned"
        assert config.skip_type == "gated"
        assert config.rbf_type == "bessel"

    def test_invalid_attention_type(self):
        """Test that invalid attention_type raises error."""
        with pytest.raises(ValueError, match="attention_type"):
            ModelConfig(attention_type="invalid")

    def test_invalid_scale_type(self):
        """Test that invalid scale_type raises error."""
        with pytest.raises(ValueError, match="scale_type"):
            ModelConfig(scale_type="invalid")

    def test_invalid_skip_type(self):
        """Test that invalid skip_type raises error."""
        with pytest.raises(ValueError, match="skip_type"):
            ModelConfig(skip_type="invalid")

    def test_invalid_rbf_type(self):
        """Test that invalid rbf_type raises error."""
        with pytest.raises(ValueError, match="rbf_type"):
            ModelConfig(rbf_type="invalid")


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = TrainingConfig()
        assert config.epochs == 100
        assert config.lr == 1e-3
        assert config.scheduler == "cosine"

    def test_invalid_epochs(self):
        """Test that invalid epochs raises error."""
        with pytest.raises(ValueError, match="epochs"):
            TrainingConfig(epochs=0)

    def test_invalid_batch_size(self):
        """Test that invalid batch_size raises error."""
        with pytest.raises(ValueError, match="batch_size"):
            TrainingConfig(batch_size=0)

    def test_invalid_lr(self):
        """Test that invalid lr raises error."""
        with pytest.raises(ValueError, match="lr"):
            TrainingConfig(lr=-0.1)

    def test_invalid_scheduler(self):
        """Test that invalid scheduler raises error."""
        with pytest.raises(ValueError, match="scheduler"):
            TrainingConfig(scheduler="invalid")


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = DataConfig()
        assert config.train_split == 0.8
        assert config.val_split == 0.1

    def test_invalid_split_sum(self):
        """Test that split > 1.0 raises error."""
        with pytest.raises(ValueError, match="train_split.*val_split"):
            DataConfig(train_split=0.8, val_split=0.3)


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ExperimentConfig()
        assert config.name == "experiment"
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.data, DataConfig)

    def test_nested_config(self):
        """Test nested config access."""
        config = ExperimentConfig(
            model=ModelConfig(attention_type="edge_wise"),
            training=TrainingConfig(epochs=50),
        )
        assert config.model.attention_type == "edge_wise"
        assert config.training.epochs == 50


class TestConfigIO:
    """Tests for config load/save functions."""

    def test_save_and_load(self):
        """Test saving and loading config."""
        config = ExperimentConfig(
            name="test_exp",
            model=ModelConfig(embed_dim=64, attention_type="edge_wise"),
            training=TrainingConfig(epochs=50, lr=0.0001),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            save_config(config, path)
            loaded = load_config(path)

        assert loaded.name == "test_exp"
        assert loaded.model.embed_dim == 64
        assert loaded.model.attention_type == "edge_wise"
        assert loaded.training.epochs == 50
        assert loaded.training.lr == 0.0001

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_partial_config(self):
        """Test loading config with only some fields specified."""
        yaml_content = """
name: partial_exp
model:
  embed_dim: 32
training:
  epochs: 25
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            with open(path, "w") as f:
                f.write(yaml_content)

            loaded = load_config(path)

        assert loaded.name == "partial_exp"
        assert loaded.model.embed_dim == 32
        # Defaults should be used for unspecified fields
        assert loaded.model.attention_type == "node_wise"
        assert loaded.training.epochs == 25
        assert loaded.training.lr == 1e-3  # default


class TestConfigToArgs:
    """Tests for config_to_args function."""

    def test_flattens_config(self):
        """Test that config is properly flattened to args."""
        config = ExperimentConfig(
            name="test",
            model=ModelConfig(embed_dim=32, attention_type="edge_wise"),
            training=TrainingConfig(epochs=50),
        )

        args = config_to_args(config)

        assert args.name == "test"
        assert args.embed_dim == 32
        assert args.attention_type == "edge_wise"
        assert args.epochs == 50


class TestMergeConfigWithArgs:
    """Tests for merge_config_with_args function."""

    def test_override_with_args(self):
        """Test that args override config values."""
        import argparse

        config = ExperimentConfig(
            model=ModelConfig(embed_dim=16),
            training=TrainingConfig(epochs=100),
        )

        args = argparse.Namespace(embed_dim=32, epochs=50, lr=None, name=None)
        merged = merge_config_with_args(config, args)

        assert merged.model.embed_dim == 32
        assert merged.training.epochs == 50
        # Unchanged values
        assert merged.training.lr == 1e-3


class TestConfigWithLayers:
    """Test that config values work correctly with layer construction."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_attention_types(self, device):
        """Test that both attention types can be constructed."""
        import lr_geom as lg

        for attention_type in ["node_wise", "edge_wise"]:
            repr_in = lg.Repr([0, 1], mult=4)
            repr_out = lg.Repr([0, 1], mult=4)
            hidden_repr = lg.Repr([0, 1], mult=8)

            model = lg.EquivariantTransformer(
                in_repr=repr_in,
                out_repr=repr_out,
                hidden_repr=hidden_repr,
                hidden_layers=2,
                edge_dim=8,
                edge_hidden_dim=16,
                k_neighbors=4,
                nheads=2,
                attention_type=attention_type,
            )

            # Test forward pass
            N = 10
            coords = torch.randn(N, 3)
            features = torch.randn(N, repr_in.mult, repr_in.dim())
            output = model(coords, features)
            assert output.shape == (N, repr_out.mult, repr_out.dim())

    def test_scale_types(self, device):
        """Test that all scale types can be constructed."""
        import lr_geom as lg

        for scale_type in ["sqrt_head_dim", "sqrt_dim", "learned", "none"]:
            repr_in = lg.Repr([0, 1], mult=4)
            repr_out = lg.Repr([0, 1], mult=4)
            hidden_repr = lg.Repr([0, 1], mult=8)

            model = lg.EquivariantTransformer(
                in_repr=repr_in,
                out_repr=repr_out,
                hidden_repr=hidden_repr,
                hidden_layers=1,
                edge_dim=8,
                edge_hidden_dim=16,
                k_neighbors=4,
                nheads=2,
                scale_type=scale_type,
            )

            N = 10
            coords = torch.randn(N, 3)
            features = torch.randn(N, repr_in.mult, repr_in.dim())
            output = model(coords, features)
            assert not torch.isnan(output).any()

    def test_skip_types(self, device):
        """Test that all skip types can be constructed."""
        import lr_geom as lg

        for skip_type in ["scaled", "gated", "none"]:
            repr_in = lg.Repr([0, 1], mult=4)
            repr_out = lg.Repr([0, 1], mult=4)
            hidden_repr = lg.Repr([0, 1], mult=8)

            model = lg.EquivariantTransformer(
                in_repr=repr_in,
                out_repr=repr_out,
                hidden_repr=hidden_repr,
                hidden_layers=2,
                edge_dim=8,
                edge_hidden_dim=16,
                k_neighbors=4,
                nheads=2,
                skip_type=skip_type,
            )

            N = 10
            coords = torch.randn(N, 3)
            features = torch.randn(N, repr_in.mult, repr_in.dim())
            output = model(coords, features)
            assert not torch.isnan(output).any()

    def test_rbf_types(self, device):
        """Test that all RBF types can be constructed."""
        import lr_geom as lg

        for rbf_type in ["gaussian", "bessel", "polynomial"]:
            repr_in = lg.Repr([0, 1], mult=4)
            repr_out = lg.Repr([0, 1], mult=4)
            hidden_repr = lg.Repr([0, 1], mult=8)

            model = lg.EquivariantTransformer(
                in_repr=repr_in,
                out_repr=repr_out,
                hidden_repr=hidden_repr,
                hidden_layers=1,
                edge_dim=8,
                edge_hidden_dim=16,
                k_neighbors=4,
                nheads=2,
                rbf_type=rbf_type,
            )

            N = 10
            coords = torch.randn(N, 3)
            features = torch.randn(N, repr_in.mult, repr_in.dim())
            output = model(coords, features)
            assert not torch.isnan(output).any()


class TestRBFFunctions:
    """Test different RBF implementations."""

    def test_gaussian_rbf(self):
        """Test Gaussian RBF produces expected output shape and values."""
        from lr_geom.equivariant import RadialBasisFunctions

        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="gaussian")
        distances = torch.rand(100) * 10
        output = rbf(distances)

        assert output.shape == (100, 16)
        assert not torch.isnan(output).any()
        assert (output >= 0).all()  # Gaussian RBF is always positive

    def test_bessel_rbf(self):
        """Test Bessel RBF produces expected output shape and values."""
        from lr_geom.equivariant import RadialBasisFunctions

        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="bessel")
        distances = torch.rand(100) * 10
        output = rbf(distances)

        assert output.shape == (100, 16)
        assert not torch.isnan(output).any()

    def test_polynomial_rbf(self):
        """Test polynomial RBF produces expected output shape and values."""
        from lr_geom.equivariant import RadialBasisFunctions

        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="polynomial")
        distances = torch.rand(100) * 10
        output = rbf(distances)

        assert output.shape == (100, 16)
        assert not torch.isnan(output).any()
        assert (output >= 0).all()  # Polynomial envelope is always positive

    def test_bessel_cutoff(self):
        """Test that Bessel RBF goes to zero at cutoff."""
        from lr_geom.equivariant import RadialBasisFunctions

        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="bessel")
        # At exactly r_max, envelope is 0
        at_cutoff = rbf(torch.tensor([10.0]))
        assert torch.allclose(at_cutoff, torch.zeros_like(at_cutoff), atol=1e-6)

    def test_polynomial_cutoff(self):
        """Test that polynomial RBF goes to zero at cutoff."""
        from lr_geom.equivariant import RadialBasisFunctions

        rbf = RadialBasisFunctions(16, r_min=0.0, r_max=10.0, rbf_type="polynomial")
        at_cutoff = rbf(torch.tensor([10.0]))
        assert torch.allclose(at_cutoff, torch.zeros_like(at_cutoff), atol=1e-6)

    def test_invalid_rbf_type(self):
        """Test that invalid RBF type raises error."""
        from lr_geom.equivariant import RadialBasisFunctions

        with pytest.raises(ValueError, match="rbf_type"):
            RadialBasisFunctions(16, rbf_type="invalid")


class TestEdgeWiseAttention:
    """Tests specifically for edge-wise attention implementation."""

    def test_edge_wise_forward(self):
        """Test edge-wise attention forward pass."""
        import lr_geom as lg

        repr_in = lg.Repr([0, 1], mult=4)
        repr_out = lg.Repr([0, 1], mult=4)
        hidden_repr = lg.Repr([0, 1], mult=8)

        model = lg.EquivariantTransformer(
            in_repr=repr_in,
            out_repr=repr_out,
            hidden_repr=hidden_repr,
            hidden_layers=2,
            edge_dim=8,
            edge_hidden_dim=16,
            k_neighbors=4,
            nheads=2,
            attention_type="edge_wise",
        )

        N = 20
        coords = torch.randn(N, 3)
        features = torch.randn(N, repr_in.mult, repr_in.dim())
        output = model(coords, features)

        assert output.shape == (N, repr_out.mult, repr_out.dim())
        assert not torch.isnan(output).any()

    def test_edge_wise_backward(self):
        """Test edge-wise attention backward pass."""
        import lr_geom as lg

        repr_in = lg.Repr([0, 1], mult=4)
        repr_out = lg.Repr([0, 1], mult=4)
        hidden_repr = lg.Repr([0, 1], mult=8)

        model = lg.EquivariantTransformer(
            in_repr=repr_in,
            out_repr=repr_out,
            hidden_repr=hidden_repr,
            hidden_layers=2,
            edge_dim=8,
            edge_hidden_dim=16,
            k_neighbors=4,
            nheads=2,
            attention_type="edge_wise",
        )

        N = 15
        coords = torch.randn(N, 3)
        features = torch.randn(N, repr_in.mult, repr_in.dim())

        output = model(coords, features)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestGatedSkip:
    """Tests specifically for gated skip connections."""

    def test_gated_skip_initialization(self):
        """Test that gated skip gate is initialized to zero."""
        import lr_geom as lg
        from lr_geom.layers import EquivariantTransformerBlock
        from lr_geom.representations import ProductRepr
        from copy import deepcopy

        repr_in = lg.Repr([0, 1], mult=4)
        prepr = ProductRepr(deepcopy(repr_in), deepcopy(repr_in))

        block = EquivariantTransformerBlock(
            prepr,
            edge_dim=8,
            edge_hidden_dim=16,
            nheads=2,
            skip_type="gated",
        )

        # Gate should be initialized to 0
        assert hasattr(block, "gate")
        assert torch.allclose(block.gate, torch.zeros(1))

    def test_gated_skip_learnable(self):
        """Test that gated skip gate is learnable."""
        import lr_geom as lg

        repr_in = lg.Repr([0, 1], mult=4)
        repr_out = lg.Repr([0, 1], mult=4)
        hidden_repr = lg.Repr([0, 1], mult=8)

        model = lg.EquivariantTransformer(
            in_repr=repr_in,
            out_repr=repr_out,
            hidden_repr=hidden_repr,
            hidden_layers=2,
            edge_dim=8,
            edge_hidden_dim=16,
            k_neighbors=4,
            nheads=2,
            skip_type="gated",
        )

        # Check that gate parameters exist
        gate_params = [name for name, _ in model.named_parameters() if "gate" in name]
        assert len(gate_params) > 0


class TestLearnedScale:
    """Tests for learned attention scaling."""

    def test_learned_scale_initialization(self):
        """Test that learned scale is initialized to sqrt(head_dim)^-0.5."""
        import lr_geom as lg
        from lr_geom.layers import EquivariantAttention
        from lr_geom.representations import ProductRepr
        from copy import deepcopy

        repr_in = lg.Repr([0, 1], mult=4)
        prepr = ProductRepr(deepcopy(repr_in), deepcopy(repr_in))

        attn = EquivariantAttention(
            prepr,
            edge_dim=8,
            edge_hidden_dim=16,
            nheads=2,
            scale_type="learned",
        )

        expected_scale = attn.head_dim ** -0.5
        assert torch.allclose(attn.scale, torch.tensor(expected_scale))

    def test_learned_scale_is_parameter(self):
        """Test that learned scale is a learnable parameter."""
        import lr_geom as lg
        from lr_geom.layers import EquivariantAttention
        from lr_geom.representations import ProductRepr
        from copy import deepcopy

        repr_in = lg.Repr([0, 1], mult=4)
        prepr = ProductRepr(deepcopy(repr_in), deepcopy(repr_in))

        attn = EquivariantAttention(
            prepr,
            edge_dim=8,
            edge_hidden_dim=16,
            nheads=2,
            scale_type="learned",
        )

        assert isinstance(attn.scale, torch.nn.Parameter)
