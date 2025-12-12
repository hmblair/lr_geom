"""Tests for lr_geom training module.

Tests cover:
- Utilities (set_seed, get_device)
- Callbacks (Callback, ModelCheckpoint, EarlyStopping, ProgressBar)
- Metrics (Metric, RMSD, KLDivergence)
- Trainer (training loop, checkpointing)
- Model Registry (registration, building)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from lr_geom.training import (
    # Utilities
    set_seed,
    get_device,
    move_batch_to_device,
    # Callbacks
    Callback,
    CallbackList,
    ModelCheckpoint,
    EarlyStopping,
    ProgressBar,
    # Metrics
    Metric,
    MetricCollection,
    AverageMeter,
    RMSD,
    MSE,
    KLDivergence,
    ReconstructionLoss,
    ELBO,
    # Trainer
    Trainer,
    TrainerConfig,
    # Factory
    ModelRegistry,
    model_registry,
)


# ============================================================================
# UTILITIES TESTS
# ============================================================================


class TestSetSeed:
    """Tests for set_seed utility."""

    def test_reproducibility(self):
        """Test that same seed produces same random numbers."""
        set_seed(42)
        x1 = torch.randn(10)

        set_seed(42)
        x2 = torch.randn(10)

        assert torch.allclose(x1, x2)

    def test_different_seeds(self):
        """Test that different seeds produce different numbers."""
        set_seed(42)
        x1 = torch.randn(10)

        set_seed(123)
        x2 = torch.randn(10)

        assert not torch.allclose(x1, x2)


class TestGetDevice:
    """Tests for get_device utility."""

    def test_cpu_preference(self):
        """Test CPU device selection."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_auto_preference(self):
        """Test auto device selection returns valid device."""
        device = get_device("auto")
        assert device.type in ["cpu", "cuda", "mps"]

    def test_invalid_cuda_raises(self):
        """Test requesting unavailable CUDA raises error."""
        if torch.cuda.is_available():
            pytest.skip("CUDA is available")
        with pytest.raises(RuntimeError, match="CUDA.*not available"):
            get_device("cuda")

    def test_invalid_mps_raises(self):
        """Test requesting unavailable MPS raises error."""
        if torch.backends.mps.is_available():
            pytest.skip("MPS is available")
        with pytest.raises(RuntimeError, match="MPS.*not available"):
            get_device("mps")


class TestMoveBatchToDevice:
    """Tests for move_batch_to_device utility."""

    def test_moves_tensors(self):
        """Test tensors are moved to device."""
        batch = {
            "coords": torch.randn(10, 3),
            "features": torch.randn(10, 16),
        }
        device = torch.device("cpu")
        moved = move_batch_to_device(batch, device)

        assert moved["coords"].device == device
        assert moved["features"].device == device

    def test_preserves_non_tensors(self):
        """Test non-tensor values are preserved."""
        batch = {
            "coords": torch.randn(10, 3),
            "name": "test",
            "count": 42,
        }
        device = torch.device("cpu")
        moved = move_batch_to_device(batch, device)

        assert moved["name"] == "test"
        assert moved["count"] == 42


# ============================================================================
# CALLBACK TESTS
# ============================================================================


class TestCallback:
    """Tests for Callback base class."""

    def test_default_methods_do_nothing(self):
        """Test default callback methods don't raise."""
        cb = Callback()
        cb.on_train_begin()
        cb.on_train_end()
        cb.on_epoch_begin(0)
        cb.on_epoch_end(0, {"loss": 1.0})
        cb.on_train_batch_begin(0)
        cb.on_train_batch_end(0, 1.0)
        cb.on_val_batch_begin(0)
        cb.on_val_batch_end(0, 1.0)


class TestCallbackList:
    """Tests for CallbackList container."""

    def test_append_callback(self):
        """Test callbacks can be appended."""
        cb_list = CallbackList()
        cb = Callback()
        cb_list.append(cb)

        assert len(cb_list) == 1

    def test_iterate_callbacks(self):
        """Test callbacks can be iterated."""
        cbs = [Callback(), Callback()]
        cb_list = CallbackList(cbs)

        assert list(cb_list) == cbs

    def test_dispatch_events(self):
        """Test events are dispatched to all callbacks."""
        class CountingCallback(Callback):
            def __init__(self):
                super().__init__()
                self.count = 0

            def on_epoch_begin(self, epoch):
                self.count += 1

        cb1 = CountingCallback()
        cb2 = CountingCallback()
        cb_list = CallbackList([cb1, cb2])

        cb_list.on_epoch_begin(0)

        assert cb1.count == 1
        assert cb2.count == 1


class TestEarlyStopping:
    """Tests for EarlyStopping callback."""

    def test_stops_after_patience(self):
        """Test training stops after patience epochs without improvement."""
        es = EarlyStopping(monitor="val_loss", patience=3)

        # Create mock trainer
        class MockTrainer:
            stop_training = False

        trainer = MockTrainer()
        es.set_trainer(trainer)
        es.on_train_begin()

        # Simulate improving then not improving
        es.on_epoch_end(0, {"val_loss": 1.0})  # Best
        es.on_epoch_end(1, {"val_loss": 0.9})  # Better
        es.on_epoch_end(2, {"val_loss": 0.95})  # Worse
        es.on_epoch_end(3, {"val_loss": 0.95})  # Worse
        assert not trainer.stop_training

        es.on_epoch_end(4, {"val_loss": 0.95})  # Patience reached
        assert trainer.stop_training

    def test_resets_on_improvement(self):
        """Test patience counter resets on improvement."""
        es = EarlyStopping(monitor="val_loss", patience=3)

        class MockTrainer:
            stop_training = False

        trainer = MockTrainer()
        es.set_trainer(trainer)
        es.on_train_begin()

        es.on_epoch_end(0, {"val_loss": 1.0})  # Best
        es.on_epoch_end(1, {"val_loss": 1.1})  # Worse, wait=1
        es.on_epoch_end(2, {"val_loss": 0.9})  # Better - resets wait=0
        es.on_epoch_end(3, {"val_loss": 1.0})  # Worse, wait=1
        es.on_epoch_end(4, {"val_loss": 1.0})  # Worse, wait=2

        # Should not have stopped yet (patience=3, wait=2)
        assert not trainer.stop_training

    def test_max_mode(self):
        """Test max mode for metrics where higher is better."""
        es = EarlyStopping(monitor="val_acc", patience=2, mode="max")

        class MockTrainer:
            stop_training = False

        trainer = MockTrainer()
        es.set_trainer(trainer)
        es.on_train_begin()

        es.on_epoch_end(0, {"val_acc": 0.8})  # Best
        es.on_epoch_end(1, {"val_acc": 0.7})  # Worse
        es.on_epoch_end(2, {"val_acc": 0.7})  # Worse

        assert trainer.stop_training


# ============================================================================
# METRIC TESTS
# ============================================================================


class TestAverageMeter:
    """Tests for AverageMeter utility."""

    def test_average_computation(self):
        """Test average is computed correctly."""
        meter = AverageMeter()
        meter.update(1.0)
        meter.update(2.0)
        meter.update(3.0)

        assert meter.avg == pytest.approx(2.0)

    def test_weighted_average(self):
        """Test weighted average works correctly."""
        meter = AverageMeter()
        meter.update(1.0, n=2)  # 2 samples with value 1.0
        meter.update(4.0, n=2)  # 2 samples with value 4.0

        assert meter.avg == pytest.approx(2.5)

    def test_reset(self):
        """Test reset clears state."""
        meter = AverageMeter()
        meter.update(10.0)
        meter.reset()

        assert meter.avg == 0.0
        assert meter.count == 0


class TestRMSDMetric:
    """Tests for RMSD metric."""

    def test_name(self):
        """Test metric name."""
        rmsd = RMSD()
        assert rmsd.name == "rmsd"

    def test_compute_rmsd(self):
        """Test RMSD computation."""
        rmsd = RMSD()
        rmsd.reset()

        # Perfect reconstruction
        batch = {"coords": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])}
        outputs = {"reconstruction": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])}
        rmsd.update(batch, outputs)

        assert rmsd.compute() == pytest.approx(0.0)

    def test_compute_nonzero_rmsd(self):
        """Test RMSD computation with error."""
        rmsd = RMSD()
        rmsd.reset()

        batch = {"coords": torch.tensor([[0.0, 0.0, 0.0]])}
        outputs = {"reconstruction": torch.tensor([[1.0, 0.0, 0.0]])}  # 1 unit error
        rmsd.update(batch, outputs)

        assert rmsd.compute() == pytest.approx(1.0)


class TestKLDivergenceMetric:
    """Tests for KL divergence metric."""

    def test_name(self):
        """Test metric name."""
        kl = KLDivergence()
        assert kl.name == "kl_divergence"

    def test_compute_kl_zero_for_standard_normal(self):
        """Test KL is ~0 for standard normal."""
        kl = KLDivergence()
        kl.reset()

        # mu=0, logvar=0 => KL ~ 0
        outputs = {
            "mu": torch.zeros(10, 16),
            "logvar": torch.zeros(10, 16),
        }
        kl.update({}, outputs)

        assert kl.compute() == pytest.approx(0.0, abs=1e-6)

    def test_compute_kl_positive(self):
        """Test KL is positive for non-standard normal."""
        kl = KLDivergence()
        kl.reset()

        # Non-zero mu gives positive KL
        outputs = {
            "mu": torch.ones(10, 16),
            "logvar": torch.zeros(10, 16),
        }
        kl.update({}, outputs)

        assert kl.compute() > 0


class TestMetricCollection:
    """Tests for MetricCollection container."""

    def test_reset_all(self):
        """Test reset calls reset on all metrics."""
        collection = MetricCollection([RMSD(), MSE()])
        collection.reset()
        # Should not raise

    def test_compute_all(self):
        """Test compute returns dict from all metrics."""
        collection = MetricCollection([RMSD(), MSE()])
        collection.reset()

        batch = {"coords": torch.zeros(10, 3)}
        outputs = {"reconstruction": torch.zeros(10, 3)}
        collection.update(batch, outputs)

        results = collection.compute()
        assert "rmsd" in results
        assert "mse" in results


# ============================================================================
# TRAINER TESTS
# ============================================================================


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, in_size=16, out_size=3):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, coords, features):
        # Simple model: features (N, F) -> output (N, out_size)
        # Handle both (N, F) and (B, N, F) shapes
        if features.dim() == 3:
            # (B, N, F) -> process each batch
            B, N, F = features.shape
            x = features.view(B * N, F)
            out = self.linear(x)
            out = out.view(B, N, -1)
        else:
            # (N, F) -> direct
            out = self.linear(features)
        return {"reconstruction": out}


class TestTrainer:
    """Tests for Trainer class."""

    def test_fit_runs_without_error(self):
        """Test basic training runs."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def loss_fn(batch, outputs):
            pred = outputs["reconstruction"]
            target = batch["coords"]
            return ((pred - target) ** 2).mean()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )

        # Create simple data
        data = [
            {"coords": torch.randn(10, 3), "features": torch.randn(10, 16)},
            {"coords": torch.randn(10, 3), "features": torch.randn(10, 16)},
        ]

        history = trainer.fit(data, epochs=2)

        assert "train_loss" in history
        assert len(history["train_loss"]) == 2

    def test_fit_with_callbacks(self):
        """Test training with callbacks."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def loss_fn(batch, outputs):
            return torch.tensor(1.0, requires_grad=True)

        # Track callback calls
        call_log = []

        class LoggingCallback(Callback):
            def on_train_begin(self):
                call_log.append("train_begin")

            def on_epoch_begin(self, epoch):
                call_log.append(f"epoch_begin_{epoch}")

            def on_epoch_end(self, epoch, metrics):
                call_log.append(f"epoch_end_{epoch}")

            def on_train_end(self):
                call_log.append("train_end")

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            callbacks=[LoggingCallback()],
        )

        data = [{"coords": torch.randn(5, 3), "features": torch.randn(5, 16)}]
        trainer.fit(data, epochs=2)

        assert "train_begin" in call_log
        assert "epoch_begin_0" in call_log
        assert "epoch_end_0" in call_log
        assert "epoch_begin_1" in call_log
        assert "epoch_end_1" in call_log
        assert "train_end" in call_log

    def test_fit_with_validation(self):
        """Test training with validation data."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def loss_fn(batch, outputs):
            return torch.tensor(1.0, requires_grad=True)

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )

        train_data = [{"coords": torch.randn(5, 3), "features": torch.randn(5, 16)}]
        val_data = [{"coords": torch.randn(5, 3), "features": torch.randn(5, 16)}]

        history = trainer.fit(train_data, val_data, epochs=2)

        assert "train_loss" in history
        assert "val_loss" in history

    def test_multi_term_loss(self):
        """Test loss function returning dict of components."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def loss_fn(batch, outputs):
            return {
                "recon_loss": torch.tensor(1.0, requires_grad=True),
                "kl_loss": torch.tensor(0.1, requires_grad=True),
            }

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=TrainerConfig(loss_weights={"recon_loss": 1.0, "kl_loss": 0.01}),
        )

        data = [{"coords": torch.randn(5, 3), "features": torch.randn(5, 16)}]
        history = trainer.fit(data, epochs=1)

        assert "train_loss" in history
        assert "train_recon_loss" in history
        assert "train_kl_loss" in history

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def loss_fn(batch, outputs):
            return torch.tensor(1.0, requires_grad=True)

        trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            trainer.save_checkpoint(str(path), extra_key="extra_value")

            # Create new trainer and load
            model2 = SimpleModel()
            optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
            trainer2 = Trainer(model=model2, optimizer=optimizer2, loss_fn=loss_fn)

            extra = trainer2.load_checkpoint(str(path))

            assert extra["extra_key"] == "extra_value"

    def test_stop_training_flag(self):
        """Test stop_training flag stops training."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def loss_fn(batch, outputs):
            return torch.tensor(1.0, requires_grad=True)

        class StopAtEpoch1(Callback):
            def on_epoch_end(self, epoch, metrics):
                if epoch >= 1:
                    self.trainer.stop_training = True

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            callbacks=[StopAtEpoch1()],
        )

        data = [{"coords": torch.randn(5, 3), "features": torch.randn(5, 16)}]
        history = trainer.fit(data, epochs=100)  # Would run 100 but stops at 2

        assert len(history["train_loss"]) == 2


# ============================================================================
# MODEL REGISTRY TESTS
# ============================================================================


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_and_build(self):
        """Test model registration and building."""
        registry = ModelRegistry()

        @registry.register("test_model")
        def build_test_model(hidden_size=64):
            return nn.Linear(hidden_size, 10)

        model = registry.build("test_model", hidden_size=32)

        assert isinstance(model, nn.Linear)
        assert model.in_features == 32

    def test_list_models(self):
        """Test listing registered models."""
        registry = ModelRegistry()
        registry.register_builder("model_a", lambda: nn.Linear(10, 10))
        registry.register_builder("model_b", lambda: nn.Linear(20, 20))

        models = registry.list_models()

        assert "model_a" in models
        assert "model_b" in models

    def test_unknown_model_raises(self):
        """Test building unknown model raises KeyError."""
        registry = ModelRegistry()

        with pytest.raises(KeyError, match="Unknown model"):
            registry.build("nonexistent")

    def test_contains(self):
        """Test __contains__ method."""
        registry = ModelRegistry()
        registry.register_builder("exists", lambda: nn.Linear(10, 10))

        assert "exists" in registry
        assert "not_exists" not in registry


class TestGlobalRegistry:
    """Tests for global model_registry instance."""

    def test_contains_builtin_models(self):
        """Test global registry has built-in models."""
        assert "equivariant_vae" in model_registry
        assert "embedding_and_vae" in model_registry


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestTrainerIntegration:
    """Integration tests for the full training system."""

    def test_full_training_pipeline(self):
        """Test complete training pipeline with callbacks and metrics."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        def loss_fn(batch, outputs):
            pred = outputs["reconstruction"]
            target = batch["coords"]
            return ((pred - target) ** 2).mean()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                callbacks=[
                    EarlyStopping(patience=5, monitor="train_loss"),
                ],
                metrics=[RMSD()],
            )

            # Create data
            train_data = [
                {"coords": torch.randn(10, 3), "features": torch.randn(10, 16)}
                for _ in range(5)
            ]

            history = trainer.fit(train_data, epochs=3)

            # Check history was recorded
            assert "train_loss" in history
            assert "train_rmsd" in history
            assert len(history["train_loss"]) == 3
