"""VAE-specific metrics.

Metrics for evaluating variational autoencoders.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from lr_geom.training.metrics.base import Metric


class KLDivergence(Metric):
    """KL Divergence metric for VAE latent space.

    Computes KL divergence between the learned posterior q(z|x) and
    the standard normal prior p(z) = N(0, I).

    KL(q || p) = -0.5 * sum(1 + log(var) - mu^2 - var)

    Expected keys in outputs:
    - outputs["mu"]: Mean of posterior
    - outputs["logvar"]: Log-variance of posterior
    """

    def __init__(
        self,
        mu_key: str = "mu",
        logvar_key: str = "logvar",
    ) -> None:
        """Initialize KL divergence metric.

        Args:
            mu_key: Key for mean in outputs dict.
            logvar_key: Key for log-variance in outputs dict.
        """
        self._mu_key = mu_key
        self._logvar_key = logvar_key
        self._total_kl: float = 0.0
        self._count: int = 0

    @property
    def name(self) -> str:
        """Return metric name."""
        return "kl_divergence"

    def reset(self) -> None:
        """Reset metric state."""
        self._total_kl = 0.0
        self._count = 0

    def update(
        self,
        batch: dict[str, Any],
        outputs: dict[str, Tensor] | Tensor,
    ) -> None:
        """Update with batch results.

        Args:
            batch: Input batch (not used for KL).
            outputs: Model outputs with mu and logvar.
        """
        if not isinstance(outputs, dict):
            return

        mu = outputs.get(self._mu_key)
        logvar = outputs.get(self._logvar_key)

        if mu is None or logvar is None:
            return

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl = kl.sum().item()

        self._total_kl += kl
        self._count += mu.shape[0]  # Batch size

    def compute(self) -> float:
        """Compute average KL divergence per sample.

        Returns:
            Average KL divergence.
        """
        if self._count == 0:
            return 0.0
        return self._total_kl / self._count


class ReconstructionLoss(Metric):
    """Track reconstruction loss component from VAE.

    Simply tracks the "reconstruction" or "recon_loss" key from outputs.
    """

    def __init__(self, key: str = "recon_loss") -> None:
        """Initialize metric.

        Args:
            key: Key for reconstruction loss in outputs.
        """
        self._key = key
        self._total: float = 0.0
        self._count: int = 0

    @property
    def name(self) -> str:
        """Return metric name."""
        return "recon_loss"

    def reset(self) -> None:
        """Reset metric state."""
        self._total = 0.0
        self._count = 0

    def update(
        self,
        batch: dict[str, Any],
        outputs: dict[str, Tensor] | Tensor,
    ) -> None:
        """Update with batch results."""
        if not isinstance(outputs, dict):
            return

        loss = outputs.get(self._key)
        if loss is None:
            return

        if isinstance(loss, Tensor):
            loss = loss.item()

        self._total += loss
        self._count += 1

    def compute(self) -> float:
        """Compute average reconstruction loss."""
        if self._count == 0:
            return 0.0
        return self._total / self._count


class ELBO(Metric):
    """Evidence Lower Bound metric.

    Tracks the full ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))

    For VAE, this is typically: -recon_loss - beta * kl_loss
    """

    def __init__(
        self,
        recon_key: str = "recon_loss",
        kl_key: str = "kl_loss",
        beta: float = 1.0,
    ) -> None:
        """Initialize ELBO metric.

        Args:
            recon_key: Key for reconstruction loss in outputs.
            kl_key: Key for KL loss in outputs.
            beta: Weight for KL term (for beta-VAE).
        """
        self._recon_key = recon_key
        self._kl_key = kl_key
        self._beta = beta
        self._total_elbo: float = 0.0
        self._count: int = 0

    @property
    def name(self) -> str:
        """Return metric name."""
        return "elbo"

    def reset(self) -> None:
        """Reset metric state."""
        self._total_elbo = 0.0
        self._count = 0

    def update(
        self,
        batch: dict[str, Any],
        outputs: dict[str, Tensor] | Tensor,
    ) -> None:
        """Update with batch results."""
        if not isinstance(outputs, dict):
            return

        recon = outputs.get(self._recon_key)
        kl = outputs.get(self._kl_key)

        if recon is None or kl is None:
            return

        if isinstance(recon, Tensor):
            recon = recon.item()
        if isinstance(kl, Tensor):
            kl = kl.item()

        # ELBO = -recon_loss - beta * kl_loss (negative because we minimize loss)
        elbo = -(recon + self._beta * kl)

        self._total_elbo += elbo
        self._count += 1

    def compute(self) -> float:
        """Compute average ELBO."""
        if self._count == 0:
            return 0.0
        return self._total_elbo / self._count
