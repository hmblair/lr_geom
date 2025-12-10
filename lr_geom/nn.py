"""Basic neural network building blocks.

Simple utility layers used by geometric models.

Classes:
    DenseNetwork: Multi-layer perceptron with configurable architecture
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["DenseNetwork"]


class DenseNetwork(nn.Module):
    """Multi-layer perceptron with configurable architecture.

    Args:
        in_size: Input feature dimension.
        out_size: Output feature dimension.
        hidden_sizes: List of hidden layer dimensions.
        bias: Whether to use bias in linear layers.
        dropout: Dropout probability.
        activation: Activation function.

    Example:
        >>> mlp = DenseNetwork(64, 10, hidden_sizes=[128, 64])
        >>> x = torch.randn(32, 64)
        >>> output = mlp(x)  # shape: (32, 10)
    """

    def __init__(
        self: DenseNetwork,
        in_size: int,
        out_size: int,
        hidden_sizes: list[int] | None = None,
        bias: bool = True,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if not isinstance(in_size, int) or in_size < 1:
            raise ValueError(f"in_size must be a positive integer, got {in_size}")
        if not isinstance(out_size, int) or out_size < 1:
            raise ValueError(f"out_size must be a positive integer, got {out_size}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        if hidden_sizes is None:
            hidden_sizes = []
        if activation is None:
            activation = nn.ReLU()

        features = [in_size] + hidden_sizes + [out_size]

        layers = []
        for l1, l2 in zip(features[:-1], features[1:]):
            layers.append(nn.Linear(l1, l2, bias))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self: DenseNetwork, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (..., in_size).

        Returns:
            Output tensor of shape (..., out_size).
        """
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)

        return self.layers[-1](x)
