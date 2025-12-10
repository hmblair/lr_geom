"""Basic neural network building blocks.

Simple utility layers used by geometric models.
"""
from __future__ import annotations

import torch
import torch.nn as nn


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)

        return self.layers[-1](x)
