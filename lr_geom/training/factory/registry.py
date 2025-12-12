"""Model registry for building models from configuration.

Provides a centralized registry for model builder functions.
"""
from __future__ import annotations

from typing import Any, Callable, TypeVar

import torch.nn as nn

T = TypeVar("T", bound=nn.Module)
BuilderFunc = Callable[..., nn.Module]


class ModelRegistry:
    """Registry for model builder functions.

    Allows registration of model builders by name for easy instantiation.

    Example:
        registry = ModelRegistry()

        @registry.register("my_model")
        def build_my_model(hidden_size: int = 64) -> nn.Module:
            return MyModel(hidden_size)

        model = registry.build("my_model", hidden_size=128)
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._builders: dict[str, BuilderFunc] = {}

    def register(
        self,
        name: str,
    ) -> Callable[[BuilderFunc], BuilderFunc]:
        """Register a model builder function.

        Can be used as a decorator:
            @registry.register("model_name")
            def build_model(...):
                ...

        Args:
            name: Name to register the builder under.

        Returns:
            Decorator function.
        """
        def decorator(func: BuilderFunc) -> BuilderFunc:
            self._builders[name] = func
            return func
        return decorator

    def register_builder(self, name: str, builder: BuilderFunc) -> None:
        """Register a builder function directly.

        Args:
            name: Name to register under.
            builder: Builder function.
        """
        self._builders[name] = builder

    def build(self, name: str, **kwargs: Any) -> nn.Module:
        """Build a model by name.

        Args:
            name: Registered model name.
            **kwargs: Arguments to pass to the builder.

        Returns:
            Built model instance.

        Raises:
            KeyError: If name is not registered.
        """
        if name not in self._builders:
            available = ", ".join(sorted(self._builders.keys()))
            raise KeyError(
                f"Unknown model '{name}'. Available models: {available}"
            )

        return self._builders[name](**kwargs)

    def list_models(self) -> list[str]:
        """List all registered model names.

        Returns:
            List of registered names.
        """
        return sorted(self._builders.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a model is registered."""
        return name in self._builders

    def __len__(self) -> int:
        """Return number of registered models."""
        return len(self._builders)


# Global registry instance
model_registry = ModelRegistry()
