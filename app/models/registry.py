from typing import Any

from app.models.base import BaseModel


class ModelRegistry:
    """Registry for managing loaded ML models."""

    def __init__(self) -> None:
        self._models: dict[str, BaseModel] = {}

    def register(self, name: str, model: BaseModel) -> None:
        """Register a model under a unique name."""

        if name in self._models:
            raise ValueError(f"Model '{name}' is already registered.")

        self._models[name] = model

    def get(self, name: str) -> BaseModel:
        """Return a registered model."""

        try:
            return self._models[name]
        except KeyError as exc:
            raise KeyError(f"Model '{name}' is not registered.") from exc

    def exists(self, name: str) -> bool:
        """Check whether a model exists."""

        return name in self._models

    def list_models(self) -> list[str]:
        """Return registered model names."""

        return list(self._models.keys())

    def metadata(self) -> dict[str, dict[str, Any]]:
        """Return metadata for all registered models."""

        return {
            name: model.metadata()
            for name, model in self._models.items()
        }