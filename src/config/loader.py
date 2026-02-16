"""Configuration loader utilities."""

import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import ValidationError

from src.config.base import BaseConfig
from src.config.feature import FeatureConfig
from src.config.inference import InferenceConfig
from src.config.training_config import TrainingConfig

T = TypeVar("T", bound=BaseConfig)


class ConfigLoader:
    """Utility class for loading configurations."""

    CONFIG_REGISTRY = {
        "feature": FeatureConfig,
        "inference": InferenceConfig,
        "training": TrainingConfig,
    }

    @classmethod
    def register(cls, name: str, config_class: type[BaseConfig]):
        """Register a new config type."""
        cls.CONFIG_REGISTRY[name] = config_class

    @staticmethod
    def from_yaml(path: str | Path, config_class: type[T]) -> T:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        try:
            return config_class(**config_dict)
        except ValidationError as e:
            print(f"Configuration validation failed for {path}:")
            print(e)
            raise

    @staticmethod
    def from_json(path: str | Path, config_class: type[T]) -> T:
        """Load configuration from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            config_dict = json.load(f)

        try:
            return config_class(**config_dict)
        except ValidationError as e:
            print(f"Configuration validation failed for {path}:")
            print(e)
            raise

    @staticmethod
    def from_dict(config_dict: dict[str, Any], config_class: type[T]) -> T:
        """Load configuration from dictionary."""
        try:
            return config_class(**config_dict)
        except ValidationError as e:
            print("Configuration validation failed:")
            print(e)
            raise

    @staticmethod
    def to_yaml(config: BaseConfig, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = config.model_dump(exclude_none=True, mode="json")

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False, indent=2)

    @staticmethod
    def to_json(config: BaseConfig, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = config.model_dump(exclude_none=True, mode="json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def auto_load(cls, config_type: str, config_path: str | Path) -> BaseConfig:
        """Auto-load configuration based on type."""
        if config_type not in cls.CONFIG_REGISTRY:
            raise ValueError(
                f"Unknown config type: {config_type}. Available: {list(cls.CONFIG_REGISTRY.keys())}"
            )

        config_class = cls.CONFIG_REGISTRY[config_type]
        path = Path(config_path)

        if path.suffix in [".yaml", ".yml"]:
            return cls.from_yaml(path, config_class)
        elif path.suffix == ".json":
            return cls.from_json(path, config_class)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")


def load_config(config_type: str, config_path: str | Path, **kwargs) -> BaseConfig:
    """Load configuration with optional overrides.

    Examples:
        >>> config = load_config("feature", "configs/feature_sentence.yaml")
        >>> config = load_config("feature", "configs/feature.yaml", batch_size=32)
    """
    config = ConfigLoader.auto_load(config_type, config_path)

    if kwargs:
        config_dict = config.model_dump()
        config_dict.update(kwargs)
        config_class = ConfigLoader.CONFIG_REGISTRY[config_type]
        config = config_class(**config_dict)

    return config
