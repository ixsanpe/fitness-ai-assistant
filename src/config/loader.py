"""Configuration loader utilities for loading and validating configs from YAML/JSON."""

import json
import os
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import ValidationError

from .base import BaseConfig
from .feature_config import FeatureConfig
from .inference_config import InferenceConfig
from .training_config import TrainingConfig

T = TypeVar("T", bound=BaseConfig)


class ConfigLoader:
    """Utility class for loading and validating configurations."""

    CONFIG_TYPES = {
        "feature": FeatureConfig,
        "inference": InferenceConfig,
        "training": TrainingConfig,
    }

    @staticmethod
    def from_yaml(path: str | Path, config_class: type[T], env_override: bool = True) -> T:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file
            config_class: Pydantic config class to instantiate
            env_override: Whether to override with environment variables

        Returns:
            Validated configuration instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config validation fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            config_dict = {}

        # Override with environment variables
        if env_override:
            config_dict = ConfigLoader._apply_env_overrides(config_dict)

        # Validate and instantiate
        try:
            return config_class(**config_dict)
        except ValidationError as e:
            print(f"Configuration validation failed for {path}:")
            print(e)
            raise

    @staticmethod
    def from_json(path: str | Path, config_class: type[T], env_override: bool = True) -> T:
        """Load configuration from JSON file.

        Args:
            path: Path to JSON configuration file
            config_class: Pydantic config class to instantiate
            env_override: Whether to override with environment variables

        Returns:
            Validated configuration instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            config_dict = json.load(f)

        if env_override:
            config_dict = ConfigLoader._apply_env_overrides(config_dict)

        try:
            return config_class(**config_dict)
        except ValidationError as e:
            print(f"Configuration validation failed for {path}:")
            print(e)
            raise

    @staticmethod
    def from_dict(config_dict: dict[str, Any], config_class: type[T]) -> T:
        """Load configuration from dictionary.

        Args:
            config_dict: Configuration dictionary
            config_class: Pydantic config class to instantiate

        Returns:
            Validated configuration instance
        """
        try:
            return config_class(**config_dict)
        except ValidationError as e:
            print("Configuration validation failed:")
            print(e)
            raise

    @staticmethod
    def to_yaml(config: BaseConfig, path: str | Path, exclude_none: bool = True) -> None:
        """Save configuration to YAML file.

        Args:
            config: Configuration instance to save
            path: Output path for YAML file
            exclude_none: Whether to exclude None values
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Use mode="json" to serialize enums as their values
        config_dict = config.model_dump(exclude_none=exclude_none, mode="json")

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                config_dict,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=80,
                indent=2,
            )

    @staticmethod
    def to_json(
        config: BaseConfig, path: str | Path, exclude_none: bool = True, indent: int = 2
    ) -> None:
        """Save configuration to JSON file.

        Args:
            config: Configuration instance to save
            path: Output path for JSON file
            exclude_none: Whether to exclude None values
            indent: JSON indentation level
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = config.model_dump(exclude_none=exclude_none, mode="json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=indent, ensure_ascii=False)

    @staticmethod
    def _apply_env_overrides(config_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply environment variable overrides to config dictionary.

        Environment variables should be prefixed with 'FITNESS_AI_' and use
        double underscores to separate nested keys, e.g.:
        FITNESS_AI_MODEL__MODEL_NAME=my-model
        FITNESS_AI_DEVICE__DEVICE=cuda

        Args:
            config_dict: Configuration dictionary to modify

        Returns:
            Modified configuration dictionary
        """
        prefix = "FITNESS_AI_"

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # Remove prefix and split by double underscore
            config_key = key[len(prefix) :].lower()
            keys = config_key.split("__")

            # Navigate to the nested dict
            current = config_dict
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            # Set the value (try to parse as JSON for complex types)
            try:
                current[keys[-1]] = json.loads(value)
            except json.JSONDecodeError:
                current[keys[-1]] = value

        return config_dict

    @classmethod
    def auto_load(
        cls, config_type: str, config_path: str | Path | None = None, env_override: bool = True
    ) -> FeatureConfig | InferenceConfig | TrainingConfig:
        """Auto-load configuration based on type.

        Args:
            config_type: Type of config ('feature', 'inference', 'training')
            config_path: Path to config file (auto-detected if None)
            env_override: Whether to override with environment variables

        Returns:
            Loaded configuration instance
        """
        if config_type not in cls.CONFIG_TYPES:
            raise ValueError(
                f"Unknown config type: {config_type}. "
                f"Must be one of {list(cls.CONFIG_TYPES.keys())}"
            )

        config_class = cls.CONFIG_TYPES[config_type]

        # Load based on file extension
        path = Path(config_path)
        if path.suffix in [".yaml", ".yml"]:
            return cls.from_yaml(path, config_class, env_override)
        elif path.suffix == ".json":
            return cls.from_json(path, config_class, env_override)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")


def load_config(
    config_type: str, config_path: str | Path | None = None, **kwargs
) -> FeatureConfig | InferenceConfig | TrainingConfig:
    """Convenience function to load configuration.

    Args:
        config_type: Type of config ('feature', 'inference', 'training')
        config_path: Path to config file (optional)
        **kwargs: Additional keyword arguments to override config values

    Returns:
        Loaded and validated configuration

    Examples:
        >>> # Load from file
        >>> config = load_config("inference", "configs/inference.yaml")

        >>> # Load with overrides
        >>> config = load_config("inference", top_k=10, device="cuda")
    """
    config = ConfigLoader.auto_load(config_type, config_path)

    # Apply any kwargs overrides
    if kwargs:
        config_dict = config.model_dump()
        config_dict.update(kwargs)
        config_class = ConfigLoader.CONFIG_TYPES[config_type]
        config = config_class(**config_dict)

    return config
