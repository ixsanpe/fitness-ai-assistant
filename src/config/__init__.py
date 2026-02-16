"""Configuration module for fitness-ai-assistant."""

from src.config.base import BaseConfig
from src.config.feature import FeatureConfig
from src.config.inference import InferenceConfig
from src.config.loader import ConfigLoader, load_config
from src.config.schemas import (
    DatasetConfig,
    DeviceConfig,
    DeviceType,
    EmbeddingConfig,
    EmbeddingType,
    PathConfig,
)
from src.config.training_config import TrainingConfig

__all__ = [
    # Base
    "BaseConfig",
    # Schemas
    "DeviceType",
    "DeviceConfig",
    "PathConfig",
    "EmbeddingType",
    "EmbeddingConfig",
    "DatasetConfig",
    # Pipelines
    "FeatureConfig",
    "InferenceConfig",
    "TrainingConfig",
    # Loader
    "ConfigLoader",
    "load_config",
]
