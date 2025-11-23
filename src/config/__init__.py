"""Configuration module for fitness-ai-assistant.

This module provides Pydantic-based configuration management for:
- Feature pipeline (data processing, embeddings)
- Training pipeline (model training)
- Inference pipeline (model serving, retrieval)
"""

from .base import BaseConfig, DeviceConfig, PathConfig
from .feature_config import DatasetConfig, EmbeddingConfig, FeatureConfig
from .inference_config import InferenceConfig, ModelConfig, RetrievalConfig, VectorDBConfig
from .loader import ConfigLoader, load_config
from .training_config import OptimizerConfig, SchedulerConfig, TrainingConfig

__all__ = [
    # Base
    "BaseConfig",
    "DeviceConfig",
    "PathConfig",
    # Feature
    "FeatureConfig",
    "EmbeddingConfig",
    "DatasetConfig",
    # Inference
    "InferenceConfig",
    "ModelConfig",
    "RetrievalConfig",
    "VectorDBConfig",
    # Training
    "TrainingConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    # Utilities
    "ConfigLoader",
    "load_config",
]
