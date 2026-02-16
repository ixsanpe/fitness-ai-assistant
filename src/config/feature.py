"""Feature pipeline configuration."""

from pydantic import Field

from src.config.base import BaseConfig
from src.config.schemas import DatasetConfig, EmbeddingConfig


class FeatureConfig(BaseConfig):
    """Configuration for feature pipeline."""

    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig, description="Embedding config"
    )
    dataset: DatasetConfig = Field(default_factory=DatasetConfig, description="Dataset config")
