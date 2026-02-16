"""Inference pipeline configuration."""

from pathlib import Path

from pydantic import Field

from src.config.base import BaseConfig
from src.config.schemas import ModelConfig, VectorDBConfig


class InferenceConfig(BaseConfig):
    """Configuration for inference pipeline."""

    model: ModelConfig = Field(
        default_factory=ModelConfig, description="Model configuration"
    )
    vector_db: VectorDBConfig = Field(
        default_factory=VectorDBConfig, description="Vector DB configuration"
    )
    dataset_path: Path = Field(
        default=Path("data/processed/exercises_dataset.jsonl"),
        description="Path to dataset",
    )
