"""Embedding generators for feature pipeline."""

from src.config import FeatureConfig
from src.feature_pipeline.embedders.base import BaseEmbedder
from src.feature_pipeline.embedders.clip import CLIPEmbedder
from src.feature_pipeline.embedders.sentence import SentenceEmbedder


def create_embedder(config: FeatureConfig, device: str) -> BaseEmbedder:
    """Factory function to create appropriate embedder."""
    if config.embedding.embedding_type == "sentence":
        return SentenceEmbedder(config, device)
    elif config.embedding.embedding_type == "clip":
        return CLIPEmbedder(config, device)
    else:
        raise ValueError(f"Unknown embedding type: {config.embedding.embedding_type}")


__all__ = ["BaseEmbedder", "SentenceEmbedder", "CLIPEmbedder", "create_embedder"]
