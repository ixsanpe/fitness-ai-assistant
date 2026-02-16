"""Base embedder interface."""

from abc import ABC, abstractmethod

import torch

from src.config import FeatureConfig
from src.feature_pipeline.storage.embeddings import EmbeddingStorage


class BaseEmbedder(ABC):
    """Abstract base class for embedding generation."""

    def __init__(self, config: FeatureConfig, device: str):
        self.config = config
        self.device = device
        self.storage = EmbeddingStorage(config.paths.embeddings_dir, device=self.device)

    def should_skip(self) -> torch.Tensor | None:
        """Check if embeddings already exist and should be reused."""
        if self.config.embedding.skip_if_exists and not self.config.embedding.overwrite:
            embeddings = self.storage.load(self.config.embedding.embedding_type)
            if embeddings is not None:
                print("Using existing embeddings (skip_if_exists=True)")
                return embeddings
        return None

    @abstractmethod
    def compute(self, items: list[dict]) -> torch.Tensor:
        """Compute embeddings for items."""
        pass
