"""Sentence embedding generator."""

import torch
from sentence_transformers import SentenceTransformer

from src.feature_pipeline.embedders.base import BaseEmbedder


class SentenceEmbedder(BaseEmbedder):
    """Generate sentence embeddings using SentenceTransformer."""

    def compute(self, items: list[dict]) -> torch.Tensor:
        existing = self.should_skip()
        if existing is not None:
            return existing

        print(f"Computing sentence embeddings with model: {self.config.embedding.model_name}")
        model = SentenceTransformer(self.config.embedding.model_name, device=self.device)
        texts = [it.get("combined_text", "") for it in items]

        embeddings = model.encode(
            texts,
            batch_size=self.config.embedding.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=self.config.embedding.normalize,
        )

        self.storage.save(
            embeddings,
            embedding_type=self.config.embedding.output_prefix,
            save_format=self.config.embedding.save_format,  # type: ignore
        )
        return embeddings
