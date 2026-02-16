"""Data loaders for embeddings and metadata."""

from src.inference_pipeline.loaders.embeddings import load_embeddings
from src.inference_pipeline.loaders.metadata import load_metadata

__all__ = ["load_embeddings", "load_metadata"]
