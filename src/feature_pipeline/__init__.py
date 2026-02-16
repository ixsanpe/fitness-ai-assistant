"""Feature pipeline for dataset building and embedding generation."""

from src.feature_pipeline.embedders import (
    BaseEmbedder,
    CLIPEmbedder,
    SentenceEmbedder,
    create_embedder,
)
from src.feature_pipeline.loaders import BaseLoader, ExerciseLoader
from src.feature_pipeline.pipeline import FeaturePipeline, run_pipeline
from src.feature_pipeline.storage import EmbeddingStorage
from src.feature_pipeline.visualizers import find_similar, plot_tsne_visualization

__all__ = [
    "BaseEmbedder",
    "SentenceEmbedder",
    "CLIPEmbedder",
    "create_embedder",
    "BaseLoader",
    "ExerciseLoader",
    "EmbeddingStorage",
    "FeaturePipeline",
    "run_pipeline",
    "find_similar",
    "plot_tsne_visualization",
]
