"""Inference pipeline configuration for model serving and retrieval."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .base import BaseConfig


class VectorDBBackend(str, Enum):
    """Supported vector database backends."""

    LOCAL = "local"  # In-memory numpy/faiss
    MILVUS = "milvus"
    MILVUS_LITE = "milvus_lite"


class SimilarityMetric(str, Enum):
    """Distance/similarity metrics for retrieval."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    L2 = "l2"


class RerankingStrategy(str, Enum):
    """Post-retrieval reranking strategies."""

    NONE = "none"
    CROSS_ENCODER = "cross_encoder"
    LLM = "llm"
    HYBRID = "hybrid"


class ModelConfig(BaseModel):
    """Configuration for embedding model used in inference."""

    model_name: str = Field(
        default="all-MiniLM-L6-v2", description="Name or path of the embedding model"
    )
    embedding_type: str = Field(
        default="sentence", description="Type of model (sentence, clip, custom)"
    )
    model_path: Path | None = Field(
        default=None, description="Local path to model weights (overrides model_name)"
    )

    # Model parameters
    max_seq_length: int | None = Field(
        default=None, description="Maximum sequence length for encoding"
    )
    normalize_embeddings: bool = Field(default=True, description="L2 normalize embeddings")

    # Performance
    batch_size: int = Field(default=32, description="Batch size for inference", ge=1, le=256)
    compile_model: bool = Field(
        default=False, description="Use torch.compile for faster inference (requires PyTorch 2.0+)"
    )


class VectorDBConfig(BaseModel):
    """Configuration for vector database backend."""

    backend: VectorDBBackend = Field(
        default=VectorDBBackend.LOCAL, description="Vector database backend to use"
    )

    # Connection settings (for remote backends)
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=19530, description="Database port", ge=1, le=65535)
    username: str | None = Field(default=None, description="Database username")
    password: str | None = Field(
        default=None, description="Database password (use env var in production)"
    )

    # Collection/Index settings
    collection_name: str = Field(
        default="exercises_embeddings", description="Name of the collection/index"
    )
    vector_field: str = Field(default="vector", description="Name of the vector field")
    dimension: int | None = Field(
        default=None, description="Vector dimension (auto-detected if None)"
    )

    # Index settings
    index_type: str = Field(default="FLAT", description="Index type (FLAT, IVF_FLAT, HNSW, etc.)")
    metric_type: SimilarityMetric = Field(
        default=SimilarityMetric.COSINE, description="Similarity metric for vector search"
    )

    # Additional parameters
    index_params: dict[str, Any] = Field(
        default_factory=dict, description="Additional index-specific parameters"
    )

    # Local backend settings
    local_path: Path | None = Field(default=None, description="Path to save the vector DB")

    # Milvus output fields
    output_fields: list[str] | None = Field(
        default=None, description="Additional fields to retrieve from Milvus"
    )

    @field_validator("password")
    @classmethod
    def warn_password(cls, v):
        """Warn if password is set directly (should use env var)."""
        if v is not None:
            import warnings

            warnings.warn(
                "Setting password directly in config is not recommended. "
                "Use environment variable instead.",
                stacklevel=2,
            )
        return v


class RetrievalConfig(BaseModel):
    """Configuration for retrieval and ranking."""

    # Basic retrieval
    top_k: int = Field(default=5, description="Number of results to retrieve", ge=1, le=100)
    min_score: float | None = Field(
        default=None, description="Minimum similarity score threshold (0-1)", ge=0.0, le=1.0
    )

    # Query processing
    query_prefix: str = Field(
        default="", description="Prefix to add to queries (e.g., 'search_query: ')"
    )
    expand_query: bool = Field(default=False, description="Use query expansion techniques")

    # Reranking
    reranking_strategy: RerankingStrategy = Field(
        default=RerankingStrategy.NONE, description="Post-retrieval reranking strategy"
    )
    reranking_model: str | None = Field(
        default=None, description="Model to use for reranking (if applicable)"
    )
    rerank_top_k: int | None = Field(
        default=None, description="Number of candidates to rerank (None = same as top_k)"
    )

    # Filtering
    filters: dict[str, Any] = Field(default_factory=dict, description="Metadata filters to apply")

    # Output fields
    return_fields: list[str] = Field(
        default=["name", "description", "combined_text"], description="Fields to return in results"
    )
    include_vectors: bool = Field(default=False, description="Include embedding vectors in results")
    include_scores: bool = Field(default=True, description="Include similarity scores in results")


class CachingConfig(BaseModel):
    """Configuration for result caching."""

    enabled: bool = Field(default=True, description="Enable query result caching")
    ttl: int = Field(default=3600, description="Cache time-to-live in seconds", ge=0)
    max_size: int = Field(default=1000, description="Maximum number of cached queries", ge=0)
    cache_embeddings: bool = Field(default=True, description="Cache query embeddings")


class InferenceConfig(BaseConfig):
    """Complete configuration for the inference pipeline."""

    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    vector_db: VectorDBConfig = Field(
        default_factory=VectorDBConfig, description="Vector database configuration"
    )
    retrieval: RetrievalConfig = Field(
        default_factory=RetrievalConfig, description="Retrieval configuration"
    )
    caching: CachingConfig = Field(
        default_factory=CachingConfig, description="Caching configuration"
    )

    # Dataset settings
    dataset_path: Path = Field(
        default=Path("data/processed/exercises_dataset.jsonl"),
        description="Path to processed dataset",
    )

    # API settings
    enable_api: bool = Field(default=True, description="Enable REST API")
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=7860, description="API port", ge=1, le=65535)
    enable_ui: bool = Field(default=True, description="Enable web UI (Gradio)")

    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable performance metrics collection")
    log_queries: bool = Field(default=True, description="Log all queries for analysis")

    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_path(cls, v):
        """Warn if dataset path doesn't exist."""
        if not Path(v).exists():
            import warnings

            warnings.warn(f"Dataset path {v} does not exist", stacklevel=2)
        return v
