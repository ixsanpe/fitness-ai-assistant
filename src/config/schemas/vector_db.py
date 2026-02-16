"""Vector database configuration schema."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class VectorDBBackend(str, Enum):
    """Supported vector database backends."""

    LOCAL = "local"
    MILVUS_LITE = "milvus_lite"


class SimilarityMetric(str, Enum):
    """Similarity metrics for retrieval."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    L2 = "l2"


class VectorDBConfig(BaseModel):
    """Vector database configuration."""

    backend: VectorDBBackend = Field(
        default=VectorDBBackend.LOCAL, description="Backend: local or milvus_lite"
    )
    local_path: Path | None = Field(
        default=None, description="Path for local Milvus database"
    )
    host: str = Field(default="localhost", description="Milvus host")
    port: int = Field(default=19530, description="Milvus port", ge=1, le=65535)
    collection_name: str = Field(
        default="exercises_embeddings", description="Collection name"
    )
    vector_field: str = Field(default="vector", description="Vector field name")
    metric_type: SimilarityMetric = Field(
        default=SimilarityMetric.COSINE, description="Similarity metric"
    )
    output_fields: list[str] | None = Field(
        default=None, description="Additional fields to retrieve"
    )
