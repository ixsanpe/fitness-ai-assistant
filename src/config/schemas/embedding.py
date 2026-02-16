"""Embedding configuration schema."""

from enum import Enum

from pydantic import BaseModel, Field


class EmbeddingType(str, Enum):
    """Embedding model types."""

    SENTENCE = "sentence"
    CLIP = "clip"


class EmbeddingConfig(BaseModel):
    """Embedding generation configuration."""

    embedding_type: EmbeddingType = Field(default=EmbeddingType.SENTENCE, description="Embedding type")
    model_name: str = Field(default="all-MiniLM-L6-v2", description="Model name or path")
    batch_size: int = Field(default=64, description="Batch size", ge=1, le=512)
    max_length: int | None = Field(
        default=None, description="Max sequence length for text models", ge=1
    )
    skip_if_exists: bool = Field(default=False, description="Skip if embeddings exist")
    overwrite: bool = Field(default=False, description="Overwrite existing embeddings")
    normalize: bool = Field(default=True, description="Normalize embeddings")

    # CLIP-specific
    image_processing_mode: str = Field(
        default="average",
        description="Image processing mode for CLIP",
        pattern="^(average|first|max_pools)$"
    )
    text_weight: float = Field(default=0.5, description="Weight for text embeddings in CLIP")
    image_weight: float = Field(default=0.5, description="Weight for image embeddings in CLIP")

    # Output
    output_prefix: str = Field(default="embeddings", description="Output file prefix")
    save_format: str = Field(default="npy", description="Save format (npy or pt)")
