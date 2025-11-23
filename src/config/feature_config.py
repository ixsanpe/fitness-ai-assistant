"""Feature pipeline configuration for data processing and embedding generation."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .base import BaseConfig


class EmbeddingType(str, Enum):
    """Types of embeddings supported."""

    SENTENCE = "sentence"
    CLIP = "clip"
    CLIP_TEXT = "clip_text"
    CLIP_IMAGE = "clip_image"
    CUSTOM = "custom"


class ImageProcessingMode(str, Enum):
    """Image processing strategies for CLIP."""

    FIRST = "first"  # Use only first image
    AVERAGE = "average"  # Average all image embeddings
    CONCATENATE = "concatenate"  # Concatenate all image embeddings
    MAX_POOL = "max_pool"  # Max pooling across images


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    embedding_type: EmbeddingType = Field(
        default=EmbeddingType.SENTENCE, description="Type of embedding to use"
    )
    model_name: str = Field(
        default="all-MiniLM-L6-v2", description="Name/path of the embedding model"
    )
    batch_size: int = Field(
        default=64, description="Batch size for embedding computation", ge=1, le=512
    )
    max_length: int | None = Field(
        default=None, description="Maximum sequence length for text embeddings"
    )
    normalize: bool = Field(default=True, description="Normalize embeddings to unit length")

    # CLIP-specific settings
    image_processing_mode: ImageProcessingMode = Field(
        default=ImageProcessingMode.FIRST, description="How to process multiple images per item"
    )
    text_weight: float = Field(
        default=0.5, description="Weight for text embeddings in multimodal (0-1)", ge=0.0, le=1.0
    )
    image_weight: float = Field(
        default=0.5, description="Weight for image embeddings in multimodal (0-1)", ge=0.0, le=1.0
    )

    # Output settings
    output_prefix: str = Field(default="embeddings", description="Prefix for saved embedding files")
    save_format: str = Field(
        default="npy",
        description="Format to save embeddings (npy, pt, safetensors)",
        pattern="^(npy|pt|safetensors)$",
    )

    @field_validator("text_weight", "image_weight")
    @classmethod
    def validate_weights_sum(cls, v, info):
        """Ensure text and image weights sum to 1.0 for multimodal."""
        if "text_weight" in info.data and "image_weight" in info.data:
            total = info.data["text_weight"] + info.data["image_weight"]
            if not (0.99 <= total <= 1.01):  # Allow small floating point error
                raise ValueError("text_weight + image_weight must equal 1.0")
        return v

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v, info):
        """Validate model name matches embedding type."""
        embedding_type = info.data.get("embedding_type")
        if embedding_type == EmbeddingType.CLIP or embedding_type == EmbeddingType.CLIP_TEXT:
            if "clip" not in v.lower():
                raise ValueError(f"Model name should contain 'clip' for {embedding_type}")
        return v


class DatasetConfig(BaseModel):
    """Configuration for dataset processing."""

    input_path: Path = Field(default=Path("data/raw/exercises"), description="Path to raw dataset")
    output_path: Path = Field(
        default=Path("data/processed/exercises_dataset.jsonl"),
        description="Path to save processed dataset",
    )

    # Text processing
    text_fields: list[str] = Field(
        default=["name", "description", "instructions"],
        description="Fields to combine for text embeddings",
    )
    text_separator: str = Field(default=" ", description="Separator for combining text fields")
    clean_text: bool = Field(
        default=True, description="Apply text cleaning (lowercase, remove special chars, etc.)"
    )

    # Image processing
    include_images: bool = Field(default=True, description="Include image paths in dataset")
    image_extensions: list[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".gif"], description="Valid image file extensions"
    )
    max_images_per_item: int | None = Field(
        default=None, description="Maximum number of images per item (None = unlimited)"
    )

    # Filtering
    min_text_length: int = Field(
        default=10, description="Minimum text length to include item", ge=0
    )
    remove_duplicates: bool = Field(
        default=True, description="Remove duplicate items based on text"
    )

    # Performance
    num_workers: int = Field(
        default=4, description="Number of workers for parallel processing", ge=1
    )


class FeatureConfig(BaseConfig):
    """Complete configuration for the feature pipeline."""

    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig, description="Embedding generation configuration"
    )
    dataset: DatasetConfig = Field(
        default_factory=DatasetConfig, description="Dataset processing configuration"
    )

    # Pipeline control
    skip_if_exists: bool = Field(
        default=True, description="Skip processing if output already exists"
    )
    overwrite: bool = Field(default=False, description="Overwrite existing outputs")
    validate_outputs: bool = Field(default=True, description="Validate outputs after generation")

    # Caching
    cache_embeddings: bool = Field(default=True, description="Cache embeddings to disk")
    cache_dir: Path | None = Field(
        default=None, description="Custom cache directory (None = use default)"
    )

    @field_validator("overwrite", "skip_if_exists")
    @classmethod
    def validate_overwrite_skip(cls, v, info):
        """Ensure overwrite and skip_if_exists are not both True."""
        if info.field_name == "overwrite" and v:
            if info.data.get("skip_if_exists", False):
                raise ValueError("Cannot set both overwrite=True and skip_if_exists=True")
        return v
