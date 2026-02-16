"""Model configuration schema."""

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Embedding model configuration."""

    model_name: str = Field(
        default="all-MiniLM-L6-v2", description="HuggingFace model name"
    )
    embedding_type: str = Field(
        default="sentence", description="Type: sentence or clip"
    )
