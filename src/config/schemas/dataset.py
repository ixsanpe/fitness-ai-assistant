"""Dataset configuration schema."""

from pathlib import Path

from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Dataset processing configuration."""

    input_path: Path = Field(default=Path("data/raw/exercises"), description="Raw data path")
    output_path: Path = Field(
        default=Path("data/processed/exercises_dataset.jsonl"), description="Output path"
    )

    # Filtering
    min_text_length: int = Field(default=10, description="Minimum text length", ge=0)
    remove_duplicates: bool = Field(default=True, description="Remove duplicates")
