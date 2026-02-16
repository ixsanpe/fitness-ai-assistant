"""Path configuration schema."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class PathConfig(BaseModel):
    """Path configuration for data and models."""

    project_root: Path = Field(default=Path("."), description="Project root")
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    raw_data_dir: Path = Field(default=Path("data/raw"), description="Raw data")
    processed_data_dir: Path = Field(default=Path("data/processed"), description="Processed data")
    embeddings_dir: Path = Field(default=Path("data/processed/embeddings"), description="Embeddings")
    models_dir: Path = Field(default=Path("models"), description="Models")
    results_dir: Path = Field(default=Path("results"), description="Results")

    @field_validator("*")
    @classmethod
    def ensure_path(cls, v):
        """Convert to Path if needed."""
        return Path(v) if not isinstance(v, Path) else v
