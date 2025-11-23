"""Base configuration classes shared across all pipelines."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Environment(str, Enum):
    """Deployment environment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DeviceType(str, Enum):
    """Compute device types."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DeviceConfig(BaseModel):
    """Device configuration for model execution."""

    model_config = ConfigDict(frozen=True)

    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Device to run models on. 'auto' selects best available.",
    )
    gpu_id: int | None = Field(
        default=None, description="Specific GPU ID to use when device is cuda"
    )
    mixed_precision: bool = Field(
        default=False, description="Enable mixed precision training/inference"
    )

    @field_validator("gpu_id")
    @classmethod
    def validate_gpu_id(cls, v, info):
        """Ensure gpu_id is only set when using CUDA."""
        if v is not None and info.data.get("device") != DeviceType.CUDA:
            raise ValueError("gpu_id can only be set when device is 'cuda'")
        return v


class PathConfig(BaseModel):
    """Path configuration for data and models."""

    model_config = ConfigDict(frozen=True)

    project_root: Path = Field(default=Path("."), description="Root directory of the project")
    data_dir: Path = Field(default=Path("data"), description="Base directory for all data")
    raw_data_dir: Path = Field(default=Path("data/raw"), description="Directory for raw data")
    processed_data_dir: Path = Field(
        default=Path("data/processed"), description="Directory for processed data"
    )
    embeddings_dir: Path = Field(
        default=Path("data/processed/embeddings"), description="Directory for embeddings storage"
    )
    models_dir: Path = Field(default=Path("models"), description="Directory for saved models")
    results_dir: Path = Field(
        default=Path("results"), description="Directory for results and outputs"
    )

    @field_validator(
        "project_root",
        "data_dir",
        "raw_data_dir",
        "processed_data_dir",
        "embeddings_dir",
        "models_dir",
        "results_dir",
    )
    @classmethod
    def ensure_path(cls, v: Path) -> Path:
        """Convert string to Path if needed."""
        return Path(v) if not isinstance(v, Path) else v


class BaseConfig(BaseModel):
    """Base configuration class with common settings."""

    model_config = ConfigDict(
        extra="forbid",  # Raise error on unknown fields
        validate_assignment=True,  # Validate on attribute assignment
        frozen=False,  # Allow modification after creation
    )

    name: str = Field(default="default", description="Configuration name/identifier")
    version: str = Field(default="0.1.0", description="Configuration version")
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Deployment environment"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility", ge=0)
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    device: DeviceConfig = Field(default_factory=DeviceConfig, description="Device configuration")
    paths: PathConfig = Field(default_factory=PathConfig, description="Path configuration")

    def get_absolute_path(self, relative_path: Path) -> Path:
        """Get absolute path relative to project root."""
        if relative_path.is_absolute():
            return relative_path
        return (self.paths.project_root / relative_path).resolve()

    def ensure_directories(self):
        """Create all configured directories if they don't exist."""
        dirs_to_create = [
            self.paths.data_dir,
            self.paths.raw_data_dir,
            self.paths.processed_data_dir,
            self.paths.embeddings_dir,
            self.paths.models_dir,
            self.paths.results_dir,
        ]
        for dir_path in dirs_to_create:
            abs_path = self.get_absolute_path(dir_path)
            abs_path.mkdir(parents=True, exist_ok=True)
