"""Base configuration class."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.config.schemas import DeviceConfig, PathConfig


class BaseConfig(BaseModel):
    """Base configuration with common fields."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(..., description="Configuration name")
    device: DeviceConfig = Field(default_factory=DeviceConfig, description="Device config")
    paths: PathConfig = Field(default_factory=PathConfig, description="Path config")

    def ensure_directories(self) -> None:
        """Create all directories specified in paths config."""
        for field_name in type(self.paths).model_fields:
            value = getattr(self.paths, field_name)
            if value is None:
                continue

            path = Path(value)
            # If it's a file path (has extension), create parent dir
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
            # Otherwise, create the directory itself
            else:
                path.mkdir(parents=True, exist_ok=True)
