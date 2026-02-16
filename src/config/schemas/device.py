"""Device configuration schema."""

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class DeviceType(str, Enum):
    """Compute device types."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


class DeviceConfig(BaseModel):
    """Device configuration for model execution."""

    device: DeviceType = Field(default=DeviceType.AUTO, description="Compute device")
    gpu_id: int | None = Field(default=None, description="GPU ID when using CUDA")
    mixed_precision: bool = Field(default=False, description="Use mixed precision")

    @field_validator("gpu_id")
    @classmethod
    def validate_gpu_id(cls, v, info):
        """Ensure gpu_id is only set when using CUDA."""
        if v is not None and info.data.get("device") != DeviceType.CUDA:
            raise ValueError("gpu_id can only be set when device is 'cuda'")
        return v
