"""Base loader interface for dataset loading."""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self, path: Path) -> list[dict]:
        """Load data from path and return list of items."""
        pass
