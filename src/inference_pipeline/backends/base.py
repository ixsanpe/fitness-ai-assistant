"""Abstract base class for vector search backends."""

from abc import ABC, abstractmethod

import numpy as np


class SearchBackend(ABC):
    """Abstract interface for vector similarity search backends."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding vector (1D numpy array)
            top_k: Number of results to return

        Returns:
            List of result dicts with keys: idx, id, score, combined_text, attributes
        """

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
